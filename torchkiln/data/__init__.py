"""Classification data pipeline (transforms + dataset)."""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ClsTransform", "ClsDataset", "train_collate", "eval_collate"]

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def _parse_image_size(image_size):
    """int -> square (h, w); list/tuple -> (h, w)."""
    if isinstance(image_size, (list, tuple)):
        h, w = int(image_size[0]), int(image_size[1])
        return h, w
    s = int(image_size)
    return s, s


def _resize_cover(img, th, tw):
    """Scale so both sides cover (th, tw), keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = max(float(th) / float(max(h, 1)), float(tw) / float(max(w, 1)))
    return cv2.resize(
        img,
        (max(tw, int(round(w * scale))), max(th, int(round(h * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )


def _resize_short_side(img, size):
    h, w = img.shape[:2]
    scale = float(size) / float(min(h, w))
    return cv2.resize(
        img,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )


def _random_crop(img, th, tw):
    h, w = img.shape[:2]
    if h < th or w < tw:
        img = _resize_cover(img, th, tw)
        h, w = img.shape[:2]
    y0 = np.random.randint(0, max(1, h - th + 1))
    x0 = np.random.randint(0, max(1, w - tw + 1))
    return img[y0 : y0 + th, x0 : x0 + tw]


def _center_crop(img, th, tw):
    h, w = img.shape[:2]
    if h < th or w < tw:
        img = _resize_cover(img, th, tw)
        h, w = img.shape[:2]
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    return img[y0 : y0 + th, x0 : x0 + tw]


class ClsTransform(object):
    """BGR uint8 image -> CHW float32 tensor-ready array (ImageNet stats).

    ``image_size``: ``224`` (square) 或 ``[320, 160]``（h, w，竖长人物）。
    """

    def __init__(self, image_size=224, mean=None, std=None, train=False, erasing=None):
        self.image_size = _parse_image_size(image_size)
        self.mean = np.array(mean or DEFAULT_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std or DEFAULT_STD, dtype=np.float32).reshape(3, 1, 1)
        self.train = bool(train)
        self.erasing = erasing if self.train else None

    def __call__(self, img):
        th, tw = self.image_size
        if self.train:
            img = _random_crop(img, th, tw)
        else:
            img = _center_crop(img, th, tw)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self.mean) / self.std
        if self.erasing is not None:
            img = self.erasing(np.ascontiguousarray(img, dtype=np.float32))
        return np.ascontiguousarray(img, dtype=np.float32)


class ClsDataset(Dataset):
    """Txt-list dataset: one line per image.

    * 单标签: ``relative/path.jpg<TAB|SPACE>class_index``
    * 多标签: ``relative/path.jpg v1 v2 ... vC``（0/1 或 0~1 软标签），
      由 ``Loss.multi_label: true`` 或 ``dataset.multi_label: true`` 打开。
    """

    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        loss_cfg = config.get("Loss") or {}
        self.multi_label = bool(ds_cfg.get("multi_label")) or bool(
            loss_cfg.get("multi_label")
        ) or str(loss_cfg.get("name") or "") in ("MultiLabelLoss", "MultiLabel")
        self.label_ratio = bool(ds_cfg.get("label_ratio", False)) and self.multi_label
        self.data_dir = ds_cfg["data_dir"]
        tf = ds_cfg.get("transform") or {}
        aug = ds_cfg.get("augment") or {}
        erasing = None
        if mode == "Train" and aug.get("erasing"):
            from torchkiln.data.augment import RandomErasing

            spec = aug["erasing"]
            erasing = RandomErasing(**(spec if isinstance(spec, dict) else {}))
        self.transform = ClsTransform(
            image_size=tf.get("image_size", 224),
            mean=tf.get("mean"),
            std=tf.get("std"),
            train=(mode == "Train"),
            erasing=erasing,
        )
        self.samples = []
        for label_file in ds_cfg["label_file_list"]:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    if self.multi_label:
                        # 支持 "p.jpg 1 0 1" 与 "p.jpg\t1,0,1" 两种写法
                        if len(parts) == 2 and ("," in parts[1] or "\t" in line):
                            vals = parts[1].replace("\t", ",").split(",")
                        else:
                            vals = parts[1:]
                        self.samples.append(
                            (parts[0], np.array([float(v) for v in vals if v != ""], np.float32))
                        )
                    else:
                        self.samples.append((parts[0], int(float(parts[1].split(",")[0]))))
        self.ratio = None
        if self.label_ratio and self.samples:
            num_classes = max(len(lb) for _, lb in self.samples)
            ratio = np.zeros(num_classes, np.float32)
            for _, lb in self.samples:
                ratio[: len(lb)] += (lb > 0.5).astype(np.float32)
            self.ratio = np.maximum(ratio / len(self.samples), 1e-6).astype(np.float32)
        if logger is not None:
            logger.info(
                "%s dataset: %d samples (data_dir=%s, multi_label=%s)",
                mode,
                len(self.samples),
                self.data_dir,
                self.multi_label,
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel, label = self.samples[index]
        img = cv2.imread(os.path.join(self.data_dir, rel))
        if img is None:
            return []
        img = self.transform(img)
        if self.multi_label:
            out = [img, label.astype(np.float32)]
            if self.ratio is not None:
                out.append(self.ratio.copy())
            return out
        return [img, np.int64(label)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(
        np.stack([s[0] for s in batch], axis=0)
    ).float()
    first_label = batch[0][1]
    if isinstance(first_label, np.ndarray):
        # multi-label: float vector (optional ratio as third item)
        c = first_label.shape[0]
        labels = torch.zeros(len(batch), c, dtype=torch.float32)
        for i, s in enumerate(batch):
            labels[i] = torch.from_numpy(s[1][:c])
        if len(batch[0]) > 2:
            ratios = torch.zeros(len(batch), c, dtype=torch.float32)
            for i, s in enumerate(batch):
                ratios[i] = torch.from_numpy(s[2][:c])
            return [images, labels, ratios]
        return [images, labels]
    labels = torch.from_numpy(
        np.stack([np.int64(s[1]) for s in batch], axis=0)
    ).long()
    return [images, labels]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
