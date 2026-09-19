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


def _resize_short_side(img, size):
    h, w = img.shape[:2]
    scale = float(size) / float(min(h, w))
    return cv2.resize(
        img,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )


def _random_crop(img, size):
    h, w = img.shape[:2]
    if h < size or w < size:
        img = _resize_short_side(img, size)
        h, w = img.shape[:2]
    y0 = np.random.randint(0, max(1, h - size + 1))
    x0 = np.random.randint(0, max(1, w - size + 1))
    return img[y0 : y0 + size, x0 : x0 + size]


def _center_crop(img, size):
    h, w = img.shape[:2]
    if h < size or w < size:
        img = _resize_short_side(img, size)
        h, w = img.shape[:2]
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return img[y0 : y0 + size, x0 : x0 + size]


class ClsTransform(object):
    """BGR uint8 image -> CHW float32 tensor-ready array (ImageNet stats)."""

    def __init__(self, image_size=224, mean=None, std=None, train=False, erasing=None):
        self.image_size = int(image_size)
        self.mean = np.array(mean or DEFAULT_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std or DEFAULT_STD, dtype=np.float32).reshape(3, 1, 1)
        self.train = bool(train)
        self.erasing = erasing if self.train else None

    def __call__(self, img):
        if self.train:
            img = _random_crop(img, self.image_size)
        else:
            img = _resize_short_side(img, self.image_size)
            img = _center_crop(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self.mean) / self.std
        if self.erasing is not None:
            img = self.erasing(np.ascontiguousarray(img, dtype=np.float32))
        return np.ascontiguousarray(img, dtype=np.float32)


class ClsDataset(Dataset):
    """Txt-list dataset: one ``relative/path.jpg<TAB|SPACE>class_index`` per line."""

    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.data_dir = ds_cfg["data_dir"]
        tf = ds_cfg.get("transform") or {}
        aug = ds_cfg.get("augment") or {}
        erasing = None
        if mode == "Train" and aug.get("erasing"):
            from pytorchx.data.augment import RandomErasing

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
                    self.samples.append((parts[0], int(parts[1])))
        if logger is not None:
            logger.info(
                "%s dataset: %d samples (data_dir=%s)",
                mode,
                len(self.samples),
                self.data_dir,
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel, label = self.samples[index]
        img = cv2.imread(os.path.join(self.data_dir, rel))
        if img is None:
            return []
        img = self.transform(img)
        return [img, np.int64(label)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(
        np.stack([s[0] for s in batch], axis=0)
    ).float()
    labels = torch.from_numpy(
        np.stack([np.int64(s[1]) for s in batch], axis=0)
    ).long()
    return [images, labels]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
