"""Detection data pipeline: YOLO-txt labels + letterbox + padding collate.

Expected layout::

    <data_dir>/images/<split>/xxx.jpg
    <data_dir>/labels/<split>/xxx.txt      # "cls cx cy w h" normalised
    <list.txt>                             # one image path per line (relative)

``<list.txt>`` may also be a plain list of images; labels are found by
replacing the ``images`` path component with ``labels`` and the extension
with ``.txt``.
"""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from torchkiln.det.ops import letterbox
from torchkiln.det.rbox import poly2rbox

__all__ = ["DetDataset", "train_collate", "eval_collate"]


def _label_path_for(img_rel):
    parts = img_rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    stem = os.path.splitext("/".join(parts))[0]
    return stem + ".txt"


class DetDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.imgsz = int((ds_cfg.get("transform") or {}).get("image_size", 640))
        self.box_format = ds_cfg.get("box_format", "xyxy")
        self.names = ds_cfg.get("names")
        self.augmenter = None
        aug_cfg = ds_cfg.get("augment")
        if aug_cfg and mode == "Train":
            from torchkiln.data.augment import TrainAugmenter

            self.augmenter = TrainAugmenter(
                aug_cfg, self.imgsz, box_format=self.box_format
            )
        self.num_classes = ds_cfg.get("num_classes") or (
            len(self.names) if self.names else None
        )
        from torchkiln.data.cache import LabelCache

        self._label_cache = LabelCache(
            self.data_dir, ds_cfg.get("label_file_list"),
            split=mode, logger=logger, enabled=bool(ds_cfg.get("labels_cache", True)),
        )
        self.img_files = []
        for label_file in ds_cfg["label_file_list"]:
            label_file = os.path.join(self.data_dir, label_file) if not os.path.isabs(
                label_file
            ) and not os.path.isfile(label_file) else label_file
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.img_files.append(line)
        if logger is not None:
            logger.info(
                "%s dataset: %d images (data_dir=%s, imgsz=%d)",
                mode,
                len(self.img_files),
                self.data_dir,
                self.imgsz,
            )

    def __len__(self):
        return len(self.img_files)

    def _load_raw_labels(self, img_rel, w, h):
        """Labels in the *original* image pixels: ``(N,5)`` xyxy or ``(N,6)`` xywhr."""
        path = os.path.join(self.data_dir, _label_path_for(img_rel))
        width = 6 if self.box_format == "xywhr" else 5
        if not os.path.isfile(path):
            return np.zeros((0, width), dtype=np.float32)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = float(parts[0])
                if self.box_format == "xywhr":
                    if len(parts) < 9:
                        continue
                    pts = np.array(parts[1:9], dtype=np.float64).reshape(4, 2)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    cx, cy, bw, bh, theta = poly2rbox(pts)
                    rows.append([cls, cx, cy, bw, bh, float(theta)])
                else:
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    rows.append(
                        [
                            cls,
                            (cx - bw / 2) * w,
                            (cy - bh / 2) * h,
                            (cx + bw / 2) * w,
                            (cy + bh / 2) * h,
                        ]
                    )
        if not rows:
            return np.zeros((0, width), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _letterbox_labels(self, labels, ratio, pad):
        if len(labels) == 0:
            return labels
        out = labels.copy()
        if self.box_format == "xywhr":
            out[:, 1] = out[:, 1] * ratio + pad[0]
            out[:, 2] = out[:, 2] * ratio + pad[1]
            out[:, 3] = out[:, 3] * ratio
            out[:, 4] = out[:, 4] * ratio
        else:
            out[:, 1] = out[:, 1] * ratio + pad[0]
            out[:, 3] = out[:, 3] * ratio + pad[0]
            out[:, 2] = out[:, 2] * ratio + pad[1]
            out[:, 4] = out[:, 4] * ratio + pad[1]
        return out

    def load_raw(self, index):
        """原图 + 原始像素坐标标签(供 ``TrainAugmenter`` 使用)。

        返回 ``(img, labels(N,5/6), kpts=None, masks=None)``,坐标系为原图像素。
        """
        import cv2

        rel = self.img_files[index]
        path = os.path.join(self.data_dir, rel)
        img = cv2.imread(path)
        if img is None:
            return None
        h0, w0 = img.shape[:2]
        labels = self._load_raw_labels(rel, w0, h0)
        return img, labels, None, None

    def set_epoch(self, epoch):
        if self.augmenter is not None:
            self.augmenter.set_epoch(epoch)
        cache = getattr(self, "_label_cache", None)
        if cache is not None and getattr(cache, "_dirty", False):
            cache.save_now()

    def __getitem__(self, index):
        if self.augmenter is not None:
            img, labels, _, _ = self.augmenter(self, index)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            if labels is None:
                labels = np.zeros((0, 6 if self.box_format == "xywhr" else 5), np.float32)
            return [
                img,
                labels.astype(np.float32),
                np.ones((labels.shape[0],), dtype=np.float32),
            ]

        rel = self.img_files[index]
        img = cv2.imread(os.path.join(self.data_dir, rel))
        if img is None:
            return []
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        img, ratio, pad = letterbox(img, self.imgsz, color=(114, 114, 114))
        labels = self._letterbox_labels(self._load_raw_labels(rel, w, h), ratio, pad)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        mask = np.ones((labels.shape[0],), dtype=np.float32)
        return [img, labels, mask]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(
        np.stack([s[0] for s in batch], axis=0)
    ).float()
    tw = max(s[1].shape[1] for s in batch)
    max_gt = max(1, max(s[1].shape[0] for s in batch))
    targets = np.zeros((len(batch), max_gt, tw), dtype=np.float32)
    masks = np.zeros((len(batch), max_gt), dtype=np.float32)
    for i, s in enumerate(batch):
        n = s[1].shape[0]
        if n:
            targets[i, :n] = s[1]
            masks[i, :n] = s[2]
    return [
        images,
        torch.from_numpy(targets).float(),
        torch.from_numpy(masks).float(),
    ]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
