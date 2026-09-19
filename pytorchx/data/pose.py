"""Pose data pipeline (YOLO-pose labels).

Label line::

    cls cx cy w h px1 py1 v1 px2 py2 v2 ...    # normalised

Outputs: ``[image, boxes(B,5), valid(B), kpts(B,K,3)]`` in letterboxed pixels.
"""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pytorchx.det.ops import letterbox

__all__ = ["PoseDataset", "train_collate", "eval_collate"]


def _label_path_for(img_rel):
    parts = img_rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return os.path.splitext("/".join(parts))[0] + ".txt"


class PoseDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.imgsz = int((ds_cfg.get("transform") or {}).get("image_size", 640))
        self.kpt_shape = tuple(ds_cfg.get("kpt_shape", (17, 3)))
        # 2 -> (x, y) only (yolov5-face style landmarks), 3 -> (x, y, visibility)
        self.kpt_dim = int(self.kpt_shape[1]) if len(self.kpt_shape) > 1 else 3
        self.augmenter = None
        if ds_cfg.get("augment") and mode == "Train":
            from pytorchx.data.augment import TrainAugmenter

            self.augmenter = TrainAugmenter(ds_cfg["augment"], self.imgsz)
        from pytorchx.data.cache import LabelCache

        self._label_cache = LabelCache(
            self.data_dir, ds_cfg.get("label_file_list"),
            split=mode, logger=logger, enabled=bool(ds_cfg.get("labels_cache", True)),
        )
        self.img_files = []
        for label_file in ds_cfg["label_file_list"]:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.img_files.append(line)
        if logger is not None:
            logger.info(
                "%s dataset: %d images (data_dir=%s, imgsz=%d, kpt_shape=%s)",
                mode, len(self.img_files), self.data_dir, self.imgsz, self.kpt_shape,
            )

    def __len__(self):
        return len(self.img_files)

    def _load(self, img_rel, w, h, ratio, pad):
        """Raw labels in original pixels (``ratio``/``pad`` = (1, 0) for no letterbox)."""
        path = os.path.join(self.data_dir, _label_path_for(img_rel))
        nk = self.kpt_shape[0]
        boxes, kpts, valid = [], [], []
        if not os.path.isfile(path):
            return (
                np.zeros((0, 5), np.float32),
                np.zeros((0, nk, 3), np.float32),
                np.zeros((0,), np.float32),
            )
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5 + nk * self.kpt_dim:
                    continue
                cls = float(p[0])
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                x1 = (cx - bw / 2) * w * ratio + pad[0]
                y1 = (cy - bh / 2) * h * ratio + pad[1]
                x2 = (cx + bw / 2) * w * ratio + pad[0]
                y2 = (cy + bh / 2) * h * ratio + pad[1]
                kp = np.array(
                    p[5 : 5 + nk * self.kpt_dim], dtype=np.float32
                ).reshape(nk, self.kpt_dim)
                if self.kpt_dim == 2:
                    # ultralytics: append a visibility column (1 visible, 0 if x/y<0)
                    vis = np.where(
                        (kp[:, 0] < 0) | (kp[:, 1] < 0), 0.0, 1.0
                    ).astype(np.float32)
                    kp = np.concatenate([kp, vis[:, None]], axis=-1)
                kp[:, 0] = kp[:, 0] * w * ratio + pad[0]
                kp[:, 1] = kp[:, 1] * h * ratio + pad[1]
                boxes.append([cls, x1, y1, x2, y2])
                kpts.append(kp)
                valid.append(1.0)
        return (
            np.asarray(boxes, np.float32) if boxes else np.zeros((0, 5), np.float32),
            np.asarray(kpts, np.float32) if kpts else np.zeros((0, nk, 3), np.float32),
            np.asarray(valid, np.float32),
        )

    def load_raw(self, index):
        """原图 + 原始像素坐标 boxes/kpts(供 ``TrainAugmenter`` 使用)。

        返回 ``(img, boxes(N,5), kpts(N,K,dim), masks=None)``。
        """
        rel = self.img_files[index]
        path = os.path.join(self.data_dir, rel)
        img = cv2.imread(path)
        if img is None:
            return None
        h0, w0 = img.shape[:2]
        boxes, kpts, _valid = self._load(rel, w0, h0, 1.0, (0.0, 0.0))
        return img, boxes, kpts, None

    def set_epoch(self, epoch):
        if self.augmenter is not None:
            self.augmenter.set_epoch(epoch)
        cache = getattr(self, "_label_cache", None)
        if cache is not None and getattr(cache, "_dirty", False):
            cache.save_now()

    def __getitem__(self, index):
        if self.augmenter is not None:
            img, boxes, kpts, _ = self.augmenter(self, index)
            if boxes is None:
                boxes = np.zeros((0, 5), np.float32)
            if kpts is None:
                kpts = np.zeros((boxes.shape[0], self.kpt_shape[0], 3), np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            return [
                img,
                boxes.astype(np.float32),
                np.ones((boxes.shape[0],), np.float32),
                kpts.astype(np.float32),
            ]

        rel = self.img_files[index]
        img = cv2.imread(os.path.join(self.data_dir, rel))
        if img is None:
            return []
        h0, w0 = img.shape[:2]
        img, ratio, pad = letterbox(img, self.imgsz)
        boxes, kpts, valid = self._load(rel, w0, h0, ratio, pad)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return [img, boxes, valid, kpts]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(np.stack([s[0] for s in batch], axis=0)).float()
    nk = batch[0][3].shape[1]
    kd = batch[0][3].shape[2]
    max_gt = max(1, max(s[1].shape[0] for s in batch))
    boxes = np.zeros((len(batch), max_gt, 5), np.float32)
    valid = np.zeros((len(batch), max_gt), np.float32)
    kpts = np.zeros((len(batch), max_gt, nk, kd), np.float32)
    for i, s in enumerate(batch):
        n = s[1].shape[0]
        if n:
            boxes[i, :n] = s[1]
            valid[i, :n] = s[2]
            kpts[i, :n] = s[3]
    return [
        images,
        torch.from_numpy(boxes).float(),
        torch.from_numpy(valid).float(),
        torch.from_numpy(kpts).float(),
    ]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
