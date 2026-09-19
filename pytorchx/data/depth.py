"""Depth data pipeline (image + depth map).

Layout::

    <data_dir>/images/<split>/xxx.jpg
    <data_dir>/depth/<split>/xxx.png      # uint16, millimetres by default
    <list.txt>                            # image paths (relative)

Depth is stored as ``uint16`` PNG with values in ``depth_scale`` units per metre
(default 1000 -> millimetres). ``0`` marks invalid pixels.
"""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pytorchx.det.ops import letterbox

__all__ = ["DepthDataset", "train_collate", "eval_collate"]


def _depth_path_for(img_rel):
    parts = img_rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "depth"
    stem = os.path.splitext("/".join(parts))[0]
    return stem + ".png"


class DepthDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.imgsz = int((ds_cfg.get("transform") or {}).get("image_size", 320))
        self.depth_scale = float(ds_cfg.get("depth_scale", 1000.0))
        self.augmenter = None
        if ds_cfg.get("augment") and mode == "Train":
            from pytorchx.data.augment import DenseAugmenter

            self.augmenter = DenseAugmenter(ds_cfg["augment"], self.imgsz)
        self.img_files = []
        for label_file in ds_cfg["label_file_list"]:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.img_files.append(line)
        if logger is not None:
            logger.info(
                "%s dataset: %d images (data_dir=%s, imgsz=%d, depth_scale=%g)",
                mode,
                len(self.img_files),
                self.data_dir,
                self.imgsz,
                self.depth_scale,
            )

    def __len__(self):
        return len(self.img_files)

    def _load_depth(self, rel, size):
        path = os.path.join(self.data_dir, _depth_path_for(rel))
        if path.endswith(".npy"):
            depth = np.load(path).astype(np.float32)
        else:
            d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if d is None:
                return np.zeros(size, dtype=np.float32)
            depth = d.astype(np.float32) / self.depth_scale
        depth = cv2.resize(depth, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return depth

    def set_epoch(self, epoch):
        if self.augmenter is not None:
            self.augmenter.set_epoch(epoch)

    def __getitem__(self, index):
        rel = self.img_files[index]
        img = cv2.imread(os.path.join(self.data_dir, rel))
        if img is None:
            return []
        if self.augmenter is not None:
            h, w = img.shape[:2]
            depth = self._load_depth(rel, (h, w))
            img, depth = self.augmenter(img, depth)
        else:
            img, ratio, pad = letterbox(img, self.imgsz)
            depth = self._load_depth(rel, img.shape[:2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return [img, depth[None, :, :].astype(np.float32)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(np.stack([s[0] for s in batch], axis=0)).float()
    depth = torch.from_numpy(np.stack([s[1] for s in batch], axis=0)).float()
    return [images, depth]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
