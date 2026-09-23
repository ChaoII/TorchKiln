"""Semantic segmentation data pipeline (image + per-pixel mask).

Layout::

    <data_dir>/images/<split>/xxx.jpg
    <data_dir>/masks/<split>/xxx.png     # uint8 class ids
    <list.txt>                           # image paths (relative)
"""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from torchkiln.det.ops import letterbox

__all__ = ["SemDataset", "train_collate", "eval_collate"]


def _mask_path_for(img_rel):
    parts = img_rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "masks"
    stem = os.path.splitext("/".join(parts))[0]
    return stem + ".png"


class SemDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.imgsz = int((ds_cfg.get("transform") or {}).get("image_size", 256))
        self.ignore_index = int(ds_cfg.get("ignore_index", 255))
        self.augmenter = None
        if ds_cfg.get("augment") and mode == "Train":
            from torchkiln.data.augment import DenseAugmenter

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
                "%s dataset: %d images (data_dir=%s, imgsz=%d)",
                mode,
                len(self.img_files),
                self.data_dir,
                self.imgsz,
            )

    def __len__(self):
        return len(self.img_files)

    def set_epoch(self, epoch):
        if self.augmenter is not None:
            self.augmenter.set_epoch(epoch)

    def __getitem__(self, index):
        rel = self.img_files[index]
        img = cv2.imread(os.path.join(self.data_dir, rel))
        if img is None:
            return []
        mask = cv2.imread(
            os.path.join(self.data_dir, _mask_path_for(rel)), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if self.augmenter is not None:
            img, mask = self.augmenter(img, mask)
        else:
            img, ratio, pad = letterbox(img, self.imgsz)
            mask = cv2.resize(
                mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return [img, mask.astype(np.int64)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(np.stack([s[0] for s in batch], axis=0)).float()
    masks = torch.from_numpy(np.stack([s[1] for s in batch], axis=0)).long()
    return [images, masks]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
