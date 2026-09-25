"""Row-based lane detection data (UFLD-style).

Layout::

    <data_dir>/images/<split>/xxx.jpg
    <data_dir>/labels/<split>/xxx.txt
    <list.txt>

Label line format (one lane per line, fixed ``num_lanes`` lines)::

    valid x0 x1 ... x_{R-1}     # valid in {0,1}; x in [0,1]; R = num_rows
"""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from torchkiln.det.ops import letterbox

__all__ = ["LaneRowDataset", "train_collate", "eval_collate"]


def _label_path_for(img_rel):
    parts = img_rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    stem = os.path.splitext("/".join(parts))[0]
    return stem + ".txt"


def _parse_label(path, num_lanes, num_rows):
    xs = np.full((num_lanes, num_rows), -1.0, dtype=np.float32)
    valid = np.zeros((num_lanes, num_rows), dtype=bool)
    if not os.path.isfile(path):
        return xs, valid
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for li, line in enumerate(lines[:num_lanes]):
        parts = line.split()
        if not parts:
            continue
        try:
            flag = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if flag != 1:
            continue
        n = min(len(coords), num_rows)
        for ri in range(n):
            xs[li, ri] = coords[ri]
            valid[li, ri] = True
    return xs, valid


class LaneRowDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.imgsz = int((ds_cfg.get("transform") or {}).get("image_size", 256))
        self.num_lanes = int(ds_cfg.get("num_lanes", 6))
        self.num_rows = int(ds_cfg.get("num_rows", 100))
        self.augmenter = None
        if ds_cfg.get("augment") and mode == "Train":
            from torchkiln.data.augment import DenseAugmenter

            self.augmenter = DenseAugmenter(ds_cfg["augment"], self.imgsz)
        self.img_files = []
        for label_file in ds_cfg["label_file_list"]:
            path = label_file
            if not os.path.isabs(path) and not os.path.isfile(path):
                path = os.path.join(self.data_dir, label_file)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.img_files.append(line)
        if logger is not None:
            logger.info(
                "%s lane_row dataset: %d images (L=%d R=%d imgsz=%d)",
                mode,
                len(self.img_files),
                self.num_lanes,
                self.num_rows,
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
        xs, valid = _parse_label(
            os.path.join(self.data_dir, _label_path_for(rel)),
            self.num_lanes,
            self.num_rows,
        )
        if self.augmenter is not None:
            # geometric augs that flip horizontally need to flip xs
            img2, _ = self.augmenter(img, np.zeros(img.shape[:2], np.uint8))
            img = img2
        else:
            img, ratio, pad = letterbox(img, self.imgsz)
        # Horizontal flip sync (simple, when no augmenter): not applied by default
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return [img, xs, valid]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(np.stack([s[0] for s in batch], 0)).float()
    xs = torch.from_numpy(np.stack([s[1] for s in batch], 0)).float()
    valid = torch.from_numpy(np.stack([s[2] for s in batch], 0)).bool()
    return [images, xs, valid]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
