"""Plate recognition data pipeline (upstream ``myNet_ocr_color`` style).

Label line (whitespace separated)::

    relative/or/abs/path.jpg  c1 c2 ... c7  [color]

``c*`` are indices into :data:`pytorchx.nn.plate.PLATE_CHARSET` (0 = the CTC
blank ``#``), ``color`` is an index into ``PLATE_COLORS`` (optional, default 0).
Multiple plate crops per image are allowed (concatenated).

Pre-processing is exactly upstream ``plate_rec.py``::

    resize(168, 48) -> /255 -> (x - 0.588) / 0.193 -> CHW
"""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pytorchx.nn.plate import PLATE_CHARSET, PLATE_COLORS

__all__ = ["PlateRecDataset", "split_merge_double_plate", "train_collate", "eval_collate"]

MEAN_VALUE = 0.588
STD_VALUE = 0.193
REC_SIZE = (168, 48)  # (w, h)


def split_merge_double_plate(img):
    """Flatten a double-layer plate into a single line (upstream ``get_split_merge``)."""
    h, w, c = img.shape
    upper = img[0 : int(5 / 12 * h), :]
    lower = img[int(1 / 3 * h) :, :]
    upper = cv2.resize(upper, (lower.shape[1], lower.shape[0]))
    out = np.full((lower.shape[0], lower.shape[1] + upper.shape[1], 3), 114, np.uint8)
    out[:, : upper.shape[1]] = upper
    out[:, upper.shape[1] :] = lower
    return out


def plate_image_processing(img, double_plate=False):
    if double_plate:
        img = split_merge_double_plate(img)
    img = cv2.resize(img, REC_SIZE)
    img = img.astype(np.float32)
    img = (img / 255.0 - MEAN_VALUE) / STD_VALUE
    img = img.transpose(2, 0, 1)
    return np.ascontiguousarray(img, dtype=np.float32)


class PlateRecDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg.get("data_dir", "")
        self.double_plate = bool(ds_cfg.get("double_plate", False))
        self.samples = []
        for label_file in ds_cfg["label_file_list"]:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    self.samples.append(parts)
        if logger is not None:
            logger.info(
                "%s dataset: %d plate crops (data_dir=%s, charset=%d, colors=%d)",
                mode,
                len(self.samples),
                self.data_dir,
                len(PLATE_CHARSET),
                len(PLATE_COLORS),
            )

    def __len__(self):
        return len(self.samples)

    def _image_path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.data_dir, rel)

    def __getitem__(self, index):
        parts = self.samples[index]
        img = cv2.imread(self._image_path(parts[0]))
        if img is None:
            return []
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        label = np.array([int(v) for v in parts[1:]], dtype=np.int64)
        if len(label) > 1:
            chars, color = label[:-1], int(label[-1])
        else:
            chars, color = label, 0
        img = plate_image_processing(img, self.double_plate)
        return [img, chars, np.int64(color)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(np.stack([s[0] for s in batch], axis=0)).float()
    targets = torch.from_numpy(np.concatenate([s[1] for s in batch], axis=0)).long()
    lengths = torch.tensor([len(s[1]) for s in batch], dtype=torch.long)
    colors = torch.from_numpy(np.stack([s[2] for s in batch], axis=0)).long()
    return [images, targets, lengths, colors]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
