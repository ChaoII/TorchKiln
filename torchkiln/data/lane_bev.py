"""BEV-LaneDet data: camera image + BEV lane GT (segment/instance/offset/z)."""
from __future__ import absolute_import

import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["LaneBEVDataset", "train_collate", "eval_collate"]

_KEYS = ["bev_seg", "bev_inst", "bev_off", "bev_z", "img_seg", "img_inst"]


def _load_image(path, size):
    if path.endswith(".npy"):
        img = np.load(path).astype(np.float32)  # (3, H, W)
    else:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)
    if size is not None and img.shape[-2:] != tuple(size):
        import cv2

        img = np.stack([cv2.resize(img[c], (size[1], size[0])) for c in range(img.shape[0])])
    return img.astype(np.float32)


class LaneBEVDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds = config[mode]["dataset"]
        self.data_dir = ds["data_dir"]
        self.size = ds.get("input_shape")
        self.gt_dir = ds.get("gt_dir", "bev_gt")
        self.files = []
        for lf in ds["label_file_list"]:
            p = lf if os.path.isabs(lf) or os.path.isfile(lf) else os.path.join(self.data_dir, lf)
            with open(p, encoding="utf-8") as f:
                self.files += [ln.strip() for ln in f if ln.strip()]
        if logger is not None:
            logger.info("%s lane_bev dataset: %d samples (data_dir=%s)",
                        mode, len(self.files), self.data_dir)

    def __len__(self):
        return len(self.files)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        rel = self.files[index]
        img = _load_image(os.path.join(self.data_dir, rel), self.size)
        stem = os.path.splitext(os.path.basename(rel))[0]
        gt_path = os.path.join(self.data_dir, self.gt_dir, stem + ".npz")
        if not os.path.isfile(gt_path):
            gt_path = os.path.join(self.data_dir, os.path.splitext(rel)[0] + ".npz")
        d = np.load(gt_path)
        out = [img]
        for k in _KEYS:
            v = d[k].astype(np.float32)
            if v.ndim == 2:
                v = v[None]
            out.append(v)
        return out


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if not batch:
        return []
    return [torch.from_numpy(np.stack([s[i] for s in batch], 0)) for i in range(len(batch[0]))]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
