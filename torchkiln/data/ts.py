"""Time-series dataset: CSV -> sliding ``(past_target, future_target)`` windows."""
from __future__ import absolute_import

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["TSDataset", "train_collate", "eval_collate"]


def _load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(x) for x in r[1:]] for r in rows[1:]], dtype=np.float32)
    return header[1:], data


class TSDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds = config[mode]["dataset"]
        self.in_chunk_len = int(ds.get("in_chunk_len", 96))
        self.out_chunk_len = int(ds.get("out_chunk_len", 24))
        cols = ds.get("target_cols")
        csv_path = ds["csv_path"]
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(ds.get("data_dir", "."), csv_path) \
                if ds.get("data_dir") else csv_path
        header, data = _load_csv(csv_path)
        if cols:
            idx = [header.index(c) for c in cols]
            data = data[:, idx]
        # split ratios: [train, val] fractions (remainder = test); use train+val window
        ratio = ds.get("split_ratio", [0.7, 0.1, 0.2])
        n = data.shape[0]
        a = int(n * ratio[0])
        b = int(n * (ratio[0] + ratio[1]))
        seg = {"Train": data[:a], "Eval": data[a:b]}.get(mode, data[b:])
        L, H = self.in_chunk_len, self.out_chunk_len
        self.past, self.future = [], []
        for i in range(len(seg) - L - H + 1):
            self.past.append(seg[i:i + L])
            self.future.append(seg[i + L:i + L + H])
        self.dim = data.shape[1]
        if logger is not None:
            logger.info("%s ts dataset: %d windows (L=%d H=%d D=%d)",
                        mode, len(self.past), L, H, self.dim)

    def __len__(self):
        return len(self.past)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        return [self.past[index].astype(np.float32),
                self.future[index].astype(np.float32)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    return [torch.from_numpy(np.stack([s[i] for s in batch], 0)) for i in range(len(batch[0]))]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
