"""Time-series dataset: CSV -> sliding windows with optional covariates.

Each window yields ``[past_target, future_target, known_cov, observed_cov,
static_cov]`` when any covariate column is configured, otherwise just
``[past_target, future_target]``.

Column roles (config under ``<mode>.dataset``):
  * ``target_cols``   : target series columns (defaults to all columns)
  * ``known_cols``    : known covariates, present over the whole (past+future) horizon
  * ``observed_cols`` : observed covariates, past horizon only
  * ``static_cols``   : static covariates (constant per series)
"""
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
        target_cols = ds.get("target_cols")
        known_cols = ds.get("known_cols")
        observed_cols = ds.get("observed_cols")
        static_cols = ds.get("static_cols")
        csv_path = ds["csv_path"]
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(ds.get("data_dir", "."), csv_path) \
                if ds.get("data_dir") else csv_path
        header, full = _load_csv(csv_path)

        def pick(names):
            if not names:
                return None
            idx = [header.index(c) for c in names]
            return full[:, idx]

        target = pick(target_cols) if target_cols else full
        known, observed, static = pick(known_cols), pick(observed_cols), pick(static_cols)
        self.has_cov = any(x is not None for x in (known, observed, static))

        # split ratios: [train, val] fractions (remainder = test)
        ratio = ds.get("split_ratio", [0.7, 0.1, 0.2])
        n = target.shape[0]
        a = int(n * ratio[0])
        b = int(n * (ratio[0] + ratio[1]))
        sl = {"Train": slice(0, a), "Eval": slice(a, b)}.get(mode, slice(b, None))

        target = target[sl]
        known = known[sl] if known is not None else None
        observed = observed[sl] if observed is not None else None
        static = static[sl][:1] if static is not None else None

        L, H = self.in_chunk_len, self.out_chunk_len
        self.past, self.future = [], []
        self.known, self.observed, self.static = [], [], []
        for i in range(len(target) - L - H + 1):
            self.past.append(target[i:i + L])
            self.future.append(target[i + L:i + L + H])
            if self.has_cov:
                self.known.append(known[i:i + L + H] if known is not None
                                  else np.zeros((L + H, 0), np.float32))
                self.observed.append(observed[i:i + L] if observed is not None
                                     else np.zeros((L, 0), np.float32))
                self.static.append(static if static is not None
                                   else np.zeros((1, 0), np.float32))
        self.dim = target.shape[1]
        if logger is not None:
            logger.info("%s ts dataset: %d windows (L=%d H=%d D=%d cov=%s)",
                        mode, len(self.past), L, H, self.dim, self.has_cov)

    def __len__(self):
        return len(self.past)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        items = [self.past[index].astype(np.float32),
                 self.future[index].astype(np.float32)]
        if self.has_cov:
            items.append(self.known[index].astype(np.float32))
            items.append(self.observed[index].astype(np.float32))
            items.append(self.static[index].astype(np.float32))
        return items


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    return [torch.from_numpy(np.stack([s[i] for s in batch], 0)) for i in range(len(batch[0]))]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
