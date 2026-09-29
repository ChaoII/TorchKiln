"""时序分类数据集：CSV -> 滑窗 + 序列级标签。

CSV 约定（两种之一）：
  1. **长格式**（默认）：``timestamp, f1, f2, ..., label``；
     每 ``in_chunk_len`` 个连续点构成一个样本，标签取窗口**最后一个时刻**的 label；
     再按 ``label_mode='majority'``（默认）对窗口内标签取多数（对逐步标注的故障类型更稳健）。
  2. **清单格式**：``sequence_path<TAB>label`` —— 每行一个 ``.npy``/``.csv``/``.txt`` 序列文件。

与框架衔接：样本返回 ``[feat(L,C), label]``。
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["TSClassifyDataset", "ts_classify_collate"]


def _load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(x) if x != "" else np.nan for x in r[1:]] for r in rows[1:]],
                    dtype=np.float64)
    return header[1:], data


class TSClassifyDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds = config[mode]["dataset"]
        self.in_chunk_len = int(ds.get("in_chunk_len", 100))
        self.stride = int(ds.get("stride", 1))
        self.label_mode = ds.get("label_mode", "majority")
        # ---- 清单格式 ----
        manifest = ds.get("manifest")
        if manifest:
            self.feat, self.lab = [], []
            with open(manifest, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p, y = line.split("\t")[:2]
                    if not os.path.isabs(p):
                        p = os.path.join(ds.get("data_dir", "."), p)
                    arr = (np.load(p) if p.endswith(".npy")
                           else np.loadtxt(p, delimiter=",", skiprows=1))
                    arr = np.asarray(arr, dtype=np.float32)
                    self.feat.append(arr[:self.in_chunk_len])
                    self.lab.append(int(y))
            self.dim = self.feat[0].shape[1] if self.feat else 0
            if logger:
                logger.info("%s ts_classify(manifest): %d seqs (L=%d C=%d)",
                            mode, len(self.feat), self.in_chunk_len, self.dim)
            return
        # ---- 长格式 ----
        csv_path = ds["csv_path"]
        if not os.path.isabs(csv_path) and ds.get("data_dir"):
            csv_path = os.path.join(ds["data_dir"], csv_path)
        label_col = ds.get("label_col", "label")
        cols, data = _load_csv(csv_path)
        # ⚠️ 表示学习(无标签)场景：label_col 不存在时**不报错**，标签全 0（不参与监督）
        if label_col in cols:
            li = cols.index(label_col)
            lab = data[:, li].astype(np.int64)
            data = np.delete(data, li, axis=1)
            cols = [c for i, c in enumerate(cols) if i != li]
        else:
            lab = np.zeros(data.shape[0], dtype=np.int64)
        feature_cols = ds.get("feature_cols")
        if feature_cols:
            idx = [cols.index(c) for c in feature_cols]
            data = data[:, idx]

        ratio = ds.get("split_ratio", [0.6, 0.2, 0.2])
        n = data.shape[0]
        a = int(n * ratio[0])
        b = int(n * (ratio[0] + ratio[1]))
        sl = {"Train": slice(0, a), "Eval": slice(a, b)}.get(mode, slice(b, None))
        data, lab = data[sl], lab[sl]

        L = self.in_chunk_len
        self.feat, self.lab = [], []
        for i in range(0, len(data) - L + 1, self.stride):
            w = np.nan_to_num(data[i:i + L], nan=0.0).astype(np.float32)
            wl = lab[i:i + L]
            y = int(np.bincount(wl).argmax()) if self.label_mode == "majority" else int(wl[-1])
            self.feat.append(w)
            self.lab.append(y)
        self.dim = data.shape[1]
        if logger:
            logger.info("%s ts_classify: %d windows (L=%d C=%d n_class=%d)",
                        mode, len(self.feat), L, self.dim,
                        len(np.unique(self.lab)) if self.lab else 0)

    def __len__(self):
        return len(self.feat)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        return [self.feat[index], np.int64(self.lab[index])]


def ts_classify_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return []
    return [torch.from_numpy(np.stack([b[0] for b in batch], 0)),
            torch.from_numpy(np.stack([b[1] for b in batch], 0))]
