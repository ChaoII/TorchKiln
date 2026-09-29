"""时序异常检测数据集：CSV -> 滑窗（``observed_cov_numeric`` + 点级标签）。

CSV 约定：
    ``timestamp, f1, f2, ..., label``
  * 首列为时间列（忽略）；``label_col``（默认 ``label``）为**点级**异常标签（0/1）；
    **训练集可以没有 label 列**（无监督），此时标签记 0。
  * ``feature_cols`` 不填时 = 除时间列/label 外的全部列。

与框架其它 TS 任务一致：按 ``split_ratio`` 切 Train/Eval/Test；每样本返回
``[feat(L,C), label]``，其中 ``label`` 取**窗口最后一个时刻**的点级标签
（异常检测的逐点判定口径）。

``standardize: true`` 时用 **Train 段**统计量做 z-score（避免泄漏）。
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["TSAnomalyDataset", "anomaly_collate"]


def _load_csv(path, label_col):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    cols = header[1:]
    data = np.array([[float(x) if x != "" else np.nan for x in r[1:]] for r in rows[1:]],
                    dtype=np.float64)
    has_label = label_col in cols
    lab = None
    if has_label:
        li = cols.index(label_col)
        lab = data[:, li].astype(np.int64)
        cols = [c for i, c in enumerate(cols) if i != li]
        data = np.delete(data, li, axis=1)
    return cols, data, lab, has_label


class TSAnomalyDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds = config[mode]["dataset"]
        csv_path = ds["csv_path"]
        if not os.path.isabs(csv_path) and ds.get("data_dir"):
            csv_path = os.path.join(ds["data_dir"], csv_path)
        self.in_chunk_len = int(ds.get("in_chunk_len", 100))
        self.stride = int(ds.get("stride", ds.get("sampling_stride", 1)))
        label_col = ds.get("label_col", "label")
        cols, data, lab, has_label = _load_csv(csv_path, label_col)
        feature_cols = ds.get("feature_cols")
        if feature_cols:
            idx = [cols.index(c) for c in feature_cols]
            cols = [cols[i] for i in idx]
            data = data[:, idx]

        ratio = ds.get("split_ratio", [0.6, 0.2, 0.2])
        n = data.shape[0]
        a = int(n * ratio[0])
        b = int(n * (ratio[0] + ratio[1]))
        sl = {"Train": slice(0, a), "Eval": slice(a, b)}.get(mode, slice(b, None))

        # ⚠️ 标准化统计量**只用 Train 段**（避免用未来数据 / 泄漏）
        if ds.get("standardize", True):
            mu = np.nanmean(data[0:a], axis=0)
            sd = np.nanstd(data[0:a], axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            data = (data - mu) / sd
        self._mu, self._sd = (mu if ds.get("standardize", True) else None,
                              sd if ds.get("standardize", True) else None)

        data = data[sl]
        lab = lab[sl] if lab is not None else np.zeros(len(data), dtype=np.int64)

        L = self.in_chunk_len
        self.feat, self.lab = [], []
        for i in range(0, len(data) - L + 1, self.stride):
            self.feat.append(np.nan_to_num(data[i:i + L], nan=0.0).astype(np.float32))
            self.lab.append(int(lab[i + L - 1]))
        self.dim = data.shape[1]
        self.has_label = has_label
        if logger is not None:
            logger.info("%s ts_anomaly dataset: %d windows (L=%d C=%d label=%s)",
                        mode, len(self.feat), L, self.dim, has_label)

    def __len__(self):
        return len(self.feat)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        return [self.feat[index], np.int64(self.lab[index])]


def anomaly_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return []
    feat = torch.from_numpy(np.stack([b[0] for b in batch], 0))
    lab = torch.from_numpy(np.stack([b[1] for b in batch], 0))
    return [feat, lab]
