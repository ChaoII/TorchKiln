"""ESC-50 数据集（torchkiln 侧，供 PANNs 训练对齐）。

对齐 PaddleSpeech ``paddlespeech/audio/datasets/esc50.py``：
  * 清单 ``meta/esc50.csv``，字段 ``filename, fold, target, category, esc10, src_file, take``
  * ``mode='train'`` → ``fold != split``；``mode='dev'`` → ``fold == split``
  * 音频 ``audio/<filename>``，读取后线性重采样到 ``sample_rate``（与官方 ``feat_type='raw'`` 一致）
  * 返回 ``(waveform(1D float32), label(int64))``
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ESC50Dataset", "esc50_collate"]


def _read_wav(path, target_sr):
    """严格复刻 PaddleSpeech `soundfile_backend.soundfile_load` 的读取口径。

    PaddleSpeech ESC50 用 `load_audio(file, sr=32000)`，其实现是：
      1. `soundfile` 以 **float32** 读取（int16 -> /32768.0，与原样一致）
      2. `resampy.resample(y, src_sr, target_sr, filter='kaiser_fast')` 重采样
      3. `normalize(y, 'linear', 1.0)`：`y / (max(|y|) + 1e-8)`

    ⚠️ 之前用 `wave` + `np.interp` 线性插值重采样，与 PaddleSpeech **不同**
    （线性插值 vs kaiser 窗 sinc），会导致特征有系统差异。已改为同款口径。
    """
    import resampy
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:                      # 多声道 -> 取平均
        y = y.mean(axis=1)
    if sr != target_sr:
        y = resampy.resample(y, sr, target_sr, filter="kaiser_fast")
    y = y / (float(np.max(np.abs(y))) + 1e-8)
    return np.ascontiguousarray(y, dtype=np.float32)


class ESC50Dataset(Dataset):
    """ESC-50 单折（与 PaddleSpeech ESC50 相同的划分口径）。

    ``order_seed``：给定时把样本顺序**按该种子的置换**重排。
    用于与 Paddle 侧做**共享同一随机顺序**的严格对照（PaddleSpeech 的 ESC50 是
    class-clustered CSV 顺序，直接 `shuffle=false` 只能得到 class-clustered 的退化训练）。
    """

    def __init__(self, data_dir, mode="train", split=1, sample_rate=32000,
                 order_seed=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.mode = mode
        meta = os.path.join(data_dir, "ESC-50-master", "meta", "esc50.csv")
        self.items = []
        with open(meta, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fold = int(row["fold"])
                if mode == "train" and fold != split:
                    self.items.append((row["filename"], int(row["target"])))
                elif mode != "train" and fold == split:
                    self.items.append((row["filename"], int(row["target"])))
        if order_seed is not None:
            perm = np.random.RandomState(int(order_seed)).permutation(len(self.items))
            self.items = [self.items[i] for i in perm]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, lab = self.items[idx]
        p = os.path.join(self.data_dir, "ESC-50-master", "audio", fn)
        wav = _read_wav(p, self.sample_rate)
        return torch.from_numpy(wav), torch.tensor(lab, dtype=torch.long)


def esc50_collate(batch):
    """与 PaddleSpeech 一致：**不做 padding**，按 batch 内最短长度截断后 stack。

    （官方 `train.py` 注释就写着 "Need a padding when lengths of waveforms
      differ in a batch"，说明它也没做 padding。）
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return []
    L = min(b[0].numel() for b in batch)
    wav = torch.stack([b[0][:L] for b in batch])
    lab = torch.stack([b[1] for b in batch])
    return [wav, lab]
