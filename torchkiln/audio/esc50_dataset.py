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
import wave

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ESC50Dataset", "esc50_collate"]


def _read_wav(path, target_sr):
    with wave.open(path, "rb") as w:
        nch, sw, sr, nfr = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        raw = w.readframes(nfr)
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    if sr != target_sr:
        n = int(len(a) * target_sr / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n),
                      np.arange(len(a)), a).astype(np.float32)
    return a.astype(np.float32)


class ESC50Dataset(Dataset):
    """ESC-50 单折（与 PaddleSpeech ESC50 相同的划分口径）。"""

    def __init__(self, data_dir, mode="train", split=1, sample_rate=32000):
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
