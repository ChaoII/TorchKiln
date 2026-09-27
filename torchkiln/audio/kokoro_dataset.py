"""kokoro TTS 训练的数据集（自写）。

清单格式（复用 torchkiln 的 `SimpleDataSet` 习惯：每行 ``<音频相对路径>\\t<文本/音素>``）：
    wav/0001.wav<TAB>hello world
    wav/0002.wav<TAB>你好

返回：``(input_ids(int64[T]), ref_s(float32[256]), wav(float32[L]))``
  * ``input_ids`` 由**音素串**经 ``cfg['vocab']`` 映射得到；若给的是原始文本，
    需外部先 G2P（misaki/paddlespeech 均可），本类**只做 vocab 映射**，保持依赖最小。
  * ``ref_s`` 若无参考音色（voice pack）则用**固定可复现的随机向量**（训练期作 style 输入）。
  * ``wav`` 目标波形（24kHz）。
"""
from __future__ import annotations

import json
import os
import wave

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["KokoroDataset", "kokoro_collate"]


def _read_wav(path, target_sr=24000, max_len=None):
    """stdlib 读 wav（不依赖 soundfile）+ 线性重采样 + 可选截断。"""
    with wave.open(path, "rb") as w:
        nch, sw, sr, nfr = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        raw = w.readframes(nfr)
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError("unsupported sample width %d" % sw)
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    if sr != target_sr:
        n = int(len(a) * target_sr / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n),
                      np.arange(len(a)), a).astype(np.float32)
    if max_len:
        a = a[:max_len]
    return a.astype(np.float32)


class KokoroDataset(Dataset):
    """按清单读 (音频, 音素串) 并编码为 ``input_ids``。"""

    def __init__(self, list_file, data_dir="", config_path=None,
                 sr=24000, max_len=24000 * 6, delimiter="\t",
                 allow_text_fallback=True):
        self.data_dir = data_dir
        self.sr, self.max_len = sr, max_len
        self.delimiter = delimiter
        self.allow_text_fallback = allow_text_fallback
        cfg = json.load(open(config_path, encoding="utf-8")) if config_path else {}
        self.vocab = cfg.get("vocab", {})
        self.n_token = cfg.get("n_token", 178)
        self.lines = []
        with open(list_file, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    self.lines.append(ln)

    def __len__(self):
        return len(self.lines)

    def _encode(self, text):
        """音素串 -> id 序列（保持 vocab 顺序；未命中=丢弃，与官方 pipeline 一致）。"""
        ids = []
        for ch in text:
            v = self.vocab.get(ch)
            if v is not None:
                ids.append(int(v))
        if not ids and self.allow_text_fallback:
            # 退化：按字符 ord 取模（仅保证可跑，不具语义）
            ids = [1 + (ord(c) % (self.n_token - 2)) for c in text[:32]]
        return [0] + ids + [0]

    def __getitem__(self, idx):
        parts = self.lines[idx].split(self.delimiter, 1)
        rel = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ""
        wav_path = rel if os.path.isabs(rel) else os.path.join(self.data_dir, rel)
        wav = _read_wav(wav_path, self.sr, self.max_len)
        ids = torch.tensor(self._encode(text), dtype=torch.long)
        # ref_s: 无参考音色时用**确定性**随机（按 idx 播种，保证可复现）
        g = torch.Generator().manual_seed(1000 + idx)
        ref_s = torch.randn(256, generator=g).float()
        return ids, ref_s, torch.from_numpy(wav)


def kokoro_collate(batch):
    """pad input_ids；wav 取同长（按 batch 内最短截断，保证可 stack）。

    返回 **list**（trainer 约定 ``batch[0]`` 为 images/input_ids）：
        ``[input_ids, ref_s, wav]``
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return []
    ids = [b[0] for b in batch]
    L = max(x.numel() for x in ids)
    ids_p = torch.zeros(len(ids), L, dtype=torch.long)
    for i, x in enumerate(ids):
        ids_p[i, : x.numel()] = x
    ref_s = torch.stack([b[1] for b in batch])
    Lw = min(b[2].numel() for b in batch)
    wav = torch.stack([b[2][:Lw] for b in batch])
    return [ids_p, ref_s, wav]
