"""CoST 表示学习（移植自 paddlets ``models/representation/dl/_cost``）。

对应关系：
  * ``encoder.py`` 的 ``SamePadConv``/``ConvLayer``/``DilatedConvLayer`` 与
    :mod:`torchkiln.nn.ts_ts2vec` **完全同构** ⇒ 直接复用（避免重复实现）。
  * :class:`TFDLayer` / :class:`SFDLayer` / :class:`CoSTEncoder` —— ``encoder.py``
  * :func:`time_contrastive_loss` / :func:`frequency_contrastive_loss` /
    :func:`convert_coefficient` —— ``losses.py``

关键点：
  * ``TFDLayer``：多尺度移动平均（kernels ``[1,2,4,8,16,32,64,128]``），
    ``k != 1`` 时在**时间维左侧** pad ``k-1``（causal），再 Conv1d(k) → 沿尺度平均。
  * ``SFDLayer``：``rfft`` → 复权重投影 → ``irfft``（torch 用 ``view_as_complex``）。
    权重形状 ``(freqs, in, out, 2)``（最后一维是实/虚部拼装）。
  * ``CoSTEncoder.forward`` 返回 ``(trend, season)``。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkiln.nn.ts_ts2vec import (DilatedConvLayer, SamePadConv,  # noqa: F401
                                    ConvLayer, generate_binomial_mask,
                                    generate_true_mask, mask_fill)

__all__ = ["TFDLayer", "SFDLayer", "CoSTEncoder", "time_contrastive_loss",
           "frequency_contrastive_loss", "convert_coefficient"]


class TFDLayer(nn.Module):
    """paddlets ``_cost/encoder.py::TFDLayer``（多尺度趋势提取）。"""

    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        self._kernels = list(kernels)
        self._tfd = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, k) for k in self._kernels])

    def forward(self, x):
        trends = []
        for idx, layer in enumerate(self._tfd):
            if self._kernels[idx] != 1:
                padding = self._kernels[idx] - 1
                # Paddle F.pad(X, (padding, 0), data_format='NCL') 等价于左侧 pad
                out = F.pad(x, (padding, 0))
                trends.append(layer(out))
                continue
            trends.append(layer(x))
        trends = torch.stack(trends, dim=0)
        return torch.mean(trends, dim=0)


class SFDLayer(nn.Module):
    """paddlets ``_cost/encoder.py::SFDLayer``（频域季节项）。"""

    def __init__(self, in_channels, out_channels, seq_len):
        super().__init__()
        freqs = seq_len // 2 + 1
        self._weight = nn.Parameter(torch.empty(freqs, in_channels, out_channels, 2))
        self._bias = nn.Parameter(torch.empty(freqs, out_channels, 2))
        # KaimingUniform（paddle 默认 a=sqrt(5)）
        fan_in = in_channels
        bound = math.sqrt(6.0 / ((1 + 5) * fan_in))     # kaiming_uniform_(a=sqrt(5))
        nn.init.uniform_(self._weight, -bound, bound)
        nn.init.uniform_(self._bias, -bound, bound)

    @staticmethod
    def _as_complex(t):
        return torch.view_as_complex(t.contiguous())

    def forward(self, x):
        batch_size, _, seq_len = x.shape
        out = torch.fft.rfft(x, dim=-1)                  # (B, C, F)
        out = out.transpose(0, 2)                        # (F, C, B)
        out = out.transpose(1, 2)                        # (F, B, C)
        w = self._as_complex(self._weight)               # (F, in, out)
        # Paddle: out (F,B,C) @ weight (F,C,out) -> (F,B,out)
        out = torch.matmul(out, w)                       # (F, B, out)
        out = out.transpose(0, 1)                        # (B, F, out)
        # ⚠️ bias 形状 (F, out)：需补 batch 维才能与 (B,F,out) 广播
        out = out + self._as_complex(self._bias).unsqueeze(0)
        out = out.transpose(1, 2)                        # (B, out, F)
        return torch.fft.irfft(out, n=seq_len, dim=-1)


class CoSTEncoder(nn.Module):
    """paddlets ``_cost/encoder.py::TSEncoder``（返回 ``(trend, season)``）。"""

    def __init__(self, in_channels, out_channels, hidden_channels, num_layers,
                 seq_len):
        super().__init__()
        k = math.sqrt(1.0 / in_channels)
        self._in_proj = nn.Linear(in_channels, hidden_channels)
        nn.init.uniform_(self._in_proj.weight, -k, k)
        nn.init.uniform_(self._in_proj.bias, -k, k)
        self._repr_dropout = nn.Dropout(0.1)
        self._dilated_conv = DilatedConvLayer(hidden_channels, out_channels,
                                              hidden_channels, kernel_size=3,
                                              num_layers=num_layers)
        kernels = [1, 2, 4, 8, 16, 32, 64, 128]
        self._tfd = TFDLayer(out_channels, out_channels // 2, kernels)
        self._sfd = SFDLayer(out_channels, out_channels // 2, seq_len)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        nan_mask = torch.any(torch.isnan(x), dim=-1)
        x = mask_fill(x, nan_mask, 0)

        out = self._in_proj(x)
        if not torch.is_tensor(mask):
            if mask == "binomial":
                mask = generate_binomial_mask(batch_size, seq_len, device=x.device)
            else:
                mask = generate_true_mask(batch_size, seq_len, device=x.device)
        mask = mask & (~nan_mask)
        out = mask_fill(out, ~mask, 0)

        out = out.transpose(1, 2)
        out = self._dilated_conv(out)
        trend = self._tfd(out).transpose(1, 2)
        season = self._sfd(out).transpose(1, 2)
        return trend, self._repr_dropout(season)


# --------------------------------------------------------------------------- #
# 损失（losses.py）
# --------------------------------------------------------------------------- #
def time_contrastive_loss(anchor, pos, neg, temperature):
    """paddlets ``_cost/losses.py::time_contrastive_loss``。

    ``anchor``/``pos``: ``(B, D)``；``neg``: ``(D, K)``（Paddle 用 ``matmul(anchor, neg)``）。
    """
    pos = torch.sum(anchor * pos, dim=1, keepdim=True)      # (B, 1)
    neg = torch.matmul(anchor, neg)                          # (B, K)
    logits = torch.cat([pos, neg], dim=-1) / temperature
    label = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, label)


def frequency_contrastive_loss(repr1, repr2):
    """paddlets ``_cost/losses.py::frequency_contrastive_loss``（与 TS2Vec instance 同形）。"""
    batch_size, seq_len = repr1.shape[:2]
    r = torch.cat([repr1, repr2], dim=0)
    r = r.transpose(0, 1)
    sim = torch.matmul(r, r.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -1.0 * F.log_softmax(logits, dim=-1)
    loss = torch.mean(logits[:, :batch_size, batch_size - 1: 2 * batch_size - 1])
    loss = loss + torch.mean(logits[:, batch_size: 2 * batch_size, :batch_size])
    return loss / 2.0


def convert_coefficient(tensor):
    """paddlets ``_cost/losses.py::convert_coefficient``（幅度/相位）。"""
    amp = torch.sqrt(torch.square(torch.real(tensor)) + torch.square(torch.imag(tensor)))
    phase = torch.atan2(torch.imag(tensor), torch.real(tensor))
    return amp, phase
