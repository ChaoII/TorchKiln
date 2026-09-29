"""RUL / 序列回归专用**注意力结构**（补 `ts_models.py` 缺 Attention 的空档）。

新增两类（文献里 RUL 低 RMSE 的主力结构）：
  * :class:`BiLSTMAttention` —— **双向 LSTM + 加性(Multiplicative)注意力 + 回归头**
    （C-MAPSS 上 BiLSTM+Attention 常见 RMSE 12~16）
  * :class:`TransformerRegressor` —— **Transformer 编码器 + 注意力池化 + 回归头**
    （Sensor Attention / 长程依赖，SOTA 常用）

接口与 `ts_models.py` 一致：输入 ``{"past_target": (B, L, C)}``，
输出 ``(B, out_chunk_len, target_dim)``；**逐传感器通道预测**（与框架其它 ts 模型同构），
RUL 取通道均值/最小（由 `RULMetric.reduce` 决定）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["BiLSTMAttention", "TransformerRegressor"]


class _AdditiveAttention(nn.Module):
    """加性注意力池化：为每个时间步打分后加权求和（Bahdanau 风格）。

    ``score_t = v^T tanh(W h_t)`` -> softmax -> ``context = sum_t a_t h_t``
    """

    def __init__(self, hidden: int, attn_dim: int = 64):
        super().__init__()
        self._w = nn.Linear(hidden, attn_dim)
        self._v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):                      # (B, L, H)
        e = self._v(torch.tanh(self._w(h)))    # (B, L, 1)
        a = torch.softmax(e, dim=1)
        return (a * h).sum(dim=1), a.squeeze(-1)   # (B, H), (B, L)


class BiLSTMAttention(nn.Module):
    """双向 LSTM + 注意力池化 -> 回归头。

    Args:
        in_chunk_len: 回看窗口 L
        out_chunk_len: 预测步长 H（RUL 时=1）
        target_dim: **输入/输出通道数**（与框架其它 ts 模型语义一致）
        hidden_dim: LSTM 隐层（双向 => 实际 2×hidden）
        num_layers: LSTM 层数
        dropout: dropout
        attn_dim: 注意力隐层
        head_hidden: 回归头隐层
        use_input_proj: 是否先用 Linear 投影到 hidden（通道多时有用）
    """

    def __init__(self, in_chunk_len: int, out_chunk_len: int, target_dim: int = 1,
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1,
                 attn_dim: int = 64, head_hidden: int = 64,
                 use_input_proj: bool = True):
        super().__init__()
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        self._in_proj = (nn.Linear(target_dim, hidden_dim)
                         if use_input_proj else nn.Identity())
        self._lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self._attn = _AdditiveAttention(hidden_dim * 2, attn_dim)
        self._drop = nn.Dropout(dropout)
        self._head = nn.Sequential(
            nn.Linear(hidden_dim * 2, head_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_chunk_len * target_dim))

    def forward(self, data):
        x = data["past_target"]                    # (B, L, C)
        x = self._in_proj(x)
        h, _ = self._lstm(x)                       # (B, L, 2H)
        ctx, _att = self._attn(h)                  # (B, 2H)
        out = self._head(self._drop(ctx))          # (B, H*C)
        return out.view(out.shape[0], self._out_chunk_len, self._target_dim)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                          # (B, L, D)
        return x + self.pe[:, :x.shape[1]]


class TransformerRegressor(nn.Module):
    """Transformer 编码器 + **可学习注意力池化** -> 回归头。

    `nn.TransformerEncoder` 自带 multi-head self-attention（框架此前没有）。
    池化用可学习 query 的 cross-attention（比 mean-pool 更能聚焦关键时间步）。
    """

    def __init__(self, in_chunk_len: int, out_chunk_len: int, target_dim: int = 1,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 256, dropout: float = 0.1, head_hidden: int = 64):
        super().__init__()
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        self._in_proj = nn.Linear(target_dim, d_model)
        self._pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self._enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self._pool_q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self._pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                           batch_first=True)
        self._drop = nn.Dropout(dropout)
        self._head = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_chunk_len * target_dim))

    def forward(self, data):
        x = data["past_target"]                    # (B, L, C)
        x = self._pos(self._in_proj(x))
        h = self._enc(x)                           # (B, L, D)
        q = self._pool_q.expand(h.shape[0], -1, -1)
        ctx, _ = self._pool(q, h, h)               # (B, 1, D)
        out = self._head(self._drop(ctx.squeeze(1)))
        return out.view(out.shape[0], self._out_chunk_len, self._target_dim)
