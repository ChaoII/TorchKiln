"""Anomaly Transformer 网络（移植自 paddlets ``models/anomaly/dl/_anomaly_transformer``）。

对应关系：
  * :class:`PositionalEmbedding` / :class:`TokenEmbedding` / :class:`DataEmbedding` —— ``embedding.py``
  * :class:`EncoderLayer` / :class:`Encoder` —— ``encoder.py``
  * :class:`TriangularCausalMask` / :class:`AnomalyAttention` / :class:`AttentionLayer` —— ``attention.py``
  * :class:`AnomalyTransformerNet` —— ``anomaly_transformer.py::_Anomaly``

⚠️ 命名与 Paddle 逐一对齐（`embedding.value_embedding.tokenConv`、`encoder.attn_layers.*`、
`attention.query_projection` 等），便于权重对拍。

关键点：
  * ``Conv1d(data_format='NLC')`` 等价于「转置到 NCL 做 conv 再转回」——用
    :class:`Conv1dNLC` 实现，**保持权重形状 ``(out, in, k)`` 与键名不变**。
  * ``AnomalyAttention.distances`` 在 Paddle 里是**普通属性**（不进 state_dict），
    这里同样用非持久化张量。
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PositionalEmbedding", "TokenEmbedding", "DataEmbedding",
           "EncoderLayer", "Encoder", "TriangularCausalMask",
           "AnomalyAttention", "AttentionLayer", "AnomalyTransformerNet"]


class Conv1dNLC(nn.Conv1d):
    """``paddle.nn.Conv1D(data_format='NLC')`` 的等价物（NLC 进出，NCL 内部）。"""

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    """paddlets ``embedding.py::PositionalEmbedding``（sin/cos 位置编码，buffer）。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1]]


class TokenEmbedding(nn.Module):
    """paddlets ``embedding.py::TokenEmbedding``（k=3、circular padding）。"""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = Conv1dNLC(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1,
                                   padding_mode="circular")

    def forward(self, x):
        return self.tokenConv(x)


class DataEmbedding(nn.Module):
    """paddlets ``embedding.py::DataEmbedding`` = token + positional。"""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class TriangularCausalMask:
    """paddlets ``attention.py::TriangularCausalMask``。"""

    def __init__(self, batch_size: int, length: int):
        mask_shape = [batch_size, 1, length, length]
        self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    """paddlets ``attention.py::AnomalyAttention``（series 关联 + 高斯核 prior）。"""

    def __init__(self, win_size: int, mask_flag: bool = True, scale=None,
                 attention_dropout: float = 0.0, output_attention: bool = False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        distances = torch.zeros((win_size, win_size))
        for i in range(win_size):
            for j in range(win_size):
                distances[i][j] = abs(i - j)
        # ⚠️ Paddle 里是普通属性（非 buffer/parameter）⇒ 不进 state_dict
        self.distances = distances

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)
            scores = scores.masked_fill(attn_mask.mask.to(scores.device), -float("inf"))
        attn = scale * scores
        series = self.dropout(F.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)
        window_size = attn.shape[-1]
        sigma = sigma.transpose(1, 2)                # B L H -> B H L
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(torch.tensor(3., device=sigma.device, dtype=sigma.dtype), sigma)
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)   # B H L L
        prior = self.distances.to(sigma.device).unsqueeze(0).unsqueeze(0).repeat(
            sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        if self.output_attention:
            return V, series, prior, sigma
        return V, None


class AttentionLayer(nn.Module):
    """paddlets ``attention.py::AttentionLayer``。"""

    def __init__(self, attention, d_model: int, n_heads: int,
                 d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).reshape(B, L, H, -1)
        keys = self.key_projection(keys).reshape(B, S, H, -1)
        values = self.value_projection(values).reshape(B, S, H, -1)
        sigma = self.sigma_projection(x).reshape(B, L, H)
        out, series, prior, sigma = self.inner_attention(
            queries, keys, values, sigma, attn_mask)
        out = out.reshape(B, L, -1)
        return self.out_projection(out), series, prior, sigma


class EncoderLayer(nn.Module):
    """paddlets ``encoder.py::EncoderLayer``。"""

    def __init__(self, attention, d_model: int, d_ff=None, dropout: float = 0.1,
                 activation=F.gelu):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = Conv1dNLC(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = Conv1dNLC(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.conv2(y)
        y = self.dropout(y)
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    """paddlets ``encoder.py::Encoder``。"""

    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list, prior_list, sigma_list = [], [], []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list


class AnomalyTransformerNet(nn.Module):
    """paddlets ``anomaly_transformer.py::_Anomaly``。

    forward 输入 ``X["observed_cov_numeric"]``（``(N, L, C)``）；
    ``output_attention=True`` 时返回 ``(enc_out, series, prior, sigmas)``。
    """

    def __init__(self, win_size: int, enc_in: int, c_out: int, d_model: int = 512,
                 n_heads: int = 8, e_layers: int = 3, d_ff: int = 512,
                 dropout: float = 0.0, activation: Callable = F.gelu,
                 output_attention: bool = True):
        super().__init__()
        self.output_attention = output_attention
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    AnomalyAttention(win_size, False, attention_dropout=dropout,
                                     output_attention=output_attention),
                    d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation)
             for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model))
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["observed_cov_numeric"]
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        if self.output_attention:
            return enc_out, series, prior, sigmas
        return enc_out
