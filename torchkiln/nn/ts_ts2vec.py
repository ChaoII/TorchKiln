"""TS2Vec 表示学习（移植自 paddlets ``models/representation/dl/_ts2vec``）。

对应关系：
  * :class:`SamePadConv` / :class:`ConvLayer` / :class:`DilatedConvLayer` / :class:`TSEncoder`
      —— ``encoder.py``
  * :func:`instance_contrastive_loss` / :func:`temporal_contrastive_loss` /
    :func:`hierarchical_contrastive_loss` —— ``losses.py``
  * mask 工具 —— ``mask.py``
  * :class:`TS2VecModule` —— ``ts2vec.py::_TS2VecModule``

⚠️ 命名与 Paddle 逐一对齐（``_feat_extractor``、``_dilated_conv``、``_conv_layers``、
``_in_proj``、``_repr_dropout``、``_conv1``、``_conv2``、``_out_proj``），便于权重对拍。

关键点：
  * Paddle ``Conv1D(padding=p, dilation=d)`` 与 torch 同名参数**语义一致**；
    但 ``receptive_field`` 为偶数时要**去掉最后一个时间步**（``_remove``）。
  * 权重初始化：``Uniform(-k, k)``，``k=sqrt(1/(in_channels*kernel_size))``
    （`TSEncoder._in_proj` 是 ``sqrt(1/in_channels)``、``ConvLayer._out_proj`` 是 ``sqrt(1/in_channels)``）。
  * ``mask`` 的随机性来自 ``np.random``（binomial）——跨框架不可复现，
    对拍需注入固定 mask。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SamePadConv", "ConvLayer", "DilatedConvLayer", "TSEncoder",
           "TS2VecModule", "generate_true_mask", "generate_false_mask",
           "generate_last_mask", "generate_binomial_mask",
           "generate_continuous_mask", "mask_fill",
           "instance_contrastive_loss", "temporal_contrastive_loss",
           "hierarchical_contrastive_loss"]


# --------------------------------------------------------------------------- #
# mask 工具（mask.py）
# --------------------------------------------------------------------------- #
def generate_true_mask(batch_size, seq_len, device=None):
    return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)


def generate_false_mask(batch_size, seq_len, device=None):
    return torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)


def generate_last_mask(batch_size, seq_len, device=None):
    m = generate_true_mask(batch_size, seq_len, device)
    m[:, -1] = False
    return m


def generate_binomial_mask(batch_size, seq_len, p=0.5, device=None):
    """⚠️ Paddle 用 ``np.random.binomial``；这里可传 ``generator`` 以便复现。"""
    return torch.as_tensor(
        np.random.binomial(1, p, (batch_size, seq_len)).astype(bool),
        device=device)


def generate_continuous_mask(batch_size, seq_len, mask_num=5, mask_ratio=0.1,
                             device=None):
    m = generate_true_mask(batch_size, seq_len, device)
    mask_num = max(min(mask_num, seq_len // 2), 1)
    mask_len = max(int(mask_ratio * seq_len), 1)
    for row in range(batch_size):
        for _ in range(mask_num):
            start = np.random.randint(seq_len - mask_len + 1)
            m[row, start:start + mask_len] = False
    return m


def mask_fill(tensor, mask, value):
    """paddlets ``mask.py::paddle_mask_fill``（mask 形状 ``(B, L)``）。"""
    m = mask[:, :, None].expand_as(tensor)
    cache = torch.full_like(tensor, float(value))
    return torch.where(m, cache, tensor)


# --------------------------------------------------------------------------- #
# 网络（encoder.py）
# --------------------------------------------------------------------------- #
class SamePadConv(nn.Module):
    """paddlets ``encoder.py::SamePadConv``（"SAME" 填充 + 必要时去掉末步）。"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        padding = receptive_field // 2
        k = math.sqrt(1.0 / (in_channels * kernel_size))
        self._conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        nn.init.uniform_(self._conv.weight, -k, k)
        nn.init.uniform_(self._conv.bias, -k, k)
        self._remove = 1 if receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self._conv(x)
        if self._remove > 0:
            out = out[:, :, :-self._remove]
        return out


class ConvLayer(nn.Module):
    """paddlets ``encoder.py::ConvLayer``（残差 + 两层 GELU 卷积）。"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, final):
        super().__init__()
        self._conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self._conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation)
        k = math.sqrt(1.0 / in_channels)
        if in_channels != out_channels or final:
            self._out_proj = nn.Conv1d(in_channels, out_channels, 1)
            nn.init.uniform_(self._out_proj.weight, -k, k)
            nn.init.uniform_(self._out_proj.bias, -k, k)
        else:
            self._out_proj = None

    def forward(self, x):
        residual = self._out_proj(x) if self._out_proj is not None else x
        out = F.gelu(x)
        out = self._conv1(out)
        out = F.gelu(out)
        out = self._conv2(out)
        return residual + out


class DilatedConvLayer(nn.Module):
    """paddlets ``encoder.py::DilatedConvLayer``（膨胀率 2**k）。"""

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size,
                 num_layers):
        super().__init__()
        channels = [in_channels] + [hidden_channels] * num_layers + [out_channels]
        layers = []
        for k, (ci, co) in enumerate(zip(channels[:-1], channels[1:])):
            layers.append(ConvLayer(ci, co, kernel_size, dilation=2 ** k,
                                    final=(k == num_layers)))
        self._conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._conv_layers(x)


class TSEncoder(nn.Module):
    """paddlets ``encoder.py::TSEncoder``。

    ``forward(X, mask)``：``mask`` 可为 ``'binomial'``/``'all_true'``/``'mask_last'``
    或**直接传 bool 张量**（对拍时用后者以注入固定 mask）。
    """

    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super().__init__()
        k = math.sqrt(1.0 / in_channels)
        self._in_proj = nn.Linear(in_channels, hidden_channels)
        nn.init.uniform_(self._in_proj.weight, -k, k)
        nn.init.uniform_(self._in_proj.bias, -k, k)
        self._repr_dropout = nn.Dropout(0.1)
        self._dilated_conv = DilatedConvLayer(hidden_channels, out_channels,
                                              hidden_channels, kernel_size=3,
                                              num_layers=num_layers)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        nan_mask = torch.any(torch.isnan(x), dim=-1)
        x = mask_fill(x, nan_mask, 0)

        out = self._in_proj(x)
        if not torch.is_tensor(mask):
            if mask == "binomial":
                mask = generate_binomial_mask(batch_size, seq_len, device=x.device)
            elif mask == "all_true":
                mask = generate_true_mask(batch_size, seq_len, device=x.device)
            elif mask == "mask_last":
                mask = generate_last_mask(batch_size, seq_len, device=x.device)
            else:                      # None -> 不过滤
                mask = generate_true_mask(batch_size, seq_len, device=x.device)

        mask = mask & (~nan_mask)
        out = mask_fill(out, ~mask, 0)

        out = out.transpose(1, 2)
        out = self._dilated_conv(out)
        out = self._repr_dropout(out)
        return out.transpose(1, 2)


class TS2VecModule(nn.Module):
    """paddlets ``ts2vec.py::_TS2VecModule``（训练用 feat_extractor；推理用 SWA 平均权重）。

    ⚠️ paddle 版推理走 ``AveragedModel``（SWA）。这里实现同语义：``eval()`` 时
    用 ``swa_state``（若已 ``update_swa()``）替换权重；未更新则退化为 feat_extractor。
    """

    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super().__init__()
        self._feat_extractor = TSEncoder(in_channels, out_channels,
                                         hidden_channels, num_layers)
        self.register_buffer("_swa_n", torch.tensor(0, dtype=torch.long))
        self._swa_state = None

    def parameters(self, recurse=True):
        return self._feat_extractor.parameters(recurse=recurse)

    def update_swa(self):
        """paddlets ``AveragedModel.update_parameters`` 的等价：累计平均权重。"""
        sd = {k: v.detach().clone().float()
              for k, v in self._feat_extractor.state_dict().items()}
        if self._swa_state is None:
            self._swa_state = sd
            self._swa_n.fill_(1)
        else:
            n = int(self._swa_n.item())
            for k in self._swa_state:
                self._swa_state[k] = (self._swa_state[k] * n + sd[k]) / (n + 1)
            self._swa_n.fill_(n + 1)

    def forward(self, x, mask=None):
        if not self.training and self._swa_state is not None:
            backup = {k: v.detach().clone()
                      for k, v in self._feat_extractor.state_dict().items()}
            self._feat_extractor.load_state_dict(self._swa_state)
            out = self._feat_extractor(x, mask)
            self._feat_extractor.load_state_dict(backup)
            return out
        return self._feat_extractor(x, mask)


# --------------------------------------------------------------------------- #
# 损失（losses.py）
# --------------------------------------------------------------------------- #
def instance_contrastive_loss(repr1, repr2):
    """paddlets ``losses.py::instance_contrastive_loss``（逐时刻的样本级对比）。"""
    batch_size, seq_len = repr1.shape[:2]
    if batch_size == 1:
        return repr1.sum() * 0.0
    r = torch.cat([repr1, repr2], dim=0)                 # (2B, L, D)
    r = r.transpose(0, 1)                                # (L, 2B, D)
    sim = torch.matmul(r, r.transpose(1, 2))             # (L, 2B, 2B)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -1.0 * F.log_softmax(logits, dim=-1)
    loss = torch.mean(logits[:, :batch_size, batch_size - 1: 2 * batch_size - 1])
    loss = loss + torch.mean(logits[:, batch_size: 2 * batch_size, :batch_size])
    return loss / 2.0


def temporal_contrastive_loss(repr1, repr2):
    """paddlets ``losses.py::temporal_contrastive_loss``（逐样本的时间级对比）。"""
    batch_size, seq_len = repr1.shape[:2]
    if seq_len == 1:
        return repr1.sum() * 0.0
    r = torch.cat([repr1, repr2], dim=1)                 # (B, 2L, D)
    sim = torch.matmul(r, r.transpose(1, 2))             # (B, 2L, 2L)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -1.0 * F.log_softmax(logits, dim=-1)
    loss = torch.mean(logits[:, :seq_len, seq_len - 1: 2 * seq_len - 1])
    loss = loss + torch.mean(logits[:, seq_len: 2 * seq_len, :seq_len])
    return loss / 2.0


def hierarchical_contrastive_loss(repr1, repr2, alpha=0.5, temporal_unit=0):
    """paddlets ``losses.py::hierarchical_contrastive_loss``（多尺度 max-pool 聚合）。"""
    loss, d = repr1.sum() * 0.0, 0
    while repr1.shape[1] > 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(repr1, repr2)
        if d >= temporal_unit and 1 - alpha != 0:
            loss = loss + (1 - alpha) * temporal_contrastive_loss(repr1, repr2)
        repr1 = F.max_pool1d(repr1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        repr2 = F.max_pool1d(repr2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        d += 1
    if repr1.shape[1] == 1 and alpha != 0:
        loss = loss + alpha * instance_contrastive_loss(repr1, repr2)
        d += 1
    return loss / d
