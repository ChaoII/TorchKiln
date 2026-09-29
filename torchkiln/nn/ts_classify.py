"""时序分类网络（移植自 paddlets ``models/classify/dl``）。

对应关系：
  * :class:`CNNBlock`        —— ``cnn.py::_CNNBlock``（Conv1d->act->AvgPool1D, valid padding）
  * :class:`InceptionModule` / :class:`InceptionBlock` / :class:`InceptionTime`
                              —— ``inception_time.py``（InceptionTime 分类器）

⚠️ 命名与 Paddle 逐一对齐（``_nn``、``inception``、``shortcut``、``bottleneck``、
``convs``、``maxconvpool``、``bn``、``gap``、``flatten``、``fc``），便于权重对拍。

关键点：
  * Paddle ``padding="SAME"``（stride=1）等价 torch ``padding=k//2``（对奇数 k）
    ——InceptionTime 里所有 k 都被强制成**奇数**，故 ``k//2`` 精确等价。
  * ``_CNNBlock`` 用 ``'valid'`` 填充（不填充）。
"""
from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn

__all__ = ["CNNBlock", "InceptionModule", "InceptionBlock", "InceptionTime",
           "ACTIVATIONS"]

ACTIVATIONS = {
    "ReLU": nn.ReLU, "PReLU": nn.PReLU, "Softplus": nn.Softplus, "Tanh": nn.Tanh,
    "SELU": nn.SELU, "LeakyReLU": nn.LeakyReLU, "Sigmoid": nn.Sigmoid,
    "GELU": nn.GELU, "ELU": nn.ELU,
}


def _act(a):
    if a is None:
        return nn.Identity()
    if isinstance(a, type):
        return a()
    return a


class CNNBlock(nn.Module):
    """paddlets ``cnn.py::_CNNBlock``（输入 dict ``{'features': (B,L,C)}``）。"""

    def __init__(self, in_chunk_dim: int, in_chunk_lens: int, n_classes: int,
                 hidden_config: List[int] = None, activation=None,
                 last_activation=None, kernel_size: int = 7,
                 avg_pool_size: int = 3, use_bn: bool = False,
                 use_drop: bool = False, dropout_rate: float = 0.5):
        super().__init__()
        hidden_config = hidden_config if hidden_config else [6, 12]
        activation = activation if activation is not None else nn.Sigmoid
        last_activation = last_activation if last_activation is not None else nn.Softmax
        self._n_classes = n_classes
        dims = [in_chunk_dim] + list(hidden_config)
        layers: List[nn.Module] = []
        L = in_chunk_lens
        for i in range(1, len(dims)):
            layers.append(nn.Conv1d(dims[i - 1], dims[i], kernel_size))   # valid
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i]))
            layers.append(_act(activation))
            layers.append(nn.AvgPool1d(avg_pool_size))
            if use_drop:
                layers.append(nn.Dropout(dropout_rate))
            L = (L - kernel_size + 1) // avg_pool_size
        if L < 1:
            raise ValueError("Conv1d output size must be >= 1; use smaller kernel or longer window")
        layers.append(nn.Flatten())
        layers.append(nn.Linear(L * dims[-1], n_classes))
        layers.append(_act(last_activation))
        self._nn = nn.Sequential(*layers)
        self.out_dim = L * dims[-1]

    def forward(self, x):
        if isinstance(x, dict):
            x = x["features"]
        x = x.transpose(1, 2)
        return self._nn(x)


class InceptionModule(nn.Module):
    """paddlets ``inception_time.py::_InceptionModule``。"""

    def __init__(self, input_len: int, conv_out_size: int, kernel_size: int = 40,
                 activation: Callable = None, use_bottleneck: bool = True):
        super().__init__()
        ks = [kernel_size // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]      # ensure odd
        use_bottleneck = use_bottleneck if input_len > 1 else False
        self.bottleneck = (nn.Conv1d(input_len, conv_out_size, 1, bias=False,
                                     padding=0) if use_bottleneck else None)
        in_c = conv_out_size if use_bottleneck else input_len
        self.convs = nn.ModuleList([
            nn.Conv1d(in_c, conv_out_size, k, bias=False, padding=k // 2)
            for k in ks])
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(input_len, conv_out_size, 1, bias=False))
        self.bn = nn.BatchNorm1d(conv_out_size * 4)
        self.act = _act(activation) if activation is not None else nn.ReLU()

    def forward(self, x):
        input_tensor = x
        if self.bottleneck is not None:
            x = self.bottleneck(input_tensor)
        x = torch.cat([conv(x) for conv in self.convs] + [self.maxconvpool(input_tensor)], dim=1)
        return self.act(self.bn(x))


class InceptionBlock(nn.Module):
    """paddlets ``inception_time.py::_InceptionBlock``（depth 个 module + 残差 shortcut）。"""

    def __init__(self, input_len: int, output_len: int = 128, kernel_size: int = 40,
                 depth: int = 6, activation: Callable = None, use_residual: bool = True,
                 use_bottleneck: bool = True):
        super().__init__()
        self.residual, self.depth = use_residual, depth
        self.inception = nn.ModuleList()
        self.shortcut = nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(
                input_len if d == 0 else output_len, int(output_len / 4),
                kernel_size=kernel_size, activation=activation,
                use_bottleneck=use_bottleneck))
            if self.residual and d % 3 == 2:
                n_in, n_out = (input_len if d == 2 else output_len), output_len
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out
                                     else nn.Conv1d(n_in, n_out, 1, padding=0))
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for d in range(self.depth):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                res = x = self.act(x + self.shortcut[d // 3](res))
        return x


class InceptionTime(nn.Module):
    """paddlets ``inception_time.py::_InceptionTime``（输入 dict ``{'features': (B,L,C)}``）。"""

    def __init__(self, channel_in: int, channel_out: int, kernel_size: int = 41,
                 block_out_size: int = 128, block_depth: int = 6,
                 activation: Callable = None, use_residual: bool = True,
                 use_bottleneck: bool = True):
        super().__init__()
        self.inceptionblock = InceptionBlock(
            channel_in, block_out_size, kernel_size, block_depth,
            activation, use_residual, use_bottleneck)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(block_out_size, channel_out)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["features"]
        x = x.transpose(1, 2)
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.flatten(x)
        return self.fc(x)
