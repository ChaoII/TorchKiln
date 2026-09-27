"""ECAPA-TDNN 说话人验证 backbone —— torch 移植。

对齐 PaddleSpeech ``paddlespeech/vector/models/ecapa_tdnn.py``（逐层同名，便于权重转换）。
论文: ECAPA-TDNN, https://arxiv.org/abs/2005.07143

输入口径（``conf/model.yaml``）:
    sr=16000, n_mels=80, window_size=400(25ms), hop_size=160(10ms)
    model.input_size=80, channels=[1024,1024,1024,1024,3072],
    kernel_sizes=[5,3,3,3,1], dilations=[1,2,3,4,1],
    attention_channels=128, lin_neurons=192
    -> 输入 (N, 80, T)，输出 (N, 192)

移植要点:
  * Paddle ``BatchNorm1d(momentum=0.9)`` ≡ torch ``BatchNorm1d(momentum=0.1)``
    （两者都是"保留 90% 旧统计量"）；eval 下用 running stats，不影响 ②③④。
  * Paddle ``Conv1d`` 的 "same" padding 是**对称** ``d*(k-1)//2``，
    与 torch 的 ``padding=`` 参数等价（本配置所有 d*(k-1) 均为偶数）。
  * Paddle ``Tensor.max(axis)`` 只返回值 / ``.clip()`` ≡ ``torch.clamp(min=)``。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["length_to_mask", "Conv1d", "BatchNorm1d", "TDNNBlock",
           "Res2NetBlock", "SEBlock", "AttentiveStatisticsPooling",
           "SERes2NetBlock", "EcapaTdnn", "ECAPA_TDNN_CONF"]


# conf/model.yaml 的模型配置
ECAPA_TDNN_CONF = dict(
    sr=16000, n_mels=80, window_size=400, hop_size=160,
    input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
    kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
    attention_channels=128, lin_neurons=192,
    res2net_scale=8, se_channels=128, global_context=True,
)


def length_to_mask(lengths, max_len=None):
    """(N,) 长度 -> (N, max_len) 0/1 掩码（PaddleSpeech 版按元素比较）。"""
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths)
    lengths = lengths.reshape(-1)
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return (ids.unsqueeze(0) < lengths.unsqueeze(1)).to(torch.float32)


def _padding_for(k, d, s):
    """Paddle 'same' 的对称 padding（stride=1 时等于 d*(k-1)//2）。"""
    if s > 1:
        return k // 2
    return (d * (k - 1)) // 2


class Conv1d(nn.Module):
    """带 same-padding 的 1D 卷积（对齐 PaddleSpeech 的 Conv1d）。"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, padding="same", **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding_mode = "zeros"
        if padding == "same":
            p = _padding_for(kernel_size, dilation, stride)
        elif isinstance(padding, int):
            p = padding
        else:
            raise ValueError(f"Padding must be 'same'. Got {padding}")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, padding=p)

    def forward(self, x):
        return self.conv(x)


class BatchNorm1d(nn.Module):
    """``input_size`` → torch 的 ``num_features``；momentum 0.9 → 0.1。"""

    def __init__(self, input_size, eps=1e-5, momentum=0.9, **kwargs):
        super().__init__()
        # Paddle momentum=0.9 表示保留 90% 旧统计量；torch 用 (1-momentum) 表达同义
        self.norm = nn.BatchNorm1d(input_size, eps=eps,
                                   momentum=1.0 - float(momentum))

    def forward(self, x):
        return self.norm(x)


class TDNNBlock(nn.Module):
    """conv -> activation -> batchnorm（顺序与 Paddle 一致）。"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 activation=nn.ReLU):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size,
                           dilation=dilation)
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x, lengths=None):
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(nn.Module):
    """chunk(scale) + 逐级累加（i==0 直通、i==1 过 TDNN、i>=2 累加后再过）。"""

    def __init__(self, in_channels, out_channels, scale=8, dilation=1):
        super().__init__()
        assert in_channels % scale == 0, (in_channels, scale)
        assert out_channels % scale == 0, (out_channels, scale)
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList([
            TDNNBlock(in_channel, hidden_channel, kernel_size=3,
                      dilation=dilation)
            for _ in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        return torch.cat(y, dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: conv1 -> relu -> conv2 -> sigmoid -> * x。"""

    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = Conv1d(in_channels, se_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = Conv1d(se_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L).unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """注意力统计池化（mean + std 拼接）。"""

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(attention_channels, channels, kernel_size=1)

    def forward(self, x, lengths=None):
        C, L = x.shape[1], x.shape[2]

        def _stats(x, m):
            mean = (m * x).sum(dim=2)
            var = (m * (x - mean.unsqueeze(2)) ** 2).sum(dim=2)
            std = torch.sqrt(var.clamp(min=self.eps))
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], dtype=torch.float32)

        mask = length_to_mask(lengths * L, max_len=L).unsqueeze(1)

        if self.global_context:
            total = mask.sum(dim=2, keepdim=True).to(torch.float32)
            mean, std = _stats(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        attn = self.conv(torch.tanh(self.tdnn(attn)))

        # 屏蔽 padding 位置
        attn = torch.where(mask.repeat(1, C, 1) == 0,
                           torch.full_like(attn, float("-inf")), attn)
        attn = F.softmax(attn, dim=2)
        mean, std = _stats(x, attn)
        pooled = torch.cat((mean, std), dim=1).unsqueeze(2)
        return pooled


class SERes2NetBlock(nn.Module):
    """tdnn1(k=1) -> Res2Net -> tdnn2(k=1) -> SE -> + residual。"""

    def __init__(self, in_channels, out_channels, res2net_scale=8,
                 se_channels=128, kernel_size=1, dilation=1,
                 activation=nn.ReLU):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(in_channels, out_channels, kernel_size=1,
                               dilation=1, activation=activation)
        self.res2net_block = Res2NetBlock(out_channels, out_channels,
                                          res2net_scale, dilation)
        self.tdnn2 = TDNNBlock(out_channels, out_channels, kernel_size=1,
                               dilation=1, activation=activation)
        self.se_block = SEBlock(out_channels, se_channels, out_channels)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, lengths=None):
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)
        return x + residual


class EcapaTdnn(nn.Module):
    """ECAPA-TDNN backbone：输出说话人 embedding (N, lin_neurons)。"""

    def __init__(self, input_size, lin_neurons=192, activation=nn.ReLU,
                 channels=(1024, 1024, 1024, 1024, 3072),
                 kernel_sizes=(5, 3, 3, 3, 1), dilations=(1, 2, 3, 4, 1),
                 attention_channels=128, res2net_scale=8, se_channels=128,
                 global_context=True):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(dilations)
        self.channels = list(channels)
        self.emb_size = lin_neurons

        blocks = [TDNNBlock(input_size, channels[0], kernel_sizes[0],
                            dilations[0], activation)]
        for i in range(1, len(channels) - 1):
            blocks.append(SERes2NetBlock(
                channels[i - 1], channels[i], res2net_scale=res2net_scale,
                se_channels=se_channels, kernel_size=kernel_sizes[i],
                dilation=dilations[i], activation=activation))
        self.blocks = nn.ModuleList(blocks)

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(channels[-1], channels[-1], kernel_sizes[-1],
                             dilations[-1], activation)
        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1], attention_channels=attention_channels,
            global_context=global_context)
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)
        # Final linear transformation
        self.fc = Conv1d(channels[-1] * 2, self.emb_size, kernel_size=1)

    def forward(self, x, lengths=None):
        """x: (N, input_size, T) log-fbank -> (N, lin_neurons)。"""
        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation: **排除第一层**（concat blocks[1..]）
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)
        x = self.fc(x)
        return x
