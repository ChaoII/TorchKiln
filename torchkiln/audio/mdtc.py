"""MDTC（Multi-scale Dilated Temporal Convolution）KWS 模型 —— torch 移植。

对齐 PaddleSpeech ``paddlespeech/kws/models/mdtc.py``（逐层同名，便于权重转换）。
配置 ``conf/mdtc.yaml``: ``kaldi_fbank(sr=16000, 25ms窗/10ms帧移, n_mels=80)`` +
``MDTC(stack_num=3, stack_size=4, in_channels=80, res_channels=32, kernel_size=5, causal=True)``

⚠️ **输入布局是 ``(N, T, 80)``（帧在前），不是 ``(N, 80, T)``** —— 实测依据：
给 ``(2,80,120)`` 会报 ``The channel of input must be divisible by groups, channel is 120``，
给 ``(2,120,80)`` 才 OK。``MDTC.forward`` 内部先 pad 再 ``transpose(1,2)`` 得 ``(N,C,T)`` 进卷积。

⚠️ ``MDTC.forward`` 返回 **``(outputs, None)`` 元组**（kaldi 风格），调用方需解包。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DSDilatedConv1d", "TCNBlock", "TCNStack", "MDTC", "KWSModel",
           "MDTC_CONF", "KALDI_FBANK_CONF"]

# conf/mdtc.yaml
MDTC_CONF = dict(num_keywords=1, stack_num=3, stack_size=4, in_channels=80,
                 res_channels=32, kernel_size=5, causal=True,
                 receptive_fields=184)
KALDI_FBANK_CONF = dict(sample_rate=16000, frame_length_ms=25,
                        frame_shift_ms=10, n_mels=80)


class DSDilatedConv1d(nn.Module):
    """depthwise dilated conv -> BN -> pointwise 1x1（``padding=0``，由上游 pad）。"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, stride=1, bias=True):
        super().__init__()
        self.receptive_fields = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                              padding=0, dilation=dilation, stride=stride,
                              groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                   padding=0, dilation=1, bias=bias)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.pointwise(outputs)
        return outputs


class TCNBlock(nn.Module):
    def __init__(self, in_channels, res_channels, kernel_size, dilation,
                 causal):
        super().__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.receptive_fields = dilation * (kernel_size - 1)
        self.half_receptive_fields = self.receptive_fields // 2
        self.conv1 = DSDilatedConv1d(in_channels, res_channels, kernel_size,
                                     dilation=dilation)
        self.bn1 = nn.BatchNorm1d(res_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(res_channels)
        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        outputs = self.relu1(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.causal:
            inputs = inputs[:, :, self.receptive_fields:]
        else:
            inputs = inputs[:, :, self.half_receptive_fields:
                            -self.half_receptive_fields]
        if self.in_channels == self.res_channels:
            res_out = self.relu2(outputs + inputs)
        else:
            res_out = self.relu2(outputs)
        return res_out


class TCNStack(nn.Module):
    def __init__(self, in_channels, stack_num, stack_size, res_channels,
                 kernel_size, causal):
        super().__init__()
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.stack_size = stack_size
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.res_blocks = nn.Sequential(*self._stack_tcn_blocks())
        self.receptive_fields = sum(b.receptive_fields
                                    for b in self.res_blocks)

    def _build_dilations(self):
        # 外层 stack_size、内层 stack_num => (stack_size x [2**0..2**(n-1)])
        dilations = []
        for _ in range(self.stack_size):
            for l in range(self.stack_num):
                dilations.append(2 ** l)
        return dilations

    def _stack_tcn_blocks(self):
        dilations = self._build_dilations()
        blocks = [TCNBlock(self.in_channels, self.res_channels,
                           self.kernel_size, dilations[0], self.causal)]
        for d in dilations[1:]:
            blocks.append(TCNBlock(self.res_channels, self.res_channels,
                                   self.kernel_size, d, self.causal))
        return blocks

    def forward(self, inputs):
        return self.res_blocks(inputs)


class MDTC(nn.Module):
    """返回 ``(outputs, None)``（kaldi 风格）；输入 ``(N, T, in_channels)``。"""

    def __init__(self, stack_num, stack_size, in_channels, res_channels,
                 kernel_size, causal=True):
        super().__init__()
        assert kernel_size % 2 == 1, kernel_size
        self.kernel_size = kernel_size
        self.causal = causal
        self.preprocessor = TCNBlock(in_channels, res_channels, kernel_size,
                                     dilation=1, causal=causal)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList()
        self.receptive_fields = self.preprocessor.receptive_fields
        for _ in range(stack_num):
            self.blocks.append(TCNStack(res_channels, stack_size, 1,
                                        res_channels, kernel_size, causal))
            self.receptive_fields += self.blocks[-1].receptive_fields
        self.half_receptive_fields = self.receptive_fields // 2
        self.hidden_dim = res_channels

    def forward(self, x):
        """x: (N, T, in_channels)。"""
        if self.causal:
            # Paddle 6 值 pad 从最内层维度配对 => C=(0,0), T=(R,0), N=(0,0)
            # torch 3D 的 pad 也是「从最后一维开始」 => 等价写法 (0,0, R,0)
            outputs = F.pad(x, (0, 0, self.receptive_fields, 0),
                            mode="constant")
        else:
            outputs = F.pad(x, (0, 0, self.half_receptive_fields,
                                self.half_receptive_fields), mode="constant")
        outputs = outputs.transpose(1, 2)          # (N, T+R, C) -> (N, C, T+R)
        outputs_list = []
        outputs = self.relu(self.preprocessor(outputs))
        for block in self.blocks:
            outputs = block(outputs)
            outputs_list.append(outputs)

        normalized = []
        output_size = outputs_list[-1].shape[-1]
        for t in outputs_list:
            remove_length = t.shape[-1] - output_size
            if self.causal and remove_length > 0:
                normalized.append(t[:, :, remove_length:])
            elif (not self.causal) and remove_length > 1:
                h = remove_length // 2
                normalized.append(t[:, :, h:-h])
            else:
                normalized.append(t)

        out = torch.zeros_like(outputs_list[-1])
        for t in normalized:
            out = out + t
        out = out.transpose(1, 2)                  # -> (N, T', C)
        return out, None


class KWSModel(nn.Module):
    """MDTC backbone + Linear(hidden_dim -> num_keywords) + Sigmoid。"""

    def __init__(self, backbone, num_keywords):
        super().__init__()
        self.backbone = backbone
        self.linear = nn.Linear(self.backbone.hidden_dim, num_keywords)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        outputs, _ = self.backbone(x)      # MDTC 返回 (tensor, None)
        outputs = self.linear(outputs)
        return self.activation(outputs)


def build_mdtc(**conf) -> KWSModel:
    """按 ``conf/mdtc.yaml`` 构建 ``KWSModel``。"""
    c = dict(MDTC_CONF)
    c.update(conf)
    bb = MDTC(stack_num=c["stack_num"], stack_size=c["stack_size"],
              in_channels=c["in_channels"], res_channels=c["res_channels"],
              kernel_size=c["kernel_size"], causal=c["causal"])
    return KWSModel(backbone=bb, num_keywords=c["num_keywords"])
