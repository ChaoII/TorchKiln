"""PP-LCNet_x1_0 + multi-label head (PaddleX/PaddleClas attribute recognition).

Ported from ``PaddleClas/ppcls/arch/backbone/legendary_models/pp_lcnet.py``:
``conv1`` (3 -> 16, k3 s2, hardswish) followed by five ``DepthwiseSeparable``
stages (SE only in the last two, per the paper), then

    1x1 ``last_conv`` (512 -> ``class_expand``=1280) -> hardswish -> dropout
    -> Linear(1280, class_num)

``dropout_prob`` is 0.2 for scratch training and 0.5 when the ``use_ssld``
checkpoints are used (PaddleX sets ``use_ssld: True`` for both attribute models).
This one network serves **both** pedestrian (26 attributes, 256x192) and vehicle
(19 attributes, 192x256) recognition - only ``class_num``/image size differ.
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchx.nn.modules import make_divisible

__all__ = ["PPLCNetX1_0", "MultiLabelHead", "AttributeNet", "ATTRIBUTE_NET_CONFIG"]

# k, in_c, out_c, stride, use_se   (PaddleClas ``NET_CONFIG`` for x1_0)
ATTRIBUTE_NET_CONFIG = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
}


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act="hardswish"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish() if act == "hardswish" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """SE block as in PaddleClas (reduction 4, relu, hardsigmoid)."""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        scale = self.hsigmoid(self.conv2(self.relu(self.conv1(self.avg_pool(x)))))
        return x * scale


class DepthwiseSeparable(nn.Module):
    """dw(k, s) -> BN -> act -> [SE] -> pw 1x1 -> BN -> act (PaddleClas order)."""

    def __init__(self, in_channels, out_channels, dw_size=3, stride=1, use_se=False, act="hardswish"):
        super().__init__()
        self.dw_conv = ConvBNLayer(
            in_channels, in_channels, dw_size, stride, groups=in_channels, act=act
        )
        self.se = SqueezeExcitation(in_channels) if use_se else nn.Identity()
        self.pw_conv = ConvBNLayer(in_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return self.pw_conv(self.se(self.dw_conv(x)))


class PPLCNetX1_0(nn.Module):
    """Backbone only (the classification head lives in :class:`MultiLabelHead`)."""

    def __init__(self, scale=1.0, stride_list=(2, 2, 2, 2, 2), act="hardswish", out_channels=None):
        super().__init__()
        cfg = {k: [list(row) for row in v] for k, v in ATTRIBUTE_NET_CONFIG.items()}
        for i, stride in enumerate(stride_list[1:]):
            cfg["blocks{}".format(i + 3)][0][3] = stride
        self.scale = scale
        self.conv1 = ConvBNLayer(3, make_divisible(16 * scale), 3, stride_list[0], act=act)
        blocks = []
        for name in ("blocks2", "blocks3", "blocks4", "blocks5", "blocks6"):
            layers = []
            for (k, in_c, out_c, s, se) in cfg[name]:
                layers.append(
                    DepthwiseSeparable(
                        make_divisible(in_c * scale),
                        make_divisible(out_c * scale),
                        dw_size=k,
                        stride=s,
                        use_se=se,
                        act=act,
                    )
                )
            blocks.append(nn.Sequential(*layers))
        self.blocks2, self.blocks3, self.blocks4, self.blocks5, self.blocks6 = blocks
        self.out_channels = int(out_channels or make_divisible(512 * scale))

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        return x


class MultiLabelHead(nn.Module):
    """1x1 ``last_conv`` -> hardswish -> dropout -> Linear(class_num)."""

    def __init__(self, in_channels=512, class_expand=1280, num_classes=26, dropout_prob=0.2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_conv = nn.Conv2d(in_channels, class_expand, 1, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(class_expand, num_classes)

    def forward(self, x):
        x = self.hardswish(self.last_conv(x))
        x = self.flatten(self.avg_pool(x))
        return self.fc(self.dropout(x))


class AttributeNet(nn.Module):
    def __init__(self, num_classes=26, scale=1.0, class_expand=1280, dropout_prob=0.2):
        super().__init__()
        self.backbone = PPLCNetX1_0(scale=scale)
        self.head = MultiLabelHead(
            in_channels=self.backbone.out_channels,
            class_expand=class_expand,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )
        self.num_classes = int(num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))
