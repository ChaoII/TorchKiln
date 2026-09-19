"""Semantic segmentation model: CSP backbone + multi-level fusion decoder.

Outputs per-pixel class logits at stride 4; the loss/metric upsample to the
target mask size.
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchx.models import Conv
from pytorchx.models.det import DetBackbone

__all__ = ["build_model", "SemModel", "SemDecoder"]


class SemDecoder(nn.Module):
    def __init__(self, channels=(256, 512, 1024), num_classes=19, hidden=128):
        super().__init__()
        c3, c4, c5 = channels
        self.lat3 = Conv(c3, hidden, 1, 1)
        self.lat4 = Conv(c4, hidden, 1, 1)
        self.lat5 = Conv(c5, hidden, 1, 1)
        self.fuse = nn.Sequential(
            Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1)
        )
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Conv2d(hidden, num_classes, 1)
        self.num_classes = int(num_classes)

    def forward(self, feats):
        p3, p4, p5 = feats
        x = self.lat3(p3)
        x = x + F.interpolate(self.lat4(p4), scale_factor=2, mode="nearest")
        x = x + F.interpolate(self.lat5(p5), scale_factor=4, mode="nearest")
        x = self.fuse(x)
        return self.head(self.up(x))  # stride 4 logits


class SemModel(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.num_classes = decoder.num_classes

    def forward(self, x):
        return self.decoder(self.backbone(x))


def build_model(arch):
    bb = arch.get("Backbone") or {}
    hd = arch.get("Head") or {}
    backbone = DetBackbone(scale=bb.get("scale", "n"), in_channels=int(bb.get("in_channels", 3)))
    decoder = SemDecoder(
        channels=backbone.out_channels,
        num_classes=int(hd.get("num_classes", 19)),
        hidden=int(hd.get("hidden", 128)),
    )
    return SemModel(backbone, decoder)
