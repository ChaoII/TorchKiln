"""Depth model: CSP backbone + multi-level fusion decoder -> 1-channel log-depth."""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchx.models import Conv
from pytorchx.models.det import DetBackbone

__all__ = ["build_model", "DepthModel", "DepthDecoder"]


class DepthDecoder(nn.Module):
    def __init__(self, channels=(256, 512, 1024), hidden=128):
        super().__init__()
        c3, c4, c5 = channels
        self.lat3 = Conv(c3, hidden, 1, 1)
        self.lat4 = Conv(c4, hidden, 1, 1)
        self.lat5 = Conv(c5, hidden, 1, 1)
        self.fuse = nn.Sequential(
            Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1)
        )
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, feats):
        p3, p4, p5 = feats
        x = self.lat3(p3)
        x = x + F.interpolate(self.lat4(p4), scale_factor=2, mode="nearest")
        x = x + F.interpolate(self.lat5(p5), scale_factor=4, mode="nearest")
        x = self.fuse(x)
        return self.head(self.up(x))  # (B, 1, H/4, W/4) log-depth


class DepthModel(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.backbone(x))


def build_model(arch):
    bb = arch.get("Backbone") or {}
    hd = arch.get("Head") or {}
    backbone = DetBackbone(
        scale=bb.get("scale", "n"), in_channels=int(bb.get("in_channels", 3))
    )
    decoder = DepthDecoder(
        channels=backbone.out_channels, hidden=int(hd.get("hidden", 128))
    )
    return DepthModel(backbone, decoder)
