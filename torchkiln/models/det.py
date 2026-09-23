"""Detection model: CSP backbone (P3/P4/P5) + PAN neck + anchor-free head.

Independent implementation (no upstream code copied). The head is DFL-free:
each point directly regresses ``ltrb`` distances (in stride units) plus per-class
logits, so exported graphs stay simple.
"""
from __future__ import absolute_import

import math

import torch
import torch.nn as nn

from torchkiln.models import C2f, Conv, SPPF, SCALES

__all__ = ["build_model", "YoloDetModel", "DetBackbone", "PAN", "DetectHead"]


class DetBackbone(nn.Module):
    """Returns the P3 (stride 8), P4 (16) and P5 (32) feature maps."""

    def __init__(self, scale="n", in_channels=3):
        super().__init__()
        depth, width, max_ch = SCALES.get(scale, SCALES["n"])

        def c(ch):
            return min(int(round(ch * width)), max_ch)

        def n(rep):
            return max(round(rep * depth), 1)

        self.stem = Conv(in_channels, c(64), 3, 2)
        self.stage1 = nn.Sequential(
            Conv(c(64), c(128), 3, 2), C2f(c(128), c(128), n(2), True)
        )
        self.stage2 = nn.Sequential(
            Conv(c(128), c(256), 3, 2), C2f(c(256), c(256), n(2), True)
        )
        self.stage3 = nn.Sequential(
            Conv(c(256), c(512), 3, 2), C2f(c(512), c(512), n(2), True)
        )
        self.stage4 = nn.Sequential(
            Conv(c(512), c(1024), 3, 2),
            C2f(c(1024), c(1024), n(2), True),
            SPPF(c(1024), c(1024)),
        )
        self.out_channels = (c(256), c(512), c(1024))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]


class PAN(nn.Module):
    """Path aggregation network (top-down + bottom-up)."""

    def __init__(self, channels=(256, 512, 1024), depth_mult=0.33):
        super().__init__()
        c3, c4, c5 = channels
        n = max(round(3 * depth_mult), 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c4 = C2f(c5 + c4, c4, n, False)
        self.td_c3 = C2f(c4 + c3, c3, n, False)
        self.down_c3 = Conv(c3, c3, 3, 2)
        self.bu_c4 = C2f(c3 + c4, c4, n, False)
        self.down_c4 = Conv(c4, c4, 3, 2)
        self.bu_c5 = C2f(c4 + c5, c5, n, False)
        self.out_channels = (c3, c4, c5)

    def forward(self, feats):
        p3, p4, p5 = feats
        n4 = self.td_c4(torch.cat([self.up(p5), p4], 1))
        n3 = self.td_c3(torch.cat([self.up(n4), p3], 1))
        o4 = self.bu_c4(torch.cat([self.down_c3(n3), n4], 1))
        o5 = self.bu_c5(torch.cat([self.down_c4(o4), p5], 1))
        return [n3, o4, o5]


class DetectHead(nn.Module):
    """Anchor-free decoupled head; output layout per level: ``[ltrb(4), cls(nc)]``."""

    def __init__(self, channels=(256, 512, 1024), num_classes=80, hidden=None,
                 prior_prob=0.01, reg_channels=4):
        super().__init__()
        hidden = int(hidden) if hidden else max(64, min(channels))
        self.num_classes = int(num_classes)
        self.reg_channels = int(reg_channels)
        self.stems = nn.ModuleList(Conv(c, hidden, 3, 1) for c in channels)
        self.cls_convs = nn.ModuleList(
            nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1))
            for _ in channels
        )
        self.reg_convs = nn.ModuleList(
            nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1))
            for _ in channels
        )
        self.cls_preds = nn.ModuleList(
            nn.Conv2d(hidden, self.num_classes, 1) for _ in channels
        )
        self.reg_preds = nn.ModuleList(
            nn.Conv2d(hidden, self.reg_channels, 1) for _ in channels
        )

        # sensible initialisation (stable first loss)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        bias = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, bias)
        # start from boxes of a few strides so the alignment metric is not ~0
        for conv in self.reg_preds:
            nn.init.constant_(conv.bias, 2.0)
            if self.reg_channels == 5:
                # last channel is the raw angle -> initialise it to 0 (theta = 0)
                with torch.no_grad():
                    conv.bias[4] = 0.0

    def forward(self, feats):
        outputs = []
        for i, feat in enumerate(feats):
            x = self.stems[i](feat)
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)
            outputs.append(
                torch.cat(
                    [self.reg_preds[i](reg_feat), self.cls_preds[i](cls_feat)], dim=1
                )
            )
        return outputs


class YoloDetModel(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))


def build_model(arch):
    bb = arch.get("Backbone") or {}
    nk = arch.get("Neck") or {}
    hd = arch.get("Head") or {}
    scale = bb.get("scale", "n")
    backbone = DetBackbone(scale=scale, in_channels=int(bb.get("in_channels", 3)))
    neck = PAN(
        channels=backbone.out_channels,
        depth_mult=float(nk.get("depth_mult", SCALES.get(scale, SCALES["n"])[0])),
    )
    head = DetectHead(
        channels=neck.out_channels,
        num_classes=int(hd.get("num_classes", 80)),
        hidden=hd.get("hidden"),
        prior_prob=float(hd.get("prior_prob", 0.01)),
        reg_channels=int(hd.get("reg_channels", 4)),
    )
    return YoloDetModel(backbone, neck, head)
