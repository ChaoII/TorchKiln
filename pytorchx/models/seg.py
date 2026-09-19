"""Instance segmentation model: detection head + mask coefficients + prototypes.

Head output per level: ``[ltrb(4), cls(nc), mask_coeff(nm)]``; the prototype
branch produces ``nm`` mask bases at stride 4. A detection's mask is
``sigmoid(coeff @ protos)``.
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

from pytorchx.models import Conv, SCALES
from pytorchx.models.det import DetectHead, PAN, DetBackbone

__all__ = ["build_model", "YoloSegModel", "SegmentHead"]


class SegmentHead(DetectHead):
    def __init__(
        self,
        channels=(256, 512, 1024),
        num_classes=80,
        hidden=None,
        prior_prob=0.01,
        nm=32,
    ):
        super().__init__(
            channels=channels,
            num_classes=num_classes,
            hidden=hidden,
            prior_prob=prior_prob,
            reg_channels=4,
        )
        hidden = int(hidden) if hidden else max(64, min(channels))
        self.nm = int(nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(hidden, hidden, 3, 1),
                nn.Conv2d(hidden, self.nm, 1),
            )
            for _ in channels
        )
        # prototypes are produced at stride 4 (P3 is stride 8, so up-sample x2)
        self.proto = nn.Sequential(
            Conv(channels[0], hidden, 3, 1),
            Conv(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, self.nm, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        for m in self.proto.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats):
        outs = []
        for i, feat in enumerate(feats):
            x = self.stems[i](feat)
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)
            outs.append(
                torch.cat(
                    [
                        self.reg_preds[i](reg_feat),
                        self.cls_preds[i](cls_feat),
                        self.cv4[i](x),
                    ],
                    dim=1,
                )
            )
        protos = self.proto(feats[0])
        return outs, protos


class YoloSegModel(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.num_classes = head.num_classes
        self.nm = head.nm

    def forward(self, x):
        feats, protos = self.head(self.neck(self.backbone(x)))
        return {"feats": feats, "protos": protos}


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
    head = SegmentHead(
        channels=neck.out_channels,
        num_classes=int(hd.get("num_classes", 80)),
        hidden=hd.get("hidden"),
        prior_prob=float(hd.get("prior_prob", 0.01)),
        nm=int(hd.get("nm", 32)),
    )
    return YoloSegModel(backbone, neck, head)
