"""Independent (clean-room) YOLO-style models for the ptcore training platform.

Implemented from the published architecture descriptions (CSP bottleneck
stacks, SPPF, anchor-free heads) rather than copying any upstream source, so the
resulting code is not a derivative of any AGPL-licensed implementation.

Currently provided:

* ``YoloClsModel`` - image classification (backbone + classify head)

Design notes
------------
* ``Conv``      : Conv2d + BatchNorm + SiLU
* ``Bottleneck``: 1x1 -> 3x3 residual block
* ``C2f``       : cross-stage partial module holding ``n`` bottlenecks
* ``SPPF``      : spatial pyramid pooling (fast)
* scales ``n/s/m/l/x`` follow the usual (depth, width, max_channels) scheme
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

__all__ = ["build_model", "YoloClsModel", "YOLOBackbone", "ClassifyHead"]

# (depth_multiple, width_multiple, max_channels)
SCALES = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}


def autopad(k, p=None):
    return k // 2 if p is None else p


class Conv(nn.Module):
    """Standard convolution: Conv2d -> BatchNorm -> SiLU."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck with optional residual connection."""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = bool(shortcut) and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP bottleneck stack with 2 convolutions (cross-stage partial)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial pyramid pooling - fast (three chained max-pools)."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class YOLOBackbone(nn.Module):
    """Compact CSP backbone with 5 downsampling stages."""

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
        self.out_channels = c(1024)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class ClassifyHead(nn.Module):
    """Global-pool classification head."""

    def __init__(self, c1, num_classes=1000, hidden=1280, dropout=0.0):
        super().__init__()
        self.conv = Conv(c1, hidden, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.pool(self.conv(x)).flatten(1)
        return self.fc(self.drop(x))


class YoloClsModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(arch):
    """Assemble a classification model from the ``Architecture`` section."""
    bb = arch.get("Backbone") or {}
    hd = arch.get("Head") or {}
    backbone = YOLOBackbone(
        scale=bb.get("scale", "n"), in_channels=int(bb.get("in_channels", 3))
    )
    head = ClassifyHead(
        backbone.out_channels,
        num_classes=int(hd.get("num_classes", 1000)),
        hidden=int(hd.get("hidden", 1280)),
        dropout=float(hd.get("dropout", 0.0)),
    )
    return YoloClsModel(backbone, head)


def build_arch_model(arch, task=None):
    """Build the model for ``Architecture``.

    * ``yaml_file`` / ``yaml_text`` present -> YAML graph model (module zoo)
    * otherwise -> the hand-written per-task builder
    """
    if arch.get("yaml_file") or arch.get("yaml_text"):
        from torchkiln.nn.graph import build_from_arch

        return build_from_arch(arch)
    task = task or arch.get("task", "classify")
    if task == "classify":
        return build_model(arch)
    if task in ("detect", "obb"):
        from torchkiln.models.det import build_model as _m

        return _m(arch)
    if task == "segment":
        from torchkiln.models.seg import build_model as _m

        return _m(arch)
    if task == "semantic":
        from torchkiln.models.sem import build_model as _m

        return _m(arch)
    if task == "depth":
        from torchkiln.models.depth import build_model as _m

        return _m(arch)
    if task == "plate_rec":
        from torchkiln.models.plate import build_rec_model

        return build_rec_model(arch)
    if task == "attribute":
        from torchkiln.attr import build_attribute_model

        return build_attribute_model(arch)
    if task == "plate_det":
        from torchkiln.nn.graph import build_from_arch

        arch = dict(arch)
        arch.setdefault("yaml_file", "torchkiln/cfg/models/plate/yolov5n-0.5.yaml")
        return build_from_arch(arch)
    raise ValueError("No model builder for task {!r}".format(task))
