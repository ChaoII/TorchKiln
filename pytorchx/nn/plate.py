"""Original licence-plate models, re-implemented from the upstream project
``we0091234/Chinese_license_plate_detection_recognition``.

Two models live here (both are usable standalone, and the detector is also
drivable from its own YAML through :mod:`pytorchx.nn.graph`):

``PlateDetect``
    yolov5-face style **anchor-based** head with landmark regression.  The
    backbone of the shipped checkpoint is yolov5-lite (ShuffleNetV2:
    :class:`StemBlock` / :class:`ShuffleV2Block`) with a YOLOv5 PAN/``C3`` head.
    Per-anchor channel layout (identical to upstream ``models/yolo.py``)::

        [xy(2), wh(2), obj(1), kpt(2*nk), cls(nc)]   ->  no = nc + 5 + 2*nk

    The shipped plate checkpoint uses ``nc=2`` (single/double layer plate) and
    ``nk=4`` (the four plate corners) -> ``no = 15``.

``myNet_ocr_color`` (:class:`PlateRecNet`)
    CNN+CTC plate recogniser with an extra colour branch, copied from upstream
    ``plateNet.py`` / ``colorNet.py``: input ``3x48x168``, ``78`` CTC classes and
    ``5`` plate colours.  No RNN at all - it is *not* LPRNet.

Attribute names intentionally match upstream so that the released ``.pth`` /
``.pt`` weights load with ``strict=True`` (see ``tools/check_plate_models.py``).
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchx.nn.modules import Conv, register

__all__ = [
    "channel_shuffle",
    "StemBlock",
    "ShuffleV2Block",
    "BottleneckV5",
    "C3V5",
    "PlateDetect",
    "PlateRecNet",
    "PLATE_CHARSET",
    "PLATE_COLORS",
    "PLATE_REC_CFG_SMALL",
    "PLATE_REC_CFG_MEDIUM",
    "PLATE_REC_CFG_BIG",
]

# upstream ``plate_recognition/plate_rec.py``
PLATE_CHARSET = (
    "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
)
PLATE_COLORS = ["黑色", "蓝色", "绿色", "白色", "黄色"]

# upstream ``crnn_plate_recognition`` small/medium/big widths
PLATE_REC_CFG_SMALL = [8, 8, 16, 16, "M", 32, 32, "M", 48, 48, "M", 64, 128]
PLATE_REC_CFG_MEDIUM = [16, 16, 32, 32, "M", 64, 64, "M", 96, 96, "M", 128, 256]
PLATE_REC_CFG_BIG = [32, 32, 64, 64, "M", 128, 128, "M", 196, 196, "M", 256, 256]


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(b, -1, h, w)


@register
class StemBlock(nn.Module):
    """yolov5-lite stem: ``conv`` + ``concat(conv, maxpool)`` (ShuffleNetV2 style)."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        super().__init__()
        self.stem_1 = Conv(c1, c2, k, s, p, g=g, act=act)
        self.stem_2a = Conv(c2, c2 // 2, 1, 1, 0)
        self.stem_2b = Conv(c2 // 2, c2, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.stem_3 = Conv(c2 * 2, c2, 1, 1, 0)

    def forward(self, x):
        y1 = self.stem_1(x)
        y2 = self.stem_2b(self.stem_2a(y1))
        y3 = self.stem_2p(y1)
        return self.stem_3(torch.cat((y2, y3), 1))


@register
class ShuffleV2Block(nn.Module):
    """ShuffleNetV2 basic (stride 1) / downsample (stride 2) unit."""

    def __init__(self, inp, oup, stride=1):
        super().__init__()
        if not 1 <= stride <= 3:
            raise ValueError("illegal stride value")
        self.stride = int(stride)
        self.branch_features = oup // 2
        branch = self.branch_features
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, self.stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch),
                nn.SiLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if self.stride > 1 else branch, branch, 1, 1, 0, bias=False
            ),
            nn.BatchNorm2d(branch),
            nn.SiLU(inplace=True),
            nn.Conv2d(branch, branch, 3, self.stride, 1, groups=branch, bias=False),
            nn.BatchNorm2d(branch),
            nn.Conv2d(branch, branch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


@register
class BottleneckV5(nn.Module):
    """YOLOv5 (v3.x/v5.x era) bottleneck: 1x1 -> 3x3, optional shortcut."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@register
class C3V5(nn.Module):
    """YOLOv5 C3 (two 3x3 bottlenecks), as used by the shipped plate detector."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(BottleneckV5(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


@register
class PlateDetect(nn.Module):
    """yolov5-face landmark head (anchor based, 3 anchors x 3 levels).

    Training returns the reshaped per-level maps ``(bs, na, ny, nx, no)`` so a
    YOLOv5 style assigner can index them; eval returns the concatenated
    ``(bs, N, no)`` with the upstream decode applied, plus the raw maps.
    """

    def __init__(self, nc=2, anchors=(), ch=(), kpt_label=4):
        super().__init__()
        self.nc = int(nc)
        self.kpt_label = int(kpt_label)
        self.no = self.nc + 5 + 2 * self.kpt_label
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.stride = torch.zeros(self.nl)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.grid = [torch.zeros(1)] * self.nl
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = True

    def normalize_anchors(self):
        """`anchors /= stride` (upstream YOLOv5 does this right after the
        stride probe, so checkpoints store stride-normalised anchors)."""
        if float(self.stride.sum()) <= 0:
            return
        self.anchors = self.anchors / self.stride.view(-1, 1, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(
            [torch.arange(ny), torch.arange(nx)], indexing="ij"
        )
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        z = []
        k2 = 2 * self.kpt_label
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            if not self.training:
                stride = self.stride[i]
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                # ``anchor_grid`` holds the anchors in pixels (the released
                # checkpoints store ``anchors`` stride-normalised and
                # ``anchor_grid`` in pixels - exactly like upstream), which is
                # the dual of the training target (encoded in grid units).
                y = torch.full_like(x[i], 0)
                sig = list(range(5)) + list(range(5 + k2, self.no))
                y[..., sig] = x[i][..., sig].sigmoid()
                y[..., 5 : 5 + k2] = x[i][..., 5 : 5 + k2]
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * stride
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                for k in range(self.kpt_label):
                    a, b = 5 + 2 * k, 5 + 2 * k + 2
                    y[..., a:b] = y[..., a:b] * self.anchor_grid[i] + self.grid[i] * stride
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)


@register
class PlateRecNet(nn.Module):
    """``myNet_ocr_color``: CNN + CTC (78 classes) + colour head (5 classes).

    ``export=False`` (training/post-process) returns ``log_softmax`` over the
    class dim in ``(T, N, C)`` order; ``export=True`` returns raw ``(N, T, C)``
    exactly like the ONNX export of the upstream project.
    """

    def __init__(self, cfg=None, num_classes=78, color_num=5, export=False):
        super().__init__()
        self.cfg = list(cfg) if cfg else list(PLATE_REC_CFG_SMALL)
        self.feature = self._make_layers(self.cfg, True)
        self.num_classes = int(num_classes)
        self.color_num = int(color_num) if color_num else 0
        self.export = bool(export)
        if self.color_num:
            self.conv1 = nn.Conv2d(self.cfg[-1], 12, kernel_size=3, stride=2)
            self.bn1 = nn.BatchNorm2d(12)
            self.relu1 = nn.ReLU(inplace=True)
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.color_classifier = nn.Conv2d(12, self.color_num, 1, 1)
            self.color_bn = nn.BatchNorm2d(self.color_num)
            self.flatten = nn.Flatten()
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(self.cfg[-1], self.num_classes, 1, 1)

    @staticmethod
    def _make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            elif cfg[i] == "M":
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(
                    in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1
                )
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x_color = None
        if self.color_num:
            x_color = self.conv1(x)
            x_color = self.bn1(x_color)
            x_color = self.relu1(x_color)
            x_color = self.color_classifier(x_color)
            x_color = self.color_bn(x_color)
            x_color = self.flatten(self.gap(x_color))
        x = self.newCnn(self.loc(x))
        if self.export:
            conv = x.squeeze(2).transpose(2, 1)  # (b, T, C)
            return (conv, x_color) if self.color_num else conv
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        conv = x.squeeze(2).permute(2, 0, 1)  # (T, b, C)
        output = F.log_softmax(conv, dim=2)
        return (output, x_color) if self.color_num else output
