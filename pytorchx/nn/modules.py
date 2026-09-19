"""PyTorch-YOLO building blocks (independent re-implementation).

A *module zoo* plus a registry, so model families can be expressed as YAML
graphs (``backbone`` / ``head`` layer lists) exactly like the upstream design,
but with our own code and APIs.

Head output contract (kept identical to our hand-written models so the task
adapters / losses / post-processing are unchanged):

* ``Detect`` / ``OBB``  : list of per-level ``[reg(4|5), cls(nc)]`` tensors
* ``Segment``           : ``(list of [reg, cls, coeff(nm)], protos)``
* ``Classify``          : single ``(B, nc)`` tensor
* ``Pose``              : list of per-level ``[reg(4), cls(nc), kpt(nk*3)]``
"""
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "autopad",
    "make_divisible",
    "Conv",
    "DWConv",
    "ConvTranspose",
    "GhostConv",
    "Bottleneck",
    "C1",
    "C2",
    "C2f",
    "C3",
    "C3k2",
    "C3k",
    "PSA",
    "C2PSA",
    "Attention",
    "SPP",
    "SPPF",
    "Concat",
    "Proto",
    "Classify",
    "Detect",
    "Segment",
    "OBB",
    "Pose",
    "REGISTRY",
    "register",
    "get_module",
]


# ----------------------------------------------------------------- utilities
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x, divisor=8):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max().item())
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


REGISTRY = {}


def register(cls=None, name=None):
    """Decorator: ``@register`` / ``@register(name="C2f")``."""

    def _wrap(c):
        REGISTRY[name or c.__name__] = c
        return c

    if cls is None:
        return _wrap
    return _wrap(cls)


def get_module(name):
    if name in REGISTRY:
        return REGISTRY[name]
    if name.startswith("nn."):
        return getattr(nn, name.split(".", 1)[1])
    raise KeyError("Unknown module: {}".format(name))


# -------------------------------------------------------------------- blocks
@register
class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if isinstance(g, bool):  # YAMLs may use ``g: true`` meaning "default"
            g = 1 if g else 1
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=int(g), dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


@register
class DWConv(Conv):
    """Depthwise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


@register
class ConvTranspose(nn.Module):
    def __init__(self, c1, c2, k=2, s=2, p=0, bn=False, act=nn.SiLU()):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _bn_conv_fuse(conv, bn):
    """Return a fuse_conv(conv, bn) tuple ready for ``Conv.forward_fuse``."""
    fusedconv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = (
        torch.zeros(conv.weight.shape[0], device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


@register
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


@register
class C1(nn.Module):
    """Cross-conv-line: one 1x1 + ``n`` 3x3 convs."""

    def __init__(self, c1, c2, n=1, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.m = nn.Sequential(*(Conv(c_, c_, 3, 1, g=g) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


@register
class C2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


@register
class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions (YOLOv8 family)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


@register
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


@register
class C3k(C3):
    """C3 with a configurable bottleneck kernel size."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


@register
class C3k2(C2f):
    """YOLO11 CSP block: outer C2f, inner C3k bottlenecks (or PSA attention)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


@register
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


@register
class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


@register
class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


@register
class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


@register
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


@register
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, n=3, *args, **kwargs):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        return self.cv2(torch.cat(y, 1))


@register
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# --------------------------------------------------------------------- heads
class _BaseHead(nn.Module):
    """Detection-style head, one branch set per feature level.

    Output per level is ``[reg(reg_channels), cls(nc)]`` — the same contract as
    our hand-written heads, so losses / post-processing are unchanged.
    ``reg_max > 1`` enables the DFL parameterisation (``reg_channels = 4*reg_max``).
    """

    def __init__(self, nc=80, ch=(), reg_channels=4, hidden=None, cls_extra=0, legacy=True):
        super().__init__()
        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_channels = int(reg_channels)
        self.cls_extra = int(cls_extra)
        self.legacy = bool(legacy)
        self.no = self.reg_channels + self.nc + self.cls_extra
        c2 = max(16, ch[0] // 4, self.reg_channels)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.reg_channels, 1)
            )
            for x in ch
        )
        if self.legacy:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3),
                    Conv(c3, c3, 3),
                    nn.Conv2d(c3, self.nc + self.cls_extra, 1),
                )
                for x in ch
            )
        else:
            # ultralytics new head (yolo11/12/26): depthwise classifier tower
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc + self.cls_extra, 1),
                )
                for x in ch
            )
        self.bias_init()

    def bias_init(self):
        for a, b in zip(self.cv2, self.cv3):
            nn.init.constant_(a[-1].bias, 2.0)
            if self.reg_channels == 5:
                with torch.no_grad():
                    a[-1].bias[4] = 0.0
            with torch.no_grad():
                b[-1].bias[: self.nc] = -math.log((1 - 0.01) / 0.01)
                if self.cls_extra:
                    b[-1].bias[self.nc :] = 0.0

    def _tower(self, feats, extra=None, order="rc"):
        outs = []
        for i, f in enumerate(feats):
            reg = self.cv2[i](f)
            cls = self.cv3[i](f)
            parts = []
            for tag in order:
                if tag == "r":
                    parts.append(reg)
                elif tag == "c":
                    parts.append(cls)
                elif tag == "x" and extra is not None:
                    parts.append(extra[i](f))
            outs.append(torch.cat(parts, 1))
        return outs


@register
class Detect(_BaseHead):
    """Anchor-free detection head. ``reg_max > 1`` uses DFL regression."""

    def __init__(self, nc=80, ch=(), hidden=None, reg_max=1):
        self.reg_max = int(reg_max)
        super().__init__(
            nc=nc, ch=ch, reg_channels=4 * int(reg_max), hidden=hidden
        )
        self.stride = torch.zeros(self.nl)

    def forward(self, x):
        return self._tower(x, order="rc")


@register
class Detect26(nn.Module):
    """YOLO26-style head: DFL-free box branch + depthwise classification branch.

    Layout per level: ``[reg(4*reg_max), cls(nc)]``. The classification branch is
    built from depthwise + pointwise stages, which is what the released YOLO26 /
    updated YOLO11 checkpoints use.
    """

    def __init__(self, nc=80, ch=(), reg_max=1, hidden=None):
        super().__init__()
        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_max = int(reg_max)
        self.no = 4 * self.reg_max + self.nc
        c2 = max(16, ch[0] // 4, 4 * self.reg_max)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        for a in self.cv2:
            nn.init.constant_(a[-1].bias, 1.0)
        for b in self.cv3:
            nn.init.constant_(b[-1].bias, -math.log((1 - 0.01) / 0.01))
        self.stride = torch.zeros(self.nl)

    def forward(self, x):
        return [
            torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)
        ]


@register
class Detect10(Detect26):
    """End-to-end (NMS-free) head: one-to-many + one-to-one branches.

    Training returns ``(one2many_feats, one2one_feats)``; evaluation returns only
    the one-to-one branch, so no NMS is required at inference.
    """

    def __init__(self, nc=80, ch=(), reg_max=1, hidden=None):
        super().__init__(nc=nc, ch=ch, reg_max=reg_max, hidden=hidden)
        self.end2end = True
        c2 = max(16, ch[0] // 4, 4 * self.reg_max)
        c3 = max(ch[0], min(self.nc, 100))
        self.one2one_cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.one2one_cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        for a in self.one2one_cv2:
            nn.init.constant_(a[-1].bias, 1.0)
        for b in self.one2one_cv3:
            nn.init.constant_(b[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        one2many = [
            torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)
        ]
        one2one = [
            torch.cat(
                (self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1
            )
            for i in range(self.nl)
        ]
        if self.training:
            return one2many, one2one
        return one2one


@register
class OBB(_BaseHead):
    """Oriented boxes.

    Layout ``[reg(4*reg_max), cls(nc) + angle(ne)]`` (upstream compatible) when
    ``cls_extra``/``layout='upstream'``; our own head keeps ``[reg, angle, cls]``.
    """

    def __init__(self, nc=80, ch=(), ne=1, hidden=None, reg_max=1, layout="upstream", legacy=True):
        self.ne = int(ne)
        self.layout = layout
        self.legacy = bool(legacy)
        if layout == "upstream":
            super().__init__(
                nc=nc, ch=ch, reg_channels=4 * int(reg_max), hidden=hidden,
                cls_extra=self.ne, legacy=self.legacy,
            )
        else:
            super().__init__(
                nc=nc, ch=ch, reg_channels=4, hidden=hidden, legacy=self.legacy,
            )
            c4 = max(16, ch[0] // 4 * 4)
            self.cv4 = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
                for x in ch
            )
            for m in self.cv4:
                nn.init.constant_(m[-1].bias, 0.0)
        self.reg_max = int(reg_max)
        self.stride = torch.zeros(self.nl)

    def forward(self, x):
        if self.layout == "upstream":
            return self._tower(x, order="rc")
        return self._tower(x, extra=self.cv4, order="rcx")


@register
class Segment(_BaseHead):
    def __init__(self, nc=80, ch=(), nm=32, npr=256, hidden=None, reg_max=1):
        self.reg_max = int(reg_max)
        super().__init__(nc=nc, ch=ch, reg_channels=4 * int(reg_max), hidden=hidden)
        self.nm = int(nm)
        c4 = max(ch[0] // 4, self.nm)  # 对齐 ultralytics Segment.cv4
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1))
            for x in ch
        )
        self.proto = Proto(ch[0], self.nm, npr)

    def forward(self, x):
        outs = self._tower(x, extra=self.cv4, order="rcx")
        return outs, self.proto(x[0])


@register
class Pose(_BaseHead):
    def __init__(self, nc=80, ch=(), kpt_shape=(17, 3), hidden=None, reg_max=1):
        self.reg_max = int(reg_max)
        super().__init__(nc=nc, ch=ch, reg_channels=4 * int(reg_max), hidden=hidden)
        self.kpt_shape = tuple(kpt_shape)
        self.nk = int(kpt_shape[0]) * int(kpt_shape[1])
        c4 = max(ch[0] // 4, self.nk)  # 对齐 ultralytics Pose.cv4
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1))
            for x in ch
        )

    def forward(self, x):
        return self._tower(x, extra=self.cv4, order="rcx")


@register
class Proto(nn.Module):
    def __init__(self, c1, c2=32, c_=256):
        super().__init__()
        self.cv1 = Conv(c1, c_, 3, 1)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, 3, 1)
        self.cv3 = Conv(c_, c2, 1)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


@register
class Classify(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, dropout=0.0, hidden=1280):
        super().__init__()
        c_ = int(hidden)
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


# --------------------------------------------------------------- v9 / v12 blocks
@register
class ADown(nn.Module):
    """Average-pool based downsampling (YOLOv9)."""

    def __init__(self, c1, c2):
        super().__init__()
        c2 = c2 // 2 if c2 > 1 else c2
        self.c = c2
        self.cv1 = Conv(c1 // 2, c2, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, c2, 1, 1, 0)

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = self.cv2(F.max_pool2d(x2, 3, 2, 1))
        return torch.cat((x1, x2), 1)


@register
class RepConv(nn.Module):
    """Reparameterisable 3x3 + 1x1 conv block (YOLOv9)."""

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        self.g = g
        self.conv1 = Conv(c1, c2, 3, s, 1, g=g, d=d, act=act)
        self.conv2 = Conv(c1, c2, 1, s, 0, g=g, d=d, act=act)

    def forward(self, x):
        return self.conv2(x) + self.conv1(x)


@register
class RepConvN(RepConv):
    """RepConv without identity (used inside GELAN blocks)."""

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__(c1, c2, k, s, p, g, d, act, bn, deploy)


@register
class RepNCSP(nn.Module):
    """CSP with RepConvN bottlenecks (YOLOv9 GELAN building block)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepConvN(c_, c_, act=True) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


@register
class RepNCSPELAN4(nn.Module):
    """GELAN block used by YOLOv9."""

    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, 1))


@register
class SPPELAN(nn.Module):
    """SPP-ELAN (YOLOv9)."""

    def __init__(self, c1, c2, c3, k=5):
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3, self.cv4))
        return self.cv5(torch.cat(y, 1))


class AreaAttention(nn.Module):
    """Area attention: view the feature map as ``area`` stacked strips and attend within."""

    def __init__(self, dim, num_heads=8, area=4, kv_per_win=4, attn_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = max(1, int(num_heads))
        self.area = int(area)
        self.head_dim = dim // self.num_heads
        self.key_dim = max(1, int(self.head_dim * attn_ratio))
        self.scale = self.key_dim ** -0.5
        h = dim + self.key_dim * 2 * self.num_heads
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        a = self.area if (h % self.area == 0 and self.area > 1) else 1
        y = x.view(b, c, a, h // a, w).permute(0, 2, 1, 3, 4).reshape(b * a, c, h // a, w)
        qkv = self.qkv(y)
        ba, hc, hh, ww = qkv.shape
        n = hh * ww
        qkv = qkv.view(ba, self.num_heads, hc // self.num_heads, n)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(ba, self.num_heads * self.head_dim, hh, ww)
        out = self.proj(out)
        out = out.view(b, a, c, h // a, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)
        return out


class ABlock(nn.Module):
    def __init__(self, dim, num_heads=8, area=4, mlp_ratio=2.0, attn_ratio=0.5):
        super().__init__()
        self.attn = AreaAttention(dim, num_heads=num_heads, area=area, attn_ratio=attn_ratio)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, hid, 1), Conv(hid, dim, 1, act=False))

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


@register
class A2C2f(nn.Module):
    """Area-attention C2f (YOLOv12)."""

    def __init__(self, c1, c2, n=1, a2=True, area=4, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            ABlock(self.c, num_heads=max(1, self.c // 64), area=area, mlp_ratio=mlp_ratio)
            if a2
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# # ------------------------------------------- ghost / v8-ghost / v9e / v10 / v12 extra
@register
class GhostConv(nn.Module):
    """Cheap "ghost" convolution: half normal + half depthwise-cheap features."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False))
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


@register
class C3Ghost(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class CIB(nn.Module):
    """Compact Inverted Block (v9e / v12)."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, 1, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, 1, g=2 * c_) if not lk else RepConvN(2 * c_, 2 * c_, 3, 1, 1, 2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, 1, g=c2),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)


@register
class C2fCIB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


@register
class AConv(nn.Module):
    """ReLU conv + avg-pool (YOLOv9e)."""

    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


@register
class ELAN1(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=None):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        b1 = self.cv2(b)
        b2 = self.cv3(b1)
        return self.cv4(torch.cat((a, b, b1, b2), 1))


@register
class SCDown(nn.Module):
    """Spatial-channel downsampling (v10 / v26)."""

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))


@register
class CBLinear(nn.Module):
    """1x1 conv producing several outputs at once (v9c/e progressive aux)."""

    def __init__(self, c1, c2s):
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x).split(self.c2s, dim=1)


@register
class CBFuse(nn.Module):
    """Sum the feature maps listed by ``idx`` (upsampled to the first one)."""

    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        # each input is either a tensor or a list of splits (CBLinear); idx[i]
        # selects which split of input i to take
        last = x[-1]
        target = last if not isinstance(last, (list, tuple)) else last[0]
        out = target
        for item, j in zip(list(x[:-1]), self.idx):
            y = item[j] if isinstance(item, (list, tuple)) else item
            if y.shape[-2:] != target.shape[-2:]:
                y = F.interpolate(y, size=target.shape[-2:], mode="nearest")
            out = out + y
        return out


@register
class ResNetLayer(nn.Module):
    """Residual stage used by the ResNet classification backbones."""

    def __init__(self, c1, c2, n=1, s=1, e=1):
        super().__init__()
        self.bottleneck = Bottleneck(c1, c2, shortcut=True, e=e)

    def forward(self, x):
        if x.shape[1] != self.bottleneck.cv2.conv.out_channels:
            x = self.bottleneck(x)
        return x


@register
class TorchVision(nn.Module):
    """Wrap a torchvision classification backbone (used by ``yolov8-cls-resnet*``)."""

    def __init__(self, model="resnet50", weights=None, unwrap=1, truncate=0, split=(), pool=()):
        super().__init__()
        import torchvision

        self.m = getattr(torchvision.models, model)(weights=weights)
        for _ in range(unwrap):
            self.m = self.m.children().__iter__().__next__() if hasattr(self.m, "children") else self.m
        self.truncate = int(truncate)

    def forward(self, x):
        for i, layer in enumerate(list(self.m.children())[: self.truncate] if self.truncate else []):
            x = layer(x)
        return x


@register
class SemanticSegment(nn.Module):
    """Semantic segmentation head (v26-sem): fuse P3/P4/P5 -> per-pixel logits."""

    def __init__(self, nc=19, ch=(), hidden=128, reg_max=1):
        super().__init__()
        self.ch = list(ch)
        self.lat = nn.ModuleList(Conv(c, hidden, 1, 1) for c in self.ch)
        self.fuse = nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1))
        self.head = nn.Conv2d(hidden, nc, 1)
        self.nc = int(nc)
        self.no = int(nc)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        ref = x[0].shape[-2:]
        y = self.lat[0](x[0])
        for i in range(1, len(x)):
            y = y + F.interpolate(self.lat[i](x[i]), size=ref, mode="nearest")
        return self.head(self.fuse(y))


@register
class Depth(nn.Module):
    """Monocular depth head (v26-depth): P3/P4/P5 -> single-channel log-depth."""

    def __init__(self, nc=1, ch=(), hidden=128, reg_max=1):
        super().__init__()
        c3, c4, c5 = ch
        self.lat3 = Conv(c3, hidden, 1, 1)
        self.lat4 = Conv(c4, hidden, 1, 1)
        self.lat5 = Conv(c5, hidden, 1, 1)
        self.fuse = nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1))
        self.head = nn.Conv2d(hidden, 1, 1)
        self.no = 1

    def forward(self, x):
        p3, p4, p5 = x
        y = self.lat3(p3)
        y = y + F.interpolate(self.lat4(p4), scale_factor=2, mode="nearest")
        y = y + F.interpolate(self.lat5(p5), scale_factor=4, mode="nearest")
        return self.head(self.fuse(y))


def _dw_cls_branch(x, c3, out):
    """Depthwise + pointwise classification branch (YOLO26 style)."""
    return nn.Sequential(
        nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
        nn.Conv2d(c3, out, 1),
    )


@register
class SegmentU(nn.Module):
    """YOLO26-style instance-segmentation head: ``[reg, cls, coeff]`` + prototypes."""

    def __init__(self, nc=80, ch=(), nm=32, npr=256, reg_max=1, hidden=None):
        super().__init__()
        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_max = int(reg_max)
        self.nm = int(nm)
        self.no = 4 * self.reg_max + self.nc + self.nm
        c2 = max(16, ch[0] // 4, 4 * self.reg_max)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(_dw_cls_branch(x, c3, self.nc) for x in ch)
        c4 = max(ch[0] // 4, self.nm)  # 对齐 ultralytics Segment.cv4
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1))
            for x in ch
        )
        self.proto = Proto(ch[0], self.nm, npr)
        for a in self.cv2:
            nn.init.constant_(a[-1].bias, 1.0)
        for b in self.cv3:
            nn.init.constant_(b[-1].bias, -math.log((1 - 0.01) / 0.01))
        self.stride = torch.zeros(self.nl)

    def forward(self, x):
        outs = [
            torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4[i](x[i])), 1)
            for i in range(self.nl)
        ]
        return outs, self.proto(x[0])


@register
class OBBU(nn.Module):
    """YOLO26-style oriented head: ``[reg(4*reg_max), cls(nc), angle(ne)]``."""

    def __init__(self, nc=80, ch=(), ne=1, reg_max=1, hidden=None, layout=None, legacy=True):
        super().__init__()
        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_max = int(reg_max)
        self.ne = int(ne)
        self.legacy = bool(legacy)
        self.no = 4 * self.reg_max + self.nc + self.ne
        c2 = max(16, ch[0] // 4, 4 * self.reg_max)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(_dw_cls_branch(x, c3, self.nc) for x in ch)
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
            for x in ch
        )
        for a in self.cv2:
            nn.init.constant_(a[-1].bias, 1.0)
        for b in self.cv3:
            with torch.no_grad():
                b[-1].bias[: self.nc] = -math.log((1 - 0.01) / 0.01)
        for m in self.cv4:
            nn.init.constant_(m[-1].bias, 0.0)
        self.stride = torch.zeros(self.nl)

    def forward(self, x):
        return [
            torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4[i](x[i])), 1)
            for i in range(self.nl)
        ]


@register
class PoseU(nn.Module):
    """YOLO26-style pose head: ``[reg, cls, kpt(nk*3)]``."""

    def __init__(self, nc=80, ch=(), kpt_shape=(17, 3), reg_max=1, hidden=None):
        super().__init__()
        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_max = int(reg_max)
        self.kpt_shape = tuple(kpt_shape)
        self.nk = int(kpt_shape[0]) * int(kpt_shape[1])
        self.no = 4 * self.reg_max + self.nc + self.nk
        c2 = max(16, ch[0] // 4, 4 * self.reg_max)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(_dw_cls_branch(x, c3, self.nc) for x in ch)
        c4 = max(ch[0] // 4, self.nk)  # 对齐 ultralytics Pose.cv4
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1))
            for x in ch
        )
        for a in self.cv2:
            nn.init.constant_(a[-1].bias, 1.0)
        for b in self.cv3:
            nn.init.constant_(b[-1].bias, -math.log((1 - 0.01) / 0.01))
        self.stride = torch.zeros(self.nl)

    def forward(self, x):
        return [
            torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4[i](x[i])), 1)
            for i in range(self.nl)
        ]


# YOLO26-style aliases: the detection-style heads above keep the same output
# layouts, so the seg/obb/pose variants can reuse them directly.
OBB26 = OBB
Segment26 = Segment
Pose26 = Pose
REGISTRY["OBB26"] = OBB
REGISTRY["Segment26"] = Segment
REGISTRY["Pose26"] = Pose


# ---------------------------------------------------------------- name mapping
# Upstream (8.4.x) uses the YOLO26-style heads under the plain names; the
# DFL-capable variants stay available as ``*DFL``.
DetectDFL = Detect
SegmentDFL = Segment
OBBDFL = OBB
PoseDFL = Pose
REGISTRY["Detect"] = Detect26
REGISTRY["Segment"] = SegmentU
REGISTRY["OBB"] = OBBU
REGISTRY["Pose"] = PoseU
REGISTRY["DetectDFL"] = DetectDFL
REGISTRY["SegmentDFL"] = SegmentDFL
REGISTRY["OBBDFL"] = OBBDFL
REGISTRY["PoseDFL"] = PoseDFL
REGISTRY["Detect26"] = Detect26
REGISTRY["Segment26"] = SegmentU
REGISTRY["OBB26"] = OBBU
REGISTRY["Pose26"] = PoseU
REGISTRY["v10Detect"] = Detect10
