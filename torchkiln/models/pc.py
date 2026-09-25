"""Point-pillar BEV segmentation models (dense CNN, no spconv)."""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PillarSegNet", "build_pc_seg_model"]


class _ConvBNReLU(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )


class PillarSegNet(nn.Module):
    """Lightweight encoder–decoder over dense pillar BEV ``(B,C,Ny,Nx)``.

    Output: per-cell logits ``(B, nc, Ny, Nx)`` (same spatial size).
    """

    def __init__(self, in_ch=4, nc=5, base=32):
        super().__init__()
        b = int(base)
        # encoder
        self.enc1 = nn.Sequential(_ConvBNReLU(in_ch, b), _ConvBNReLU(b, b))
        self.enc2 = nn.Sequential(_ConvBNReLU(b, b * 2, s=2), _ConvBNReLU(b * 2, b * 2))
        self.enc3 = nn.Sequential(
            _ConvBNReLU(b * 2, b * 4, s=2), _ConvBNReLU(b * 4, b * 4)
        )
        self.bot = nn.Sequential(_ConvBNReLU(b * 4, b * 4, s=2), _ConvBNReLU(b * 4, b * 4))
        # decoder
        self.dec3 = _ConvBNReLU(b * 4 + b * 4, b * 2)
        self.dec2 = _ConvBNReLU(b * 2 + b * 2, b)
        self.dec1 = _ConvBNReLU(b + b, b)
        self.head = nn.Conv2d(b, nc, 1)
        self.nc = int(nc)
        self.no = int(nc)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        x = x[0] if len(x) == 1 else x[0]  # accept list or tensor
        e1 = self.enc1(x)  # 1/1
        e2 = self.enc2(e1)  # 1/2
        e3 = self.enc3(e2)  # 1/4
        bot = self.bot(e3)  # 1/8
        d3 = F.interpolate(bot, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        return self.head(d1)


def build_pc_seg_model(arch):
    """Build from ``Architecture`` (no yaml graph required)."""
    algo = str(arch.get("algorithm", "pointpillars")).lower()
    head = arch.get("Head") or {}
    if algo == "squeezesegv3":
        from torchkiln.nn.sac_rangenet import SqueezeSegV3

        nc = int(head.get("num_classes", arch.get("num_classes", 20)))
        return SqueezeSegV3(
            num_classes=nc,
            in_channels=int(head.get("in_channels", arch.get("in_channels", 5))),
            num_layers=int(head.get("num_layers", 53)),
            encoder_dropout_prob=float(head.get("encoder_dropout_prob", 0.01)),
            decoder_dropout_prob=float(head.get("decoder_dropout_prob", 0.01)),
        )
    nc = int(head.get("num_classes", arch.get("num_classes", 5)))
    in_ch = int(arch.get("in_channels", arch.get("ch", 4)))
    base = int(head.get("base_channels", arch.get("base_channels", 32)))
    return PillarSegNet(in_ch=in_ch, nc=nc, base=base)
