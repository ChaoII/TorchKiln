"""Pillar BEV detection models (dense CNN, CenterPoint-style heads)."""
from __future__ import absolute_import

import torch
import torch.nn as nn

from torchkiln.models.pc import _ConvBNReLU

__all__ = ["PillarDetNet", "build_det3d_model"]


class PillarDetNet(nn.Module):
    """Lightweight BEV encoder → (heatmap, 8-dof reg) at input resolution.

    Output: ``{"heatmap": (B, nc, Ny, Nx), "reg": (B, 8, Ny, Nx)}``.
    """

    def __init__(self, in_ch=4, nc=3, base=32):
        super().__init__()
        b = int(base)
        self.enc1 = nn.Sequential(_ConvBNReLU(in_ch, b), _ConvBNReLU(b, b))
        self.enc2 = nn.Sequential(_ConvBNReLU(b, b * 2, s=2), _ConvBNReLU(b * 2, b * 2))
        self.enc3 = nn.Sequential(
            _ConvBNReLU(b * 2, b * 4, s=2), _ConvBNReLU(b * 4, b * 4)
        )
        # fuse multi-scale back to full res
        self.up3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = nn.Sequential(
            _ConvBNReLU(b * 4 + b * 2 + b, b * 2),
            _ConvBNReLU(b * 2, b),
        )
        self.hm = nn.Conv2d(b, nc, 1)
        self.reg = nn.Conv2d(b, 8, 1)
        # bias init: heatmap prior (CenterPoint)
        nn.init.constant_(self.hm.bias, -2.0)
        self.nc = int(nc)
        self.no = int(nc)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        f = self.fuse(
            torch.cat(
                [self.up3(e3), self.up2(e2), e1],
                dim=1,
            )
        )
        return {"heatmap": self.hm(f), "reg": self.reg(f)}


def build_det3d_model(arch):
    algo = str(arch.get("algorithm", "pointpillars")).lower()
    head = arch.get("Head") or {}
    if algo == "centerpoint":
        from torchkiln.nn.centerpoint import CenterPointPillars

        tasks = head.get("tasks")
        if tasks:
            num_class = tuple(int(t.get("num_class", len(t.get("class_names", [])))) for t in tasks)
        else:
            nc = int(head.get("num_classes", 3))
            num_class = (1, nc - 1) if nc > 1 else (1,)
        return CenterPointPillars(
            num_class=num_class,
            point_cloud_range=head.get(
                "point_cloud_range", arch.get("point_cloud_range",
                                              [0, -39.68, -3, 69.12, 39.68, 1])
            ),
            voxel_size=head.get("voxel_size", arch.get("voxel_size", [0.16, 0.16, 4.0])),
            max_points=int(head.get("max_num_points_in_voxel", 100)),
            feat_channels=tuple(head.get("feat_channels", (64, 64))),
            in_channels=int(arch.get("in_channels", arch.get("ch", 4))),
        )
    nc = int(head.get("num_classes", arch.get("num_classes", 3)))
    in_ch = int(arch.get("in_channels", arch.get("ch", 4)))
    base = int(head.get("base_channels", arch.get("base_channels", 32)))
    return PillarDetNet(in_ch=in_ch, nc=nc, base=base)
