"""BEV-LaneDet model builder."""
from __future__ import absolute_import

__all__ = ["build_lane_bev_model"]


def build_lane_bev_model(arch):
    from torchkiln.nn.bev_lanedet import BEVLaneDet

    head = arch.get("Head") or {}
    return BEVLaneDet(
        bev_shape=list(head.get("bev_shape", [200, 48])),
        output_2d_shape=list(head.get("output_2d_shape", [144, 256])),
        train=bool(arch.get("train", True)),
    )
