"""Plate model builders (detection goes through the YAML graph, recognition here)."""

from __future__ import absolute_import

import os

from torchkiln.nn.plate import PLATE_CHARSET, PLATE_COLORS, PlateRecNet

__all__ = ["build_rec_model", "PLATE_DET_YAML"]

PLATE_DET_YAML = os.path.join("torchkiln", "cfg", "models", "plate", "yolov5n-0.5.yaml")


def build_rec_model(arch):
    """``myNet_ocr_color`` from ``Architecture`` (``Head.cfg`` / ``num_classes``)."""
    head = (arch or {}).get("Head") or {}
    cfg = head.get("cfg") or None
    num_classes = int(head.get("num_classes") or len(PLATE_CHARSET))
    color_num = head.get("color_num")
    color_num = len(PLATE_COLORS) if color_num is None else int(color_num)
    return PlateRecNet(
        cfg=cfg,
        num_classes=num_classes,
        color_num=color_num,
        export=bool(head.get("export", False)),
    )
