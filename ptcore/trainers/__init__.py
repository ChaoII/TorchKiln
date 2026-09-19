"""Per-task trainers.

Each task has its own trainer class/file; they all share the training loop in
:class:`ptcore.trainers.base.BaseTrainer`. Configuration decides which trainer is
built (``Architecture.model_family`` + ``Architecture.task``).
"""
from __future__ import absolute_import

from ptcore.trainers.attribute import AttributeTrainer
from ptcore.trainers.base import BaseTrainer, get_logger
from ptcore.trainers.classify import ClassifyTrainer
from ptcore.trainers.depth import DepthTrainer
from ptcore.trainers.detect import DetectTrainer
from ptcore.trainers.obb import ObbTrainer
from ptcore.trainers.ocr import OcrTrainer
from ptcore.trainers.plate_det import PlateDetTrainer
from ptcore.trainers.plate_rec import PlateRecTrainer
from ptcore.trainers.pose import PoseTrainer
from ptcore.trainers.pose_action import PoseActionTrainer
from ptcore.trainers.segment import SegmentTrainer
from ptcore.trainers.semantic import SemanticTrainer
from ptcore.trainers.video_cls import VideoClsTrainer

#: ``Architecture.task`` -> trainer class
TRAINER_REGISTRY = {
    "classify": ClassifyTrainer,
    "detect": DetectTrainer,
    "obb": ObbTrainer,
    "segment": SegmentTrainer,
    "pose": PoseTrainer,
    "semantic": SemanticTrainer,
    "depth": DepthTrainer,
    "plate_det": PlateDetTrainer,
    "plate_rec": PlateRecTrainer,
    "attribute": AttributeTrainer,
    "pose_action": PoseActionTrainer,
    "video_cls": VideoClsTrainer,
    # OCR family: text det / rec share one adapter (OcrTask dispatches on algorithm)
    "det": OcrTrainer,
    "rec": OcrTrainer,
    "ocr_det": OcrTrainer,
    "ocr_rec": OcrTrainer,
}

AVAILABLE = tuple(sorted(TRAINER_REGISTRY))
_OCR_TASKS = ("det", "rec", "ocr_det", "ocr_rec")


def get_trainer(task):
    """Return the trainer class for ``Architecture.task``."""
    try:
        return TRAINER_REGISTRY[task]
    except KeyError:
        raise NotImplementedError(
            "no trainer for task {!r} (available: {})".format(task, ", ".join(AVAILABLE))
        )


def build_trainer(config, **kwargs):
    """Build the trainer for a config (``model_family`` defaults to ``ocr``)."""
    arch = config.get("Architecture") or {}
    family = arch.get("model_family", "ocr")
    task = arch.get("task")
    if family == "ocr" or task in _OCR_TASKS:
        cls = OcrTrainer
    else:
        cls = get_trainer(task)
    return cls(config, **kwargs)


__all__ = [
    "BaseTrainer",
    "get_logger",
    "TRAINER_REGISTRY",
    "AVAILABLE",
    "get_trainer",
    "build_trainer",
] + sorted({c.__name__ for c in TRAINER_REGISTRY.values()})
