"""Per-task adapters: one module per task (ultralytics style)."""
from __future__ import absolute_import

from pytorchx.tasks.classify import YoloClsTask
from pytorchx.tasks.detect import YoloDetTask
from pytorchx.tasks.obb import YoloObbTask
from pytorchx.tasks.segment import YoloSegTask
from pytorchx.tasks.pose import YoloPoseTask
from pytorchx.tasks.semantic import YoloSemTask
from pytorchx.tasks.depth import YoloDepthTask
from pytorchx.tasks.plate_det import PlateDetTask
from pytorchx.tasks.plate_rec import PlateRecTask
from pytorchx.tasks.attribute import AttributeTask
from pytorchx.tasks.pose_action import PoseActionTask
from pytorchx.tasks.video_cls import VideoClsTask

TASK_REGISTRY = {
    "classify": YoloClsTask,
    "detect": YoloDetTask,
    "obb": YoloObbTask,
    "segment": YoloSegTask,
    "pose": YoloPoseTask,
    "semantic": YoloSemTask,
    "depth": YoloDepthTask,
    "plate_det": PlateDetTask,
    "plate_rec": PlateRecTask,
    "attribute": AttributeTask,
    "pose_action": PoseActionTask,
    "video_cls": VideoClsTask,
}

AVAILABLE = tuple(sorted(TASK_REGISTRY))


def get_task(task):
    """Return the TaskAdapter instance for ``Architecture.task``."""
    try:
        cls = TASK_REGISTRY[task]
    except KeyError:
        raise NotImplementedError(
            "task {!r} is not implemented (available: {})".format(
                task, ", ".join(AVAILABLE)
            )
        )
    return cls()

__all__ = ["TASK_REGISTRY", "AVAILABLE", "get_task"] + [c.__name__ for c in TASK_REGISTRY.values()]
