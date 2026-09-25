"""Per-task adapters: one module per task (ultralytics style)."""
from __future__ import absolute_import

from torchkiln.tasks.classify import YoloClsTask
from torchkiln.tasks.detect import YoloDetTask
from torchkiln.tasks.obb import YoloObbTask
from torchkiln.tasks.segment import YoloSegTask
from torchkiln.tasks.pose import YoloPoseTask
from torchkiln.tasks.semantic import YoloSemTask
from torchkiln.tasks.depth import YoloDepthTask
from torchkiln.tasks.lane_seg import LaneSegTask
from torchkiln.tasks.lane_row import LaneRowTask
from torchkiln.tasks.pc_seg import PcSegTask
from torchkiln.tasks.det3d import Det3DTask
from torchkiln.tasks.lane_bev import LaneBEVTask
from torchkiln.tasks.plate_det import PlateDetTask
from torchkiln.tasks.plate_rec import PlateRecTask
from torchkiln.tasks.attribute import AttributeTask
from torchkiln.tasks.pose_action import PoseActionTask
from torchkiln.tasks.video_cls import VideoClsTask

TASK_REGISTRY = {
    "classify": YoloClsTask,
    "detect": YoloDetTask,
    "obb": YoloObbTask,
    "segment": YoloSegTask,
    "pose": YoloPoseTask,
    "semantic": YoloSemTask,
    "depth": YoloDepthTask,
    "lane_seg": LaneSegTask,
    "lane_row": LaneRowTask,
    "pc_seg": PcSegTask,
    "det3d": Det3DTask,
    "lane_bev": LaneBEVTask,
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
