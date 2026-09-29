"""Per-task adapters: one module per task (ultralytics style).

⚠️ **这里必须按需加载（懒导入）**。早期版本在模块顶层 import 全部 24 个任务，导致：

  1. **缺一个可选依赖就全盘皆挂**——``panns_cls`` 需要 ``torchaudio``、``kokoro_tts``
     需要 ``kokoro``，没装这两个连 ``tkiln train -c configs/yolo/yolov8n-det.yml``
     都起不来（实测 ``ModuleNotFoundError: No module named 'torchaudio'``）。
     做 YOLO 目标检测却被一个语音任务的依赖卡死，这在容器化部署里尤其致命。
  2. 每次起进程都要 import 全部任务模块（含 3D / 点云 / TTS / 时序），白白多花数秒。

现在只登记 ``task -> (模块名, 类名)``，真正要用到的任务才 import；可选依赖缺失时，
报错也只在你**真的要跑那个任务**时才出现，并且提示怎么修。

对外接口保持不变：``TASK_REGISTRY`` / ``AVAILABLE`` / ``get_task``，
外加 ``from torchkiln.tasks import YoloDetTask`` 这类直接取类名的老用法（走 PEP 562）。
"""
from __future__ import absolute_import

import importlib

#: task(Architecture.task) -> (模块路径, 类名)
TASK_SPECS = {
    "classify": ("torchkiln.tasks.classify", "YoloClsTask"),
    "detect": ("torchkiln.tasks.detect", "YoloDetTask"),
    "obb": ("torchkiln.tasks.obb", "YoloObbTask"),
    "segment": ("torchkiln.tasks.segment", "YoloSegTask"),
    "pose": ("torchkiln.tasks.pose", "YoloPoseTask"),
    "semantic": ("torchkiln.tasks.semantic", "YoloSemTask"),
    "depth": ("torchkiln.tasks.depth", "YoloDepthTask"),
    "lane_seg": ("torchkiln.tasks.lane_seg", "LaneSegTask"),
    "lane_row": ("torchkiln.tasks.lane_row", "LaneRowTask"),
    "pc_seg": ("torchkiln.tasks.pc_seg", "PcSegTask"),
    "det3d": ("torchkiln.tasks.det3d", "Det3DTask"),
    "mono3d": ("torchkiln.tasks.mono3d", "Mono3DTask"),
    "lane_bev": ("torchkiln.tasks.lane_bev", "LaneBEVTask"),
    "plate_det": ("torchkiln.tasks.plate_det", "PlateDetTask"),
    "plate_rec": ("torchkiln.tasks.plate_rec", "PlateRecTask"),
    "attribute": ("torchkiln.tasks.attribute", "AttributeTask"),
    "pose_action": ("torchkiln.tasks.pose_action", "PoseActionTask"),
    "video_cls": ("torchkiln.tasks.video_cls", "VideoClsTask"),
    "ts_forecast": ("torchkiln.tasks.ts_forecast", "TsForecastTask"),
    "ts_anomaly": ("torchkiln.tasks.ts_anomaly", "TsAnomalyTask"),
    "ts_classify": ("torchkiln.tasks.ts_classify", "TsClassifyTask"),
    "ts_embed": ("torchkiln.tasks.ts_embed", "TsEmbedTask"),
    "ts_rul": ("torchkiln.tasks.ts_rul", "TsRulTask"),
    "kokoro_tts": ("torchkiln.tasks.kokoro_tts", "KokoroTtsTask"),
    "panns_cls": ("torchkiln.tasks.panns_cls", "PannsClsTask"),
}

#: 类名 -> task（供 __getattr__ 反查）
_CLASS_TO_TASK = {cls: task for task, (_mod, cls) in TASK_SPECS.items()}

AVAILABLE = tuple(sorted(TASK_SPECS))


def _load_task_class(task):
    """import 并取回某个任务的适配器类。"""
    mod_name, cls_name = TASK_SPECS[task]
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ImportError(
            "task {!r} requires extra dependencies that are not installed: {}\n"
            "hint: install them, or pick another task (available: {}).".format(
                task, exc, ", ".join(AVAILABLE)
            )
        ) from exc
    return getattr(mod, cls_name)


class _LazyTaskRegistry(dict):
    """``TASK_REGISTRY[task]`` 仍返回类，但**在取值那一刻**才 import 对应模块。"""

    def __missing__(self, key):
        if key not in TASK_SPECS:
            raise KeyError(key)
        cls = _load_task_class(key)
        self[key] = cls
        return cls

    def keys(self):
        return TASK_SPECS.keys()

    def items(self):
        return [(k, self[k]) for k in TASK_SPECS]

    def values(self):
        return [self[k] for k in TASK_SPECS]


#: 兼容原接口：``TASK_REGISTRY["detect"]`` -> YoloDetTask
TASK_REGISTRY = _LazyTaskRegistry()


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


def __getattr__(name):
    """PEP 562：让 ``from torchkiln.tasks import YoloDetTask`` 继续可用（按需 import）。"""
    task = _CLASS_TO_TASK.get(name)
    if task is not None:
        return _load_task_class(task)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(list(globals().keys()) + list(_CLASS_TO_TASK.values())))


__all__ = ["TASK_REGISTRY", "TASK_SPECS", "AVAILABLE", "get_task"]
# ⚠️ 类名（``YoloDetTask`` 等）**故意不放进 __all__**：``from ... import *`` 会按
#    __all__ 逐个取属性，等于强制 import 全部任务模块，懒加载就白做了。
#    需要具体类时用显式导入（``from torchkiln.tasks import YoloDetTask``，走 __getattr__ 按需加载）。
