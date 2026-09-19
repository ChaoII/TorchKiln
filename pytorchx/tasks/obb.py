"""YOLO oriented-bbox task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from pytorchx.data import ClsDataset, eval_collate, train_collate
from pytorchx.models import build_model
from pytorchx.nn.graph import task_head


from pytorchx.tasks._base import num_classes_of  # noqa: F401
from pytorchx.tasks.detect import YoloDetTask  # noqa: F401


class YoloObbTask(YoloDetTask):
    """Oriented object detection (``Architecture.task: obb``).

    Same dataset / post-process / metric plumbing as detection, but rotated
    boxes (``xywhr``) and a ProbIoU-based loss.
    """

    name = "yolo_obb"

    def build_loss(self, config, model):
        from pytorchx.det import build_obb_loss

        return build_obb_loss(
            config.get("Loss"),
            num_classes_of(model),
            reg_max=getattr(task_head(model), "reg_max", 1),
            reg_layout=getattr(task_head(model), "reg_layout", "ltrb_angle"),
            ne=getattr(task_head(model), "ne", 1),
        )

    def summary_lines(self, config, global_config, post_process):
        lines = super().summary_lines(config, global_config, post_process)
        return [line.replace("task=detect", "task=obb") for line in lines]

