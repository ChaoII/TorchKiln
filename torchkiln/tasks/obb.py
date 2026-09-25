"""YOLO oriented-bbox task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from torchkiln.data import ClsDataset, eval_collate, train_collate
from torchkiln.models import build_model
from torchkiln.nn.graph import task_head


from torchkiln.tasks._base import num_classes_of  # noqa: F401
from torchkiln.tasks.detect import YoloDetTask  # noqa: F401


class YoloObbTask(YoloDetTask):
    """Oriented object detection (``Architecture.task: obb``).

    Same dataset / post-process / metric plumbing as detection, but rotated
    boxes (``xywhr``) and a ProbIoU-based loss.
    """

    name = "yolo_obb"

    def build_loss(self, config, model):
        from torchkiln.det import build_obb_loss

        head = task_head(model)
        return build_obb_loss(
            config.get("Loss"),
            num_classes_of(model),
            reg_max=getattr(head, "reg_max", 1),
            reg_layout=getattr(head, "reg_layout", "ltrb_angle"),
            ne=getattr(head, "ne", 1),
            use_one2one=getattr(head, "end2end", False),
        )

    def summary_lines(self, config, global_config, post_process):
        lines = super().summary_lines(config, global_config, post_process)
        return [line.replace("task=detect", "task=obb") for line in lines]

