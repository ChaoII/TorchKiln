"""YOLO pose task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from pytorchx.data import ClsDataset, eval_collate, train_collate
from pytorchx.models import build_model
from pytorchx.nn.graph import task_head


from pytorchx.tasks._base import num_classes_of  # noqa: F401


class YoloPoseTask(TaskAdapter):
    """Keypoint / pose estimation (``Architecture.task: pose``)."""

    name = "yolo_pose"

    def _kpt_shape(self, config):
        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        return tuple(head.get("kpt_shape", (17, 3)))

    def build_post_process(self, config):
        from pytorchx.pose import build_pose_postprocess

        arch = config.get("Architecture") or {}
        head = arch.get("Head") or {}
        return build_pose_postprocess(
            config.get("PostProcess"),
            self._kpt_shape(config),
            reg_max=head.get("reg_max"),
        )

    def build_model(self, config, post_process):
        from pytorchx.models import build_arch_model

        return build_arch_model(config["Architecture"], "pose")

    def build_loss(self, config, model):
        from pytorchx.pose import build_pose_loss

        return build_pose_loss(
            config.get("Loss"),
            num_classes_of(model),
            self._kpt_shape(config),
            reg_max=getattr(task_head(model), "reg_max", 1),
        )

    def build_metric(self, config):
        from pytorchx.pose import build_pose_metric

        return build_pose_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from pytorchx.data.pose import PoseDataset

        train_ds = PoseDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = PoseDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from pytorchx.data.pose import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from pytorchx.data.pose import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        metric(post_process(model(images)), batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        ds = config.get("Train", {}).get("dataset") or {}
        return [
            "task=pose yaml={} num_classes={} kpt_shape={} imgsz={}".format(
                arch.get("yaml_file"),
                head.get("num_classes"),
                head.get("kpt_shape", (17, 3)),
                (ds.get("transform") or {}).get("image_size"),
            )
        ]

