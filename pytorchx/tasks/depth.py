"""Depth-regression task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from pytorchx.data import ClsDataset, eval_collate, train_collate
from pytorchx.models import build_model
from pytorchx.nn.graph import task_head


from pytorchx.tasks._base import num_classes_of  # noqa: F401


class YoloDepthTask(TaskAdapter):
    """Monocular depth estimation (``Architecture.task: depth``)."""

    name = "yolo_depth"

    def build_post_process(self, config):
        from pytorchx.depth import build_depth_postprocess

        return build_depth_postprocess(config.get("PostProcess"))

    def build_model(self, config, post_process):
        from pytorchx.models import build_arch_model

        return build_arch_model(config["Architecture"], "depth")

    def build_loss(self, config, model):
        from pytorchx.depth import build_depth_loss

        return build_depth_loss(config.get("Loss"))

    def build_metric(self, config):
        from pytorchx.depth import build_depth_metric

        return build_depth_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from pytorchx.data.depth import DepthDataset

        train_ds = DepthDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = DepthDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from pytorchx.data.depth import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from pytorchx.data.depth import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        preds = model(images)
        metric(post_process(preds, size=batch[1].shape[-2:]), batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture", {}) or {}
        backed = arch.get("Backbone") or {}
        ds = config.get("Train", {}).get("dataset") or {}
        return [
            "task=depth backbone={} scale={} imgsz={}".format(
                backed.get("name", "DetBackbone"),
                backed.get("scale"),
                (ds.get("transform") or {}).get("image_size"),
            )
        ]

