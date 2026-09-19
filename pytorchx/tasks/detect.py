"""YOLO detection task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from pytorchx.data import ClsDataset, eval_collate, train_collate
from pytorchx.models import build_model
from pytorchx.nn.graph import task_head


from pytorchx.tasks._base import _class_weights_from_config  # noqa: F401
from pytorchx.tasks._base import num_classes_of  # noqa: F401


class YoloDetTask(TaskAdapter):
    """Object detection (``Architecture.task: detect``)."""

    name = "yolo_det"

    def build_post_process(self, config):
        from pytorchx.det import build_det_postprocess

        arch = config.get("Architecture") or {}
        head = dict(arch.get("Head") or {})
        if head.get("end2end") is not None:
            head.setdefault("end2end", head["end2end"])
        return build_det_postprocess(config.get("PostProcess"), head)

    def build_model(self, config, post_process):
        from pytorchx.models import build_arch_model

        return build_arch_model(config["Architecture"], "detect")

    def build_loss(self, config, model):
        from pytorchx.det import build_det_loss

        return build_det_loss(
            config.get("Loss"),
            num_classes_of(model),
            reg_max=getattr(task_head(model), "reg_max", 1),
            use_one2one=getattr(task_head(model), "end2end", False),
            class_weights=_class_weights_from_config(config),
        )

    def build_metric(self, config):
        from pytorchx.det import build_det_metric

        return build_det_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from pytorchx.data.det import DetDataset

        train_ds = DetDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = DetDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from pytorchx.data.det import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from pytorchx.data.det import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        preds = model(images)
        metric(post_process(preds), batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        backed = arch.get("Backbone") or {}
        ds = (config.get("Train", {}).get("dataset") or {})
        return [
            "task=detect backbone={} scale={} num_classes={} imgsz={}".format(
                backed.get("name", "DetBackbone"),
                backed.get("scale"),
                head.get("num_classes"),
                (ds.get("transform") or {}).get("image_size"),
            )
        ]

