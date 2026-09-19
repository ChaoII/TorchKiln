"""YOLO instance-segmentation task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from pytorchx.data import ClsDataset, eval_collate, train_collate
from pytorchx.models import build_model
from pytorchx.nn.graph import task_head


from pytorchx.tasks._base import num_classes_of  # noqa: F401


class YoloSegTask(TaskAdapter):
    """Instance segmentation (``Architecture.task: segment``)."""

    name = "yolo_seg"

    def build_post_process(self, config):
        from pytorchx.seg import build_seg_postprocess

        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        return build_seg_postprocess(
            config.get("PostProcess"), nm=head.get("nm"), reg_max=head.get("reg_max")
        )

    def build_model(self, config, post_process):
        from pytorchx.models import build_arch_model

        return build_arch_model(config["Architecture"], "segment")

    def build_loss(self, config, model):
        from pytorchx.seg import build_seg_loss

        return build_seg_loss(
            config.get("Loss"),
            model.num_classes,
            nm=model.nm,
            reg_max=getattr(task_head(model), "reg_max", 1),
        )

    def build_metric(self, config):
        from pytorchx.seg import build_seg_metric

        return build_seg_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from pytorchx.data.seg import SegDataset

        train_ds = SegDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = SegDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from pytorchx.data.seg import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from pytorchx.data.seg import eval_collate

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
        ds = config.get("Train", {}).get("dataset") or {}
        return [
            "task=segment backbone={} scale={} num_classes={} nm={} imgsz={}".format(
                backed.get("name", "DetBackbone"),
                backed.get("scale"),
                head.get("num_classes"),
                head.get("nm"),
                (ds.get("transform") or {}).get("image_size"),
            )
        ]

