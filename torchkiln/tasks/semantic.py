"""Semantic-segmentation task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from torchkiln.data import ClsDataset, eval_collate, train_collate
from torchkiln.models import build_model
from torchkiln.nn.graph import task_head


from torchkiln.tasks._base import num_classes_of  # noqa: F401


class YoloSemTask(TaskAdapter):
    """Semantic segmentation (``Architecture.task: semantic``)."""

    name = "yolo_sem"

    def build_post_process(self, config):
        from torchkiln.sem import build_sem_postprocess

        return build_sem_postprocess(config.get("PostProcess"))

    def build_model(self, config, post_process):
        from torchkiln.models import build_arch_model

        return build_arch_model(config["Architecture"], "semantic")

    def build_loss(self, config, model):
        from torchkiln.sem import build_sem_loss

        return build_sem_loss(config.get("Loss"))

    def build_metric(self, config):
        from torchkiln.sem import build_sem_metric

        arch = config.get("Architecture", {}) or {}
        num_classes = (arch.get("Head") or {}).get("num_classes")
        return build_sem_metric(config.get("Metric"), num_classes=num_classes)

    def build_datasets(self, config, logger):
        from torchkiln.data.sem import SemDataset

        train_ds = SemDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = SemDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.sem import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.sem import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        preds = model(images)
        # logits are at stride 4 -> upsample back to the mask resolution
        result = post_process(preds, size=batch[1].shape[-2:])
        metric(result, batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        backed = arch.get("Backbone") or {}
        ds = config.get("Train", {}).get("dataset") or {}
        return [
            "task=semantic backbone={} scale={} num_classes={} imgsz={}".format(
                backed.get("name", "DetBackbone"),
                backed.get("scale"),
                head.get("num_classes"),
                (ds.get("transform") or {}).get("image_size"),
            )
        ]

