"""YOLO classification task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from torchkiln.data import ClsDataset, eval_collate, train_collate
from torchkiln.models import build_model
from torchkiln.nn.graph import task_head


from torchkiln.tasks._cls import (
    build_cls_loss,
    build_cls_metric,
    build_cls_post_process,
    is_multi_label_loss_cfg,
    num_classes_of,
)
from torchkiln.tasks._base import num_classes_of  # noqa: F401


class YoloClsTask(TaskAdapter):
    """Image classification (``Architecture.task: classify``).

    多标签:``-o Loss.multi_label=true`` 自动切 BCE + sigmoid 阈值后处理 + mAP。
    """

    name = "yolo_cls"

    def build_post_process(self, config):
        return build_cls_post_process(config.get("PostProcess"), config.get("Loss"))

    def build_model(self, config, post_process):
        from torchkiln.models import build_arch_model

        return build_arch_model(config["Architecture"], "classify")

    def build_loss(self, config, model):
        loss = build_cls_loss(config.get("Loss"))
        # 多标签一键开关:trainer 的 best 指标在 Metric.main_indicator;默认 acc 时改 mAP
        if is_multi_label_loss_cfg(config.get("Loss")):
            mcfg = config.setdefault("Metric", {})
            if str(mcfg.get("main_indicator") or "acc") in ("acc", "top1", "hmean", ""):
                mcfg["main_indicator"] = "mAP"
        return loss

    def build_metric(self, config):
        return build_cls_metric(config.get("Metric"), config.get("Loss"))

    def build_datasets(self, config, logger):
        train_ds = ClsDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = ClsDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        return train_collate(batch)

    def eval_collate(self, batch):
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
        multi = is_multi_label_loss_cfg(config.get("Loss"))
        return [
            "task=classify backbone={} scale={} num_classes={} image_size={} multi_label={}".format(
                backed.get("name", "YOLOBackbone"),
                backed.get("scale"),
                head.get("num_classes"),
                (config.get("Train", {}).get("dataset", {}).get("transform") or {}).get(
                    "image_size"
                ),
                multi,
            )
        ]

