"""YOLO instance-segmentation task: loss / metric / post-process / task adapter."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from torchkiln.data import ClsDataset, eval_collate, train_collate
from torchkiln.models import build_model
from torchkiln.nn.graph import task_head


from torchkiln.tasks._base import num_classes_of  # noqa: F401


class YoloSegTask(TaskAdapter):
    """Instance segmentation (``Architecture.task: segment``)."""

    name = "yolo_seg"

    def build_post_process(self, config):
        from torchkiln.seg import build_seg_postprocess

        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        return build_seg_postprocess(
            config.get("PostProcess"), nm=head.get("nm"), reg_max=head.get("reg_max")
        )

    def build_model(self, config, post_process):
        from torchkiln.models import build_arch_model

        return build_arch_model(config["Architecture"], "segment")

    def build_loss(self, config, model):
        from torchkiln.seg import build_seg_loss

        head = task_head(model)
        loss_cfg = dict(config.get("Loss") or {})
        # yolo26 端到端分割头:启用 one2many+one2one 双分支(对齐 ultralytics E2ELoss)
        if getattr(head, "end2end", False) and hasattr(head, "one2one_cv2"):
            loss_cfg.setdefault("use_one2one", True)
            loss_cfg.setdefault("one2one_topk", 7)
            loss_cfg.setdefault("one2one_topk2", 1)
            loss_cfg.setdefault("e2e_gain_schedule", True)
            loss_cfg.setdefault("e2e_o2m", 0.8)
            loss_cfg.setdefault("e2e_final_o2m", 0.1)
        return build_seg_loss(
            loss_cfg,
            model.num_classes,
            nm=model.nm,
            reg_max=getattr(head, "reg_max", 1),
            epochs=int((config.get("Global") or {}).get("epoch_num", 0) or 0),
        )

    def build_metric(self, config):
        from torchkiln.seg import build_seg_metric

        return build_seg_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from torchkiln.data.seg import SegDataset

        train_ds = SegDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = SegDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.seg import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.seg import eval_collate

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

