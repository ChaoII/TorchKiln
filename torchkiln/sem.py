"""Semantic segmentation components: loss / metric / post-process + builders."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SemLoss",
    "SemMetric",
    "SemPostProcess",
    "build_sem_loss",
    "build_sem_metric",
    "build_sem_postprocess",
]


class SemLoss(nn.Module):
    def __init__(self, ignore_index=255, dice_weight=0.0, label_smoothing=0.0, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=float(label_smoothing)
        )
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)

    def _dice(self, logits, target):
        num_classes = logits.shape[1]
        probs = logits.softmax(1)
        valid = (target != self.ignore_index).unsqueeze(1)
        one_hot = F.one_hot(
            target.clamp(0, num_classes - 1), num_classes
        ).permute(0, 3, 1, 2).float()
        probs = probs * valid
        one_hot = one_hot * valid
        inter = (probs * one_hot).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2 * inter + 1.0) / (denom + 1.0)
        return 1.0 - dice.mean()

    def forward(self, preds, batch):
        mask = batch[1]
        if preds.shape[-2:] != mask.shape[-2:]:
            preds = F.interpolate(
                preds, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = self.ce(preds, mask.long())
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self._dice(preds, mask.long())
        return {"loss": loss}


class SemPostProcess(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, preds, size=None):
        if size is not None and preds.shape[-2:] != tuple(size):
            preds = F.interpolate(
                preds, size=tuple(size), mode="bilinear", align_corners=False
            )
        return preds.argmax(1)  # (B, H, W)


class SemMetric(object):
    def __init__(self, num_classes=None, main_indicator="mIoU", ignore_index=255, **kwargs):
        self.num_classes = num_classes
        self.main_indicator = main_indicator
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self):
        self.conf = None
        self.correct = 0
        self.total = 0

    def __call__(self, post_result, batch):
        pred = post_result
        target = batch[1]
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred)
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)
        pred = pred.detach().cpu().numpy().astype(np.int64).ravel()
        target = target.detach().cpu().numpy().astype(np.int64).ravel()
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        n_cls = self.num_classes or (
            int(max(pred.max(initial=0), target.max(initial=0))) + 1
        )
        if self.conf is None:
            self.conf = np.zeros((n_cls, n_cls), dtype=np.int64)
        elif self.conf.shape[0] < n_cls:
            pad = n_cls - self.conf.shape[0]
            self.conf = np.pad(self.conf, ((0, pad), (0, pad)))
        for t, p in zip(target, pred):
            if t < self.conf.shape[0] and p < self.conf.shape[1]:
                self.conf[t, p] += 1
        self.correct += int((pred == target).sum())
        self.total += int(pred.size)

    def get_metric(self):
        if self.conf is None or self.conf.sum() == 0:
            return {"mIoU": 0.0, "acc": 0.0}
        inter = np.diag(self.conf).astype(np.float64)
        union = (
            self.conf.sum(0) + self.conf.sum(1) - np.diag(self.conf)
        ).astype(np.float64)
        present = union > 0
        iou = np.zeros_like(union)
        iou[present] = inter[present] / union[present]
        miou = float(iou[present].mean()) if present.any() else 0.0
        acc = float(self.correct / max(self.total, 1))
        return {"mIoU": miou, "acc": acc}


def build_sem_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "SemLoss")
    if name not in ("SemLoss", "CrossEntropy"):
        raise ValueError("Unknown sem loss: {}".format(name))
    return SemLoss(**cfg)


def build_sem_metric(metric_cfg, num_classes=None):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    cfg.setdefault("num_classes", num_classes)
    return SemMetric(**cfg)


def build_sem_postprocess(pp_cfg):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    return SemPostProcess(**cfg)
