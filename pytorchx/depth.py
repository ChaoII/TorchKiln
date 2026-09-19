"""Monocular depth components: loss / metric / post-process + builders.

The model predicts log-depth; depth in meters is ``exp(logit)``. Training uses a
log-space L1 term (absolute scale) plus an optional scale-invariant term.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DepthLoss",
    "DepthMetric",
    "DepthPostProcess",
    "build_depth_loss",
    "build_depth_metric",
    "build_depth_postprocess",
]


class DepthLoss(nn.Module):
    def __init__(self, silog_weight=0.0, min_depth=1e-3, **kwargs):
        super().__init__()
        self.silog_weight = float(silog_weight)
        self.min_depth = float(min_depth)

    def forward(self, preds, batch):
        target = batch[1]  # (B, 1 or H, W) depth in meters
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if preds.shape[-2:] != target.shape[-2:]:
            preds = F.interpolate(
                preds, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        valid = target > 0
        if valid.sum() == 0:
            return {"loss": preds.sum() * 0.0}
        pred_log = preds[valid]
        gt_log = torch.log(target[valid].clamp(min=self.min_depth))
        loss = F.l1_loss(pred_log, gt_log)
        if self.silog_weight > 0:
            d = pred_log - gt_log
            silog = (d * d).mean() - 0.5 * (d.mean() ** 2)
            loss = loss + self.silog_weight * silog.clamp(min=0)
        return {"loss": loss}


class DepthPostProcess(object):
    def __init__(self, min_depth=1e-3, max_depth=1e3, **kwargs):
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def __call__(self, preds, size=None):
        if size is not None and preds.shape[-2:] != tuple(size):
            preds = F.interpolate(
                preds, size=tuple(size), mode="bilinear", align_corners=False
            )
        if preds.dim() == 4:
            preds = preds[:, 0]
        return torch.exp(preds).clamp(self.min_depth, self.max_depth)


class DepthMetric(object):
    """delta1 / delta2 / delta3, AbsRel, RMSE over valid pixels."""

    def __init__(self, main_indicator="delta1", **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.n = 0
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.abs_rel = 0.0
        self.sq_err = 0.0

    def __call__(self, post_result, batch):
        pred = post_result
        if pred.dim() == 4:
            pred = pred[:, 0]
        target = batch[1]
        if target.dim() == 4:
            target = target[:, 0]
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        pred = pred.detach().cpu().numpy().astype(np.float64)
        gt = target.detach().cpu().numpy().astype(np.float64)
        valid = gt > 0
        if valid.sum() == 0:
            return
        p = pred[valid]
        g = gt[valid]
        ratio = np.maximum(p / g, g / p)
        self.n += p.size
        self.d1 += int((ratio < 1.25).sum())
        self.d2 += int((ratio < 1.25**2).sum())
        self.d3 += int((ratio < 1.25**3).sum())
        self.abs_rel += float(np.abs(p - g).sum() / g.sum())
        self.sq_err += float(((p - g) ** 2).sum())

    def get_metric(self):
        n = max(self.n, 1)
        return {
            "delta1": self.d1 / n,
            "delta2": self.d2 / n,
            "delta3": self.d3 / n,
            "abs_rel": self.abs_rel,
            "rmse": float(np.sqrt(self.sq_err / n)),
        }


def build_depth_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "DepthLoss")
    if name not in ("DepthLoss", "L1"):
        raise ValueError("Unknown depth loss: {}".format(name))
    return DepthLoss(**cfg)


def build_depth_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    return DepthMetric(**cfg)


def build_depth_postprocess(pp_cfg):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    return DepthPostProcess(**cfg)
