"""Time-series forecasting components: loss / metric + builders."""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TSLoss", "TSMetric", "build_ts_loss", "build_ts_metric"]


class TSLoss(nn.Module):
    """MSE loss (PaddleTS default ``F.mse_loss``)."""

    def __init__(self, loss="mse", **kwargs):
        super().__init__()
        self.loss = str(loss).lower()

    def forward(self, preds, batch):
        target = batch[1]
        if self.loss == "mae":
            return {"loss": F.l1_loss(preds, target)}
        return {"loss": F.mse_loss(preds, target)}


class TSMetric(object):
    def __init__(self, main_indicator="MSE", **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.se = 0.0
        self.ae = 0.0
        self.n = 0

    def __call__(self, preds, batch):
        target = batch[1]
        if not torch.is_tensor(preds):
            preds = torch.as_tensor(preds)
        d = (preds.detach().cpu().double() - target.detach().cpu().double())
        self.se += float((d ** 2).sum())
        self.ae += float(d.abs().sum())
        self.n += int(d.numel())

    def get_metric(self):
        n = max(self.n, 1)
        return {"MSE": self.se / n, "MAE": self.ae / n}


def build_ts_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    cfg.pop("name", None)
    return TSLoss(**cfg)


def build_ts_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    return TSMetric(**cfg)
