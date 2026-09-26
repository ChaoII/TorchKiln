"""Time-series forecasting components: loss / metric + builders."""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TSLoss", "TSQuantileLoss", "TSNLLLoss", "TSMetric",
           "build_ts_loss", "build_ts_metric"]


class TSLoss(nn.Module):
    """Point-forecast loss (MSE / MAE, PaddleTS default ``F.mse_loss``)."""

    def __init__(self, loss="mse", **kwargs):
        super().__init__()
        self.loss = str(loss).lower()

    def forward(self, preds, batch):
        target = batch[1]
        if self.loss == "mae":
            return {"loss": F.l1_loss(preds, target)}
        return {"loss": F.mse_loss(preds, target)}


class TSQuantileLoss(nn.Module):
    """Pinball (quantile) loss for TFT: ``preds`` is ``(B,H,D,Q)``.

    对齐 PaddleTS ``distributions/likelihood.py::QuantileRegression.loss``:
      errors = target.unsqueeze(-1) - preds
      loss   = 2 * max((q-1)*errors, q*errors)   # 乘 2
      q_loss = (sum over Q) . mean over D . mean over B,H   # 对分位数求和(非平均)
    即框架旧实现的 2*Q=6 倍;若用 mean-over-Q 会导致梯度/等效学习率差 Q 倍。
    """

    def __init__(self, quantiles=None, **kwargs):
        super().__init__()
        self.quantiles = list(quantiles) if quantiles else [0.1, 0.5, 0.9]

    def forward(self, preds, batch):
        target = batch[1]  # (B, H, D)
        q = torch.tensor(self.quantiles, device=preds.device, dtype=preds.dtype)
        errors = target.unsqueeze(-1) - preds        # (B, H, D, Q)
        pinball = torch.maximum((q - 1.0) * errors, q * errors)  # (B,H,D,Q)
        losses = 2.0 * pinball.sum(dim=-1)           # 对分位数求和 -> (B,H,D)
        return {"loss": losses.mean()}


class TSNLLLoss(nn.Module):
    """Gaussian NLL loss for DeepAR: ``preds`` is ``(B,H,D,2)`` = (mu, sigma)."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, preds, batch):
        target = batch[1]
        mu = preds[..., 0]
        sigma = preds[..., 1] if preds.shape[-1] > 1 else preds[..., 0]
        dist = torch.distributions.Normal(mu, sigma)
        return {"loss": -dist.log_prob(target).mean()}


class TSMetric(object):
    """MSE/MAE metric.

    ``pred_mode``:
      * ``point``    : ``preds`` is ``(B,H,D)`` (default)
      * ``quantile`` : ``preds`` is ``(B,H,D,Q)``, use the median quantile
      * ``params``   : ``preds`` is ``(B,H,D,2)``, use the mean (mu)
    """

    def __init__(self, main_indicator="MSE", pred_mode="point",
                 quantile_index=None, **kwargs):
        self.main_indicator = main_indicator
        self.pred_mode = pred_mode
        self.quantile_index = quantile_index
        self.reset()

    def reset(self):
        self.se = 0.0
        self.ae = 0.0
        self.n = 0

    def _reduce(self, preds):
        if self.pred_mode == "quantile":
            idx = self.quantile_index
            if idx is None:
                idx = preds.shape[-1] // 2
            return preds[..., idx]
        if self.pred_mode == "params":
            return preds[..., 0]
        return preds

    def __call__(self, preds, batch):
        target = batch[1]
        if not torch.is_tensor(preds):
            preds = torch.as_tensor(preds)
        preds = self._reduce(preds)
        d = (preds.detach().cpu().double() - target.detach().cpu().double())
        self.se += float((d ** 2).sum())
        self.ae += float(d.abs().sum())
        self.n += int(d.numel())

    def get_metric(self):
        n = max(self.n, 1)
        return {"MSE": self.se / n, "MAE": self.ae / n}


def build_ts_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    name = str(cfg.pop("name", "") or cfg.pop("type", "") or "mse").lower()
    if name in ("quantile", "pinball"):
        return TSQuantileLoss(**cfg)
    if name in ("nll", "gaussian_nll", "deepar"):
        return TSNLLLoss(**cfg)
    return TSLoss(loss=name, **cfg)


def build_ts_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    cfg.pop("type", None)
    return TSMetric(**cfg)
