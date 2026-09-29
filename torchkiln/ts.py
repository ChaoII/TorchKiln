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
    """MSE/MAE + **功率预测常用指标**。

    ``pred_mode``:
      * ``point``    : ``preds`` is ``(B,H,D)`` (default)
      * ``quantile`` : ``preds`` is ``(B,H,D,Q)``, use the median quantile
      * ``params``   : ``preds`` is ``(B,H,D,2)``, use the mean (mu)

    额外指标（`extra_metrics`，默认全开）：
      * ``RMSE`` / ``nRMSE``  —— **归一化 RMSE**（除以 target 全体的 RMS，=``RMSE/rms``；
        新能源功率预测最常见的口径；也可用 ``norm_value`` 指定装机容量做归一化）
      * ``MAPE`` / ``sMAPE``  —— 注意二者在 target≈0 时会爆，仅作参考
      * ``R2``                —— 决定系数
      * ``PICP`` / ``MPIW``   —— **区间覆盖率 / 平均区间宽度**（仅 ``pred_mode='quantile'``）；
        再配 ``nominal_coverage``（默认 0.9）判区间是否可信；
        ``pinball`` 为分位数损失（与 TFT 训练口径一致）
    """

    def __init__(self, main_indicator="MSE", pred_mode="point",
                 quantile_index=None, quantiles=None, nominal_coverage=0.9,
                 norm_value=None, extra_metrics=True, **kwargs):
        self.main_indicator = main_indicator
        self.pred_mode = pred_mode
        self.quantile_index = quantile_index
        self.quantiles = list(quantiles) if quantiles else None
        self.nominal_coverage = float(nominal_coverage)
        self.norm_value = norm_value
        self.extra = bool(extra_metrics)
        self.reset()

    def reset(self):
        self.se = 0.0
        self.ae = 0.0
        self.n = 0
        self.sum_t = 0.0
        self.sum_t2 = 0.0
        self.ape = 0.0            # MAPE 分子（|d|/|t|，|t|>eps 才计）
        self.ape_n = 0
        self.sape = 0.0           # sMAPE 分子
        self.sape_n = 0
        self.cov_hit = 0          # PICP 命中数
        self.width = 0.0          # MPIW 累计
        self.cov_n = 0
        self.pinball = 0.0
        self.pinball_n = 0

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
        p_all = preds.detach().cpu().double()
        t = target.detach().cpu().double()
        preds = self._reduce(p_all)
        d = preds - t
        self.se += float((d ** 2).sum())
        self.ae += float(d.abs().sum())
        self.n += int(d.numel())
        self.sum_t += float(t.sum())
        self.sum_t2 += float((t ** 2).sum())

        if self.extra:
            # MAPE / sMAPE（|t|>eps 才计入）
            mask = t.abs() > 1e-8
            if bool(mask.any()):
                self.ape += float((d[mask].abs() / t[mask].abs()).sum())
                self.ape_n += int(mask.sum())
            denom = (preds.abs() + t.abs()).clamp_min(1e-8)
            self.sape += float((2 * d.abs() / denom).sum())
            self.sape_n += int(d.numel())

        # 分位数：PICP / MPIW / pinball
        if self.pred_mode == "quantile" and p_all.shape[-1] >= 2:
            Q = p_all.shape[-1]
            qs = self.quantiles or [j / (Q - 1) for j in range(Q)]
            if len(qs) != Q:                 # 配置与模型输出不符时按等距分位数兜底
                qs = [j / (Q - 1) for j in range(Q)]
            target_lo, target_hi = ((1 - self.nominal_coverage) / 2,
                                    1 - (1 - self.nominal_coverage) / 2)
            lo_i = min(range(Q), key=lambda j: abs(qs[j] - target_lo))
            hi_i = min(range(Q), key=lambda j: abs(qs[j] - target_hi))
            lo, hi = p_all[..., lo_i], p_all[..., hi_i]
            self.cov_hit += int(((t >= lo) & (t <= hi)).sum())
            self.width += float((hi - lo).sum())
            self.cov_n += int(t.numel())
            for j, q in enumerate(qs):
                if q > 0:
                    e = t - p_all[..., j]
                    self.pinball += float(torch.maximum(q * e, (q - 1) * e).sum())
                    self.pinball_n += int(t.numel())

    def get_metric(self):
        n = max(self.n, 1)
        rmse = (self.se / n) ** 0.5
        out = {"MSE": self.se / n, "MAE": self.ae / n}
        if not self.extra:
            return out
        out["RMSE"] = rmse
        # nRMSE：优先用 norm_value（装机容量）；否则用 target 的 RMS
        if self.norm_value:
            nrmse = rmse / float(self.norm_value)
        else:
            rms = (self.sum_t2 / n) ** 0.5
            nrmse = rmse / (rms if rms > 1e-12 else 1.0)
        out["nRMSE"] = nrmse
        out["MAPE"] = (self.ape / max(self.ape_n, 1)) if self.ape_n else float("nan")
        out["sMAPE"] = (self.sape / max(self.sape_n, 1)) if self.sape_n else float("nan")
        # R² = 1 - SSE/SST
        mean_t = self.sum_t / n
        sst = self.sum_t2 - n * mean_t ** 2
        out["R2"] = 1.0 - (self.se / sst) if sst > 1e-12 else float("nan")
        if self.cov_n:
            out["PICP"] = self.cov_hit / self.cov_n
            out["MPIW"] = self.width / self.cov_n
            out["pinball"] = self.pinball / max(self.pinball_n, 1)
        out[self.main_indicator] = out.get(self.main_indicator, out["MSE"])
        return out


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
