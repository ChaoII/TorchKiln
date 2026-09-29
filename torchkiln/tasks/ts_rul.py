"""RUL（剩余寿命预测）组件：损失 + 指标 + 任务适配器。

**指标**（工业界 RUL 标配）：
  * ``RMSE`` / ``MAE`` —— 常规回归误差
  * **``NASA_Score``（PHM08 Score）** —— **非对称评分**：
    ``∑ (exp(err/13) - 1)`` 若 ``err = pred - true < 0``（**预测"晚了"= 危险**，罚重）
    ``∑ (exp(-err/10) - 1)`` 若 ``err >= 0``（预测"早了"= 保守，罚轻）
    这是 NASA C-MAPSS 挑战赛的官方口径，**直接关系维护决策风险**：
    把"还能跑 10 次"的设备预测成"还能跑 50 次"远比反过来危险。
  * ``Score_avg`` = NASA_Score / N（便于跨规模比较）
  * ``acc_at``（``|err| <= tol`` 的比例，反映工程可用性）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from torchkiln.data.ts_rul import RULDataset, rul_collate

__all__ = ["RULLoss", "RULMetric", "TsRulTask", "nasa_score"]


def nasa_score(err: np.ndarray) -> float:
    """PHM08 非对称评分（``err = pred - true``；提前预测罚轻、滞后预测罚重）。"""
    err = np.asarray(err, dtype=np.float64)
    d = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(d.sum())


class RULLoss(nn.Module):
    """RUL 回归损失（``mse`` / ``mae`` / ``smooth_l1``）。

    ⚠️ 这些 ts 模型是**多变量 → 多变量**结构（``target_dim`` 同时是输入/输出维），
    所以 RUL 训练时模型输出 ``(B, 1, C)``（C = 特征数）。三种口径：
      * ``reduce='mean'``（默认）：**所有通道都回归 RUL 再取均值** ——
        等价"每个传感器通道各预测一次 RUL"的**集成**，比单通道更稳；
      * ``reduce='first'``：只用第 0 通道；
      * ``reduce='min'``：取所有通道的最小值（**最保守**，与"宁可早修不可晚修"一致）。

    ``asymmetric_weight > 0``：对**高估剩余寿命**（pred > true，危险方向）加大惩罚，
    使训练目标与 NASA Score 的风险取向一致。
    """

    def __init__(self, loss="mse", asymmetric_weight=0.0, reduce="mean", **kw):
        super().__init__()
        self.loss = str(loss).lower()
        self.aw = float(asymmetric_weight)
        self.reduce = str(reduce).lower()

    def _reduce(self, preds):
        # 模型输出 (B, 1, C) 或 (B, 1)；target 是 (B,)
        p = preds[..., 0] if preds.ndim >= 3 and self.reduce == "first" else preds
        if self.reduce == "first" and p.ndim >= 2:
            p = p.reshape(p.shape[0], -1)[:, 0]
        elif p.ndim >= 3:
            p = p.reshape(p.shape[0], -1)                 # (B, C)
        if self.reduce == "min" and p.ndim == 2:
            p = p.min(dim=-1).values
        elif self.reduce == "mean" and p.ndim == 2:
            p = p.mean(dim=-1)
        return p.reshape(-1)

    def forward(self, preds, batch):
        pred = self._reduce(preds)
        target = batch[1].float().reshape(-1)
        if self.loss == "mae":
            return {"loss": nn.functional.l1_loss(pred, target)}
        if self.loss == "smooth_l1":
            return {"loss": nn.functional.smooth_l1_loss(pred, target)}
        if self.aw > 0:
            err = pred - target                      # >0 = 高估寿命（危险）
            w = torch.where(err > 0, 1.0 + self.aw, torch.ones_like(err))
            return {"loss": (w * err ** 2).mean()}
        return {"loss": nn.functional.mse_loss(pred, target)}


class RULMetric:
    """RMSE / MAE / **NASA Score** / Score_avg / 容差内比例。"""

    def __init__(self, main_indicator="RMSE", score_tol=10.0, reduce="mean", **kw):
        self.main_indicator = str(main_indicator)
        self.tol = float(score_tol)
        self.reduce = str(reduce).lower()
        self.reset()

    def reset(self):
        self.pred, self.true = [], []

    def __call__(self, preds, batch):
        if preds is None:
            return
        p = preds.detach().cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
        t = batch[1].detach().cpu().numpy() if torch.is_tensor(batch[1]) else np.asarray(batch[1])
        p = np.asarray(p)
        if p.ndim >= 3:                      # (B, 1, C)：与训练同口径
            p = p.reshape(p.shape[0], -1)
            p = p.min(axis=-1) if self.reduce == "min" else (
                p[:, 0] if self.reduce == "first" else p.mean(axis=-1))
        self.pred.append(p.reshape(-1))
        self.true.append(np.asarray(t).reshape(-1))

    def get_metric(self):
        if not self.pred:
            return {}
        p = np.concatenate(self.pred).astype(np.float64)
        t = np.concatenate(self.true).astype(np.float64)
        err = p - t
        rmse = float(np.sqrt((err ** 2).mean()))
        out = {
            "num": int(len(p)),
            "RMSE": rmse,
            "MAE": float(np.abs(err).mean()),
            "NASA_Score": nasa_score(err),
            "Score_avg": nasa_score(err) / max(len(p), 1),
            "acc_at_%d" % int(self.tol): float((np.abs(err) <= self.tol).mean()),
        }
        return out


class TsRulTask(TaskAdapter):
    """RUL 任务（``Architecture.task: ts_rul``）—— 复用 ts_forecast 的模型族。

    ``Architecture.Head`` 与 `ts_forecast` 同构（``model: tcn/lstm/transformer/...``），
    但 ``out_chunk_len`` 固定为 **1**（只预测一个 RUL 标量）。
    """

    name = "ts_rul"

    def build_post_process(self, config):
        return None

    def build_model(self, config, post_process):
        from torchkiln.models.ts import build_ts_model

        arch = dict(config["Architecture"])
        head = dict(arch.get("Head") or {})
        ds = ((config.get("Train") or {}).get("dataset")
              or (config.get("Eval") or {}).get("dataset") or {})
        # ⚠️ 这些 ts 模型是「多变量 → 多变量」映射：`target_dim` 实为**输入通道数**
        #    （= 特征数，C-MAPSS 为 21 传感器 + 3 操作 = 24）。
        #    RUL 只需**一个标量** ⇒ 模型输出 (B,1,24)，训练/评估时**取第 0 通道**
        #    （见 RULLoss / RULMetric 的 `preds[..., 0]` 处理）。
        head["out_chunk_len"] = 1
        # ⚠️ 维度必须在 build_model 时确定，但 `build_datasets` 在它**之后**才调用
        #    ⇒ 这里**直接构建一次 Train 数据集**来探测实际特征维
        #    （C-MAPSS 自动筛选会 24 -> 17；仅读 CSV + 统计，开销很小）
        n_feat = ds.get("num_features")
        if n_feat is None and getattr(self, "_n_feat", None) is None:
            try:
                from torchkiln.data.ts_rul import RULDataset
                self._n_feat = RULDataset(config, "Train").dim
            except Exception:
                self._n_feat = None
        n_feat = n_feat or getattr(self, "_n_feat", None)
        if n_feat is None:
            n_feat = len(ds.get("sensor_cols") or []) or 24
        head["target_dim"] = int(n_feat)
        head.setdefault("in_chunk_len", int(ds.get("window", 30)))
        arch["Head"] = head
        arch["_dataset"] = ds
        return build_ts_model(arch)

    def build_loss(self, config, model):
        lc = dict(config.get("Loss") or {})
        return RULLoss(loss=lc.get("type", lc.get("name", "mse")),
                       asymmetric_weight=lc.get("asymmetric_weight", 0.0),
                       reduce=lc.get("reduce", "mean"))

    def build_metric(self, config):
        mc = config.get("Metric", {}) or {}
        return RULMetric(main_indicator=mc.get("main_indicator", "RMSE"),
                         score_tol=mc.get("score_tol", 10.0),
                         reduce=mc.get("reduce", "mean"))

    def build_datasets(self, config, logger):
        tr = RULDataset(config, "Train", logger)
        ev = RULDataset(config, "Eval", logger) if config.get("Eval", {}).get("dataset") else None
        # ⚠️ 自动筛选后特征维会变（C-MAPSS 24 -> 17），把**实际维度**回写给 config，
        #    供 build_model 使用（build_model 在 build_datasets 之后被调用）
        self._n_feat = tr.dim
        return tr, ev

    def train_collate(self, batch):
        return rul_collate(batch)

    def eval_collate(self, batch):
        return rul_collate(batch)

    def forward_train(self, model, images, batch):
        out = model({"past_target": images})
        return out[0] if isinstance(out, tuple) else out

    def eval_step(self, model, batch, post_process, metric, device):
        past = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            out = model({"past_target": past})
            out = out[0] if isinstance(out, tuple) else out
        metric(out, [batch[0], batch[1].to(device)])

    def summary_lines(self, config, global_config, post_process):
        a = config.get("Architecture", {}) or {}
        h = a.get("Head") or {}
        ds = (config.get("Train") or {}).get("dataset") or {}
        return ["RUL: model=%s window=%s max_rul=%s"
                % (h.get("model"), ds.get("window"), ds.get("max_rul")),
                "指标: RMSE / MAE / NASA_Score(PHM08 非对称) / Score_avg / 容差内比例"]
