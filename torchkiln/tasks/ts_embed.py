"""时序表示学习任务（``Architecture.task: ts_embed``）。

算法（``Architecture.algorithm``）：``ts2vec`` / ``cost``

定位：**自监督预训练**（无需标签）——训练完用 ``encode()`` 抽特征，
再送下游分类/聚类/预测（对应 paddlets 的 ``repr_classifier``/``repr_cluster``/
``repr_forecasting``，本框架通过「导出特征 → 用 ts_classify / ts_forecast」实现同样的事）。

训练：
  * TS2Vec：两个**独立随机 mask** 视图 → 分层对比损失（instance + temporal）
  * CoST  ：时间域对比（anchor/pos/neg）+ 频域对比（两个视图）

指标：训练用对比损失；``Eval`` 额外报 ``repr_std``（表征各维标准差均值，
用于发现"表征塌缩"——若趋 0 说明没学到东西）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from torchkiln.data.ts_classify import TSClassifyDataset, ts_classify_collate

__all__ = ["TsEmbedTask", "TSReprMetric"]


class TSReprMetric:
    def __init__(self, main_indicator="loss", **kw):
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.n = 0
        self.stds = []

    def __call__(self, loss, feat=None):
        if loss is None:
            return
        if torch.is_tensor(loss):
            loss = float(loss.detach())
        self.loss_sum += float(loss)
        self.n += 1
        if feat is not None:
            f = feat.detach().float()
            if f.ndim == 3:
                f = f.reshape(-1, f.shape[-1])
            self.stds.append(float(f.std(dim=0).mean()))

    def get_metric(self):
        out = {"num": int(self.n)}
        if self.n == 0:
            return out
        out["loss"] = self.loss_sum / self.n
        if self.stds:
            out["repr_std"] = float(np.mean(self.stds))
        out[self.main_indicator] = out.get(self.main_indicator, out["loss"])
        return out


class TsEmbedTask(TaskAdapter):
    def __init__(self, config=None):
        arch = (config or {}).get("Architecture", {}) if config else {}
        self.name = arch.get("task", "ts_embed")
        self.kind = str(arch.get("algorithm", "ts2vec")).lower()

    def _kind(self, config):
        return str((config.get("Architecture") or {}).get("algorithm", "ts2vec")).lower()

    def build_post_process(self, config):
        return None

    def build_model(self, config, post_process):
        self.kind = self._kind(config)
        arch = config["Architecture"]
        L = int(arch.get("in_chunk_len", 96))
        C = int(arch.get("num_features", 1))
        H = int(arch.get("hidden_dim", 32))
        D = int(arch.get("output_dim", 64))
        depth = int(arch.get("depth", arch.get("num_layers", 4)))
        if self.kind == "ts2vec":
            from torchkiln.nn.ts_ts2vec import TS2VecModule
            return TS2VecModule(C, D, H, depth)
        if self.kind == "cost":
            from torchkiln.nn.ts_cost import CoSTEncoder
            return CoSTEncoder(C, D, H, depth, L)
        raise ValueError("unknown ts_embed algorithm: %s" % self.kind)

    def build_loss(self, config, model):
        self.kind = self._kind(config)
        lc = config.get("Loss") or {}
        self.alpha = float(lc.get("alpha", 0.5))
        self.temporal_unit = int(lc.get("temporal_unit", 0))
        self.temperature = float(lc.get("temperature", 0.1))
        return _ReprLoss(self.kind, alpha=self.alpha,
                         temporal_unit=self.temporal_unit,
                         temperature=self.temperature)

    def build_metric(self, config):
        mc = config.get("Metric", {}) or {}
        return TSReprMetric(main_indicator=mc.get("main_indicator", "loss"))

    def build_datasets(self, config, logger):
        tr = TSClassifyDataset(config, "Train", logger)
        ev = TSClassifyDataset(config, "Eval", logger) \
            if config.get("Eval", {}).get("dataset") else None
        return tr, ev

    def train_collate(self, batch):
        return ts_classify_collate(batch)

    def eval_collate(self, batch):
        return ts_classify_collate(batch)

    def forward_train(self, model, images, batch):
        """生成两视图并算对比损失；返回 (loss, feat) 供 trainer 使用。

        ⚠️ 这里直接**返回 dict**（loss 已算好），`_ReprLoss.forward` 会原样透传。
        """
        # ⚠️ kind/alpha 等由 __init__ 与 build_loss 设置，但 trainer 可能以无 config
        #    方式构造任务 -> 这里做兜底默认值
        alpha = getattr(self, "alpha", 0.5)
        temporal_unit = getattr(self, "temporal_unit", 0)
        temperature = getattr(self, "temperature", 0.1)
        x = images
        if self.kind == "ts2vec":
            from torchkiln.nn.ts_ts2vec import hierarchical_contrastive_loss
            # 两个独立随机 mask 视图
            r1 = model(x, "binomial")
            r2 = model(x, "binomial")
            loss = hierarchical_contrastive_loss(r1, r2, alpha, temporal_unit)
            return {"_repr": (loss, r1)}
        from torchkiln.nn.ts_cost import (frequency_contrastive_loss,
                                          time_contrastive_loss)
        trend1, season1 = model(x, "binomial")
        trend2, season2 = model(x, "binomial")
        # 时间域：anchor/pos = 两视图的趋势表征，neg = 季节表征（paddlets CoST 做法）
        anchor = trend1.mean(dim=1)                       # (B, D)
        pos = trend2.mean(dim=1)                          # (B, D)
        # ⚠️ neg 需 (D, K)：torch 的 matmul(anchor(B,D), neg(D,K)) -> (B,K)
        neg = season2.mean(dim=1).t()                     # (D, D)  取 season 的通道均值
        lt = time_contrastive_loss(anchor, pos, neg, temperature)
        lf = frequency_contrastive_loss(trend1, trend2)
        return {"_repr": (lt + lf, trend1)}

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device)
        with torch.no_grad():
            d = self.forward_train(model, images, batch)
            loss, feat = d["_repr"]
        metric(loss, feat)

    def summary_lines(self, config, global_config, post_process):
        a = config.get("Architecture", {})
        return ["时序表示学习: algorithm=%s in_chunk_len=%s output_dim=%s"
                % (a.get("algorithm"), a.get("in_chunk_len"), a.get("output_dim")),
                "自监督（无标签）；表征可直接用于 ts_classify / 聚类 / 预测"]


class _ReprLoss(nn.Module):
    """透传 `forward_train` 已经算好的对比损失（dict 里的 ``_repr``）。"""

    def __init__(self, kind, alpha=0.5, temporal_unit=0, temperature=0.1):
        super().__init__()
        self.kind = kind
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.temperature = temperature

    def forward(self, preds, batch):
        loss, _ = preds["_repr"]
        return {"loss": loss, "repr_loss": loss.detach()}
