"""时序异常检测任务适配器（``Architecture.task: ts_anomaly``）。

支持 5 种算法（``Architecture.algorithm``）：
    ``ae`` / ``vae`` / ``usad`` / ``mtad_gat`` / ``anomaly_transformer``

与框架的衔接：
  * ``forward_train`` 把 batch 包成 ``{"observed_cov_numeric": (B,L,C)}`` 喂给网络；
    **两阶段模型（USAD / AT）返回 ``None``** —— 它们的真实更新由 ``TSAnomalyLoss.train_step``
    钩子在 trainer 里完成（需要两次 backward/step）。
  * ``eval_step`` 计算每窗口异常分数并交给 ``TSAnomalyMetric``。
"""
from __future__ import annotations

import torch

from ptcore.task import TaskAdapter
from torchkiln.data.ts_anomaly import TSAnomalyDataset, anomaly_collate
from torchkiln.ts_anomaly import (TSAnomalyLoss, TSAnomalyMetric, anomaly_score)

__all__ = ["TsAnomalyTask"]

KINDS = ("ae", "vae", "usad", "mtad_gat", "anomaly_transformer")


class TsAnomalyTask(TaskAdapter):
    def __init__(self, config=None):
        arch = (config or {}).get("Architecture", {}) if config else {}
        self.name = arch.get("task", "ts_anomaly")
        self.kind = str(arch.get("algorithm", "ae")).lower()
        if self.kind in ("at", "anomalytransformer"):
            self.kind = "anomaly_transformer"

    # ---------------- build ----------------
    def build_post_process(self, config):
        return None

    def build_model(self, config, post_process):
        arch = config["Architecture"]
        L = int(arch.get("in_chunk_len", 100))
        C = int(arch.get("num_features", 1))
        fit = {"observed_cat_cols": {}, "observed_num_dim": C, "observed_cat_dim": 0}
        k = self.kind
        if k == "ae":
            from torchkiln.nn.s_anomaly import AEBlock
            net = AEBlock(L, arch.get("ed_type", "MLP"), fit,
                          arch.get("hidden_config", [32, 16]),
                          torch.nn.ReLU, torch.nn.Identity,
                          arch.get("kernel_size", 3), arch.get("dropout_rate", 0.2),
                          arch.get("use_bn", False), arch.get("embedding_size", 16),
                          arch.get("pooling", False))
        elif k == "vae":
            from torchkiln.nn.s_anomaly import VAEBlock
            net = VAEBlock(L, arch.get("hidden_config", [32, 16]), C,
                           arch.get("base_en", "MLP"), arch.get("base_de", "MLP"),
                           arch.get("use_bn", True), True,
                           arch.get("dropout_rate", 0.2),
                           arch.get("kernel_size", 1), 1, "forward",
                           None, None, arch.get("stdev", 0.1))
        elif k == "usad":
            from torchkiln.nn.s_anomaly import USADBlock
            net = USADBlock(L, arch.get("ed_type", "MLP"), fit,
                            arch.get("hidden_config", [32, 16]),
                            torch.nn.ReLU, torch.nn.Identity,
                            arch.get("kernel_size", 3), arch.get("dropout_rate", 0.2),
                            arch.get("use_bn", False), arch.get("embedding_size", 16),
                            arch.get("pooling", False), arch.get("flatten", True))
        elif k == "mtad_gat":
            from torchkiln.nn.s_anomaly_gat import MTADGATBlock
            net = MTADGATBlock(
                L, fit, arch.get("target_dims"), arch.get("kernel_size", 7),
                arch.get("feat_gat_embed_dim"), arch.get("time_gat_embed_dim"),
                arch.get("use_gatv2", True), arch.get("use_bias", True),
                arch.get("gru_n_layers", 1), arch.get("gru_hid_size", 32),
                arch.get("forecast_n_layers", 1), arch.get("forecast_hid_size", 32),
                arch.get("recon_n_layers", 1), arch.get("recon_hid_size", 32),
                arch.get("dropout", 0.0), arch.get("alpha", 0.2))
        elif k == "anomaly_transformer":
            from torchkiln.nn.s_anomaly_transformer import AnomalyTransformerNet
            net = AnomalyTransformerNet(
                L, C, C, arch.get("d_model", 32), arch.get("n_heads", 4),
                arch.get("e_layers", 2), arch.get("d_ff", 32),
                arch.get("dropout", 0.0), torch.nn.functional.gelu, True)
        else:
            raise ValueError("unknown ts_anomaly algorithm: %s" % k)
        return net

    def build_loss(self, config, model):
        arch = config["Architecture"]
        loss_cfg = config.get("Loss", {}) or {}
        post = config.get("PostProcess", {}) or {}
        self.alpha = float(loss_cfg.get("alpha", 0.5))
        self.beta = float(loss_cfg.get("beta", 0.5))
        self.flatten = bool(arch.get("flatten", True)) and arch.get("ed_type", "MLP") == "MLP"
        self.at_temperature = float(post.get("at_temperature", 50.0))
        self.at_score_mode = str(post.get("at_score_mode", "paddlets"))
        self.target_dims = arch.get("target_dims")
        return TSAnomalyLoss(kind=self.kind,
                             in_chunk_len=int(arch.get("in_chunk_len", 100)),
                             k=int(loss_cfg.get("k", 3)),
                             kld_beta=float(loss_cfg.get("kld_beta", 0.2)),
                             alpha=self.alpha, beta=self.beta,
                             target_dims=self.target_dims)

    def build_metric(self, config):
        mc = config.get("Metric", {}) or {}
        return TSAnomalyMetric(main_indicator=mc.get("main_indicator", "f1"),
                               threshold_percentile=float(mc.get("threshold_percentile", 99.0)),
                               point_adjust=bool(mc.get("point_adjust", True)))

    def build_datasets(self, config, logger):
        tr = TSAnomalyDataset(config, "Train", logger)
        ev = TSAnomalyDataset(config, "Eval", logger) \
            if config.get("Eval", {}).get("dataset") else None
        return tr, ev

    def train_collate(self, batch):
        return anomaly_collate(batch)

    def eval_collate(self, batch):
        return anomaly_collate(batch)

    # ---------------- forward ----------------
    def forward_train(self, model, images, batch):
        if self.kind in TSAnomalyLoss.TWO_PHASE:
            return None          # 真实更新在 loss.train_step 里（需要两次 backward/step）
        return model({"observed_cov_numeric": images})

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        X = {"observed_cov_numeric": images}
        with torch.no_grad():
            if self.kind in ("usad", "anomaly_transformer"):
                # 两阶段模型推理时也要真的跑一次网络
                preds = model(X)
                if self.kind == "anomaly_transformer":
                    preds = list(preds)
            else:
                preds = model(X)
        scores = anomaly_score(
            self.kind, preds, [images, labels], int(batch[0].shape[1]),
            alpha=getattr(self, "alpha", 0.5),
            beta=getattr(self, "beta", 0.5),
            flatten=getattr(self, "flatten", False),
            at_temperature=getattr(self, "at_temperature", 50.0),
            at_score_mode=getattr(self, "at_score_mode", "paddlets"),
            target_dims=getattr(self, "target_dims", None))
        metric(scores, labels.detach().cpu().numpy())

    def summary_lines(self, config, global_config, post_process):
        a = config.get("Architecture", {})
        return ["时序异常检测: algorithm=%s in_chunk_len=%s"
                % (a.get("algorithm"), a.get("in_chunk_len")),
                "指标: 点级 P/R/F1(point-adjust) + AUC-ROC/AUC-PR + best_f1(oracle)"]
