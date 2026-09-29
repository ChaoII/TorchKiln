"""时序分类任务适配器（``Architecture.task: ts_classify``）。

算法（``Architecture.algorithm``）：``cnn`` / ``inception_time``

指标：``TSClassifyMetric`` —— accuracy / macro-F1 / 各类 P/R/F1（故障诊断关注**每类**召回）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from torchkiln.data.ts_classify import TSClassifyDataset, ts_classify_collate

__all__ = ["TsClassifyTask", "TSClassifyMetric"]


class TSClassifyMetric:
    def __init__(self, main_indicator="accuracy", **kw):
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.n = 0
        self.correct = 0
        self.loss_sum = 0.0
        self.y, self.p = [], []

    def __call__(self, logits, labels):
        if logits is None:
            return
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        pred = np.argmax(logits, axis=-1).reshape(-1)
        y = np.asarray(labels).reshape(-1)
        self.n += len(y)
        self.correct += int((pred == y).sum())
        self.y.append(y)
        self.p.append(pred)

    def get_metric(self):
        out = {"num": int(self.n)}
        if self.n == 0:
            return out
        y = np.concatenate(self.y)
        p = np.concatenate(self.p)
        out["accuracy"] = float((y == p).mean())
        classes = np.unique(np.concatenate([y, p]))
        f1s, precs, recs = [], [], []
        for c in classes:
            tp = float(((p == c) & (y == c)).sum())
            fp = float(((p == c) & (y != c)).sum())
            fn = float(((p != c) & (y == c)).sum())
            pr = tp / (tp + fp) if tp + fp > 0 else 0.0
            rc = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
            precs.append(pr)
            recs.append(rc)
            f1s.append(f1)
            out["f1_class%d" % c] = f1
        out["precision_macro"] = float(np.mean(precs))
        out["recall_macro"] = float(np.mean(recs))
        out["f1_macro"] = float(np.mean(f1s))
        out[self.main_indicator] = out.get(self.main_indicator, out["accuracy"])
        return out


class TsClassifyTask(TaskAdapter):
    def __init__(self, config=None):
        arch = (config or {}).get("Architecture", {}) if config else {}
        self.name = arch.get("task", "ts_classify")
        self.kind = str(arch.get("algorithm", "cnn")).lower()

    def _kind(self, config):
        """⚠️ BaseTrainer 用**无 config** 的 ``get_task(name)`` 构造任务，
        故 ``__init__`` 里的 kind 是默认值；这里每次都从 config 重新取。"""
        a = str((config.get("Architecture") or {}).get("algorithm", "cnn")).lower()
        return a

    def build_post_process(self, config):
        return None

    def build_model(self, config, post_process):
        self.kind = self._kind(config)
        arch = config["Architecture"]
        L = int(arch.get("in_chunk_len", 100))
        C = int(arch.get("num_features", 1))
        NC = int(arch.get("num_classes", 2))
        if self.kind == "cnn":
            from torchkiln.nn.ts_classify import CNNBlock, ACTIVATIONS
            # ⚠️ paddlets `_CNNBlock` 默认逐层 Sigmoid（会梯度消失、实测 accuracy 仅 ~0.38）；
            #    这里 activation 可配（demo 用 ReLU 才学得动），**不传时保持 paddlets 默认 Sigmoid**。
            act = ACTIVATIONS.get(str(arch.get("activation", "Sigmoid")), torch.nn.Sigmoid)
            last = (torch.nn.Identity
                    if str(arch.get("last_activation", "Softmax")).lower()
                    in ("identity", "none", "logits") else torch.nn.Softmax)
            return CNNBlock(C, L, NC,
                            arch.get("hidden_config", [6, 12]),
                            act, last,
                            arch.get("kernel_size", 7),
                            arch.get("avg_pool_size", 3),
                            arch.get("use_bn", False), arch.get("use_drop", False),
                            arch.get("dropout_rate", 0.5))
        if self.kind in ("inception_time", "inceptiontime", "it"):
            from torchkiln.nn.ts_classify import InceptionTime, ACTIVATIONS
            act = ACTIVATIONS.get(str(arch.get("activation", "ReLU")), torch.nn.ReLU)
            return InceptionTime(C, NC, arch.get("kernel_size", 41),
                                 arch.get("block_out_size", 128),
                                 arch.get("block_depth", 6), act,
                                 arch.get("use_residual", True),
                                 arch.get("use_bottleneck", True))
        raise ValueError("unknown ts_classify algorithm: %s" % self.kind)

    def build_loss(self, config, model):
        self.kind = self._kind(config)
        return _TsCE()

    def build_metric(self, config):
        mc = config.get("Metric", {}) or {}
        return TSClassifyMetric(main_indicator=mc.get("main_indicator", "accuracy"))

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
        return model({"features": images})

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.no_grad():
            logits = model({"features": images})
        metric(logits, labels)

    def summary_lines(self, config, global_config, post_process):
        a = config.get("Architecture", {})
        return ["时序分类: algorithm=%s in_chunk_len=%s num_classes=%s"
                % (a.get("algorithm"), a.get("in_chunk_len"), a.get("num_classes")),
                "指标: accuracy / macro-F1 / 每类 F1"]


class _TsCE(nn.CrossEntropyLoss):
    """trainer 传的 ``labels`` 是 ``[feat, label]`` list；取最后一项做 target、返回 dict。

    ⚠️ paddlets 的 CNN 输出层是 ``Softmax``，**再喂 softmax 前的 logits 会给 CE**；
    对已过 softmax 的输出直接用 CE 会因"双重 softmax"导致梯度极小、学不动
    （实测 accuracy 卡在 1/3 随机水平）
    ⇒ 这里对**概率输出**取 ``log`` 后计算 NLL（等价于用其 logits，数值稳定版）。
    """

    def forward(self, logits, labels):
        tgt = labels[-1] if isinstance(labels, (list, tuple)) else labels
        if not torch.is_tensor(tgt):
            tgt = torch.as_tensor(tgt)
        tgt = tgt.long().reshape(-1)
        x = logits
        # 若输出已是概率（每行和≈1 且非负）-> 取 log 走 NLL，等价于其 logits
        with torch.no_grad():
            prob_like = bool(torch.all(x >= 0)) and bool(
                torch.allclose(x.sum(dim=-1), torch.ones_like(x.sum(dim=-1)), atol=1e-3))
        if prob_like:
            loss = nn.functional.nll_loss(torch.log(x.clamp_min(1e-12)), tgt)
        else:
            loss = nn.functional.cross_entropy(x, tgt)
        return {"loss": loss, "cls_loss": loss.detach()}
