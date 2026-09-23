"""Shared classification head pieces (loss / metric / post-process / builders).

单一实现来源:分类相关的 ``ClsLoss`` / ``ClsPostProcess`` / ``ClsMetric`` 以及它们的
build 函数都放这里,避免在每个任务文件里复制一份。
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

__all__ = [
    "num_classes_of",
    "ClsLoss",
    "ClsPostProcess",
    "ClsMetric",
    "build_cls_loss",
    "build_cls_metric",
]


def num_classes_of(model):
    """Class count of a head (`num_classes` on hand-written heads, `nc` on graph heads)."""
    from torchkiln.nn.graph import task_head

    head = task_head(model)
    return int(getattr(head, "num_classes", None) or getattr(head, "nc", 0))


class ClsLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, preds, labels):
        target = labels[1].long()
        return {"loss": self.criterion(preds, target)}


class ClsPostProcess(object):
    def __init__(self, topk=5, **kwargs):
        self.topk = int(topk)

    def __call__(self, preds):
        probs = torch.softmax(preds.detach(), dim=1)
        k = min(self.topk, probs.shape[1])
        vals, idxs = probs.topk(k, dim=1)
        vals = vals.cpu().numpy()
        idxs = idxs.cpu().numpy()
        out = []
        for i in range(idxs.shape[0]):
            out.append([(int(idxs[i, j]), float(vals[i, j])) for j in range(k)])
        return out


class ClsMetric(object):
    def __init__(self, topk=(1, 5), main_indicator="acc", **kwargs):
        self.topk = tuple(int(k) for k in topk)
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.correct = {k: 0 for k in self.topk}
        self.total = 0

    def __call__(self, post_result, batch):
        labels = batch[1]
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        for i, preds in enumerate(post_result):
            gt = int(labels[i])
            self.total += 1
            for k in self.topk:
                if any(p[0] == gt for p in preds[:k]):
                    self.correct[k] += 1

    def get_metric(self):
        denom = max(1, self.total)
        metrics = {"top{}".format(k): self.correct[k] / denom for k in self.topk}
        metrics["acc"] = metrics.get("top1", 0.0)
        return metrics


def build_cls_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "CrossEntropy")
    if name not in ("CrossEntropy", "ClsLoss"):
        raise ValueError("Unknown cls loss: {}".format(name))
    return ClsLoss(**cfg)


def build_cls_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    return ClsMetric(**cfg)
