"""Shared classification head pieces (loss / metric / post-process / builders).

单一实现来源:分类相关的 ``ClsLoss`` / ``ClsPostProcess`` / ``ClsMetric`` 以及它们的
build 函数都放这里,避免在每个任务文件里复制一份。

多标签开关:配置里写 ``Loss.multi_label: true``（或命令行 ``-o Loss.multi_label=true``）
即可自动切换 BCE 多标签损失 / 阈值后处理 / mAP·mA 指标,无需再写一串 name。
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
    "build_cls_post_process",
    "is_multi_label_loss_cfg",
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


def is_multi_label_loss_cfg(loss_cfg):
    cfg = loss_cfg or {}
    name = str(cfg.get("name") or "CrossEntropy")
    return bool(cfg.get("multi_label")) or name in ("MultiLabelLoss", "MultiLabel")


def build_cls_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    multi = bool(cfg.pop("multi_label", False)) or str(
        cfg.get("name", "CrossEntropy")
    ) in ("MultiLabelLoss", "MultiLabel")
    name = cfg.pop("name", "CrossEntropy")
    if multi or name in ("MultiLabelLoss", "MultiLabel"):
        from torchkiln.tasks.attribute import MultiLabelLoss

        # classify 友好默认:mean 聚合、不做类平衡 ratio(需 label_ratio 时再显式开)
        cfg.setdefault("size_sum", False)
        cfg.setdefault("weight_ratio", False)
        # 丢掉单标签 CE / 检测风格残留键
        for key in ("topk", "alpha", "beta", "cls_gain", "box_gain", "label_smoothing"):
            cfg.pop(key, None)
        return MultiLabelLoss(**cfg)
    if name not in ("CrossEntropy", "ClsLoss"):
        raise ValueError("Unknown cls loss: {}".format(name))
    return ClsLoss(**cfg)


def build_cls_metric(metric_cfg, loss_cfg=None):
    cfg = dict(metric_cfg or {})
    name = cfg.pop("name", None)
    multi = is_multi_label_loss_cfg(loss_cfg) or name in (
        "AttrMetric",
        "MultiLabelMetric",
    )
    if multi:
        from torchkiln.tasks.attribute import AttrMetric

        cfg.setdefault("main_indicator", "mAP")
        cfg.setdefault("threshold", 0.5)
        # 丢掉 topk 等 ClsMetric 专属键
        cfg.pop("topk", None)
        return AttrMetric(**cfg)
    return ClsMetric(**cfg)


def build_cls_post_process(post_cfg, loss_cfg=None):
    cfg = dict(post_cfg or {})
    name = cfg.pop("name", None)
    multi = is_multi_label_loss_cfg(loss_cfg) or name in (
        "MultiLabelThresPostProcess",
        "MultiLabelThres",
    )
    if multi:
        from torchkiln.tasks.attribute import MultiLabelThresPostProcess

        cfg.pop("topk", None)
        cfg.pop("conf_thres", None)
        cfg.pop("iou_thres", None)
        cfg.pop("strides", None)
        cfg.setdefault("threshold", 0.5)
        return MultiLabelThresPostProcess(**cfg)
    # 单标签:丢掉检测后处理残留键
    for key in ("conf_thres", "iou_thres", "strides", "threshold", "label_list"):
        cfg.pop(key, None)
    return ClsPostProcess(**cfg)
