"""Detection components (loss / post-process / metric) + config builders."""
from __future__ import absolute_import

from torchkiln.det.loss import DetLoss, ObbLoss, TaskAlignedAssigner
from torchkiln.det.metric import DetMetric
from torchkiln.det.postprocess import DetPostProcess

__all__ = [
    "DetLoss",
    "ObbLoss",
    "TaskAlignedAssigner",
    "DetMetric",
    "DetPostProcess",
    "build_det_loss",
    "build_obb_loss",
    "build_det_metric",
    "build_det_postprocess",
]


def build_det_loss(loss_cfg, num_classes, reg_max=1, use_one2one=False, class_weights=None):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "DetLoss")
    if name not in ("DetLoss", "DetectionLoss"):
        raise ValueError("Unknown det loss: {}".format(name))
    cfg.setdefault("reg_max", reg_max)
    cfg.setdefault("use_one2one", use_one2one)
    loss = DetLoss(num_classes=num_classes, **cfg)
    if class_weights is not None:
        import torch as _t

        w = list(class_weights)[: int(num_classes)]
        w += [1.0] * (int(num_classes) - len(w))          # 补齐到类别数
        loss.class_weights = _t.tensor(w, dtype=_t.float32).view(1, 1, -1)
    return loss


def build_obb_loss(loss_cfg, num_classes, reg_max=1, reg_layout="ltrb_angle", ne=1):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "ObbLoss")
    if name not in ("ObbLoss", "OBBLoss"):
        raise ValueError("Unknown obb loss: {}".format(name))
    cfg.setdefault("reg_max", reg_max)
    cfg.setdefault("reg_layout", reg_layout)
    cfg.setdefault("ne", ne)
    return ObbLoss(num_classes=num_classes, **cfg)


def build_det_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    return DetMetric(**cfg)


def build_det_postprocess(pp_cfg, head=None):
    """``head`` supplies ``reg_max`` / ``reg_layout`` / ``ne`` when the yml omits them."""
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    head = head or {}
    for key in ("reg_max", "reg_layout", "ne", "box_type", "end2end"):
        if head.get(key) is not None and key not in cfg:
            cfg[key] = head[key]
    return DetPostProcess(**cfg)
