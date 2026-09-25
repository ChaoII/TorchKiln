"""Lane detection components: seg (mask) + row (UFLD-style) loss/metric/postprocess."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LaneSegLoss",
    "LaneSegMetric",
    "LaneRowLoss",
    "LaneRowMetric",
    "LaneRowPostProcess",
    "build_lane_seg_loss",
    "build_lane_seg_metric",
    "build_lane_seg_postprocess",
    "build_lane_row_loss",
    "build_lane_row_metric",
    "build_lane_row_postprocess",
]


class LaneSegLoss(nn.Module):
    """CE + optional dice + focal for sparse lane masks (SOTA TwinLiteNet/UFLD hybrid)."""

    def __init__(
        self,
        ignore_index=255,
        dice_weight=0.5,
        focal_weight=0.5,
        focal_gamma=2.0,
        label_smoothing=0.0,
        **kwargs
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=float(label_smoothing)
        )
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.ignore_index = int(ignore_index)

    def _dice(self, logits, target):
        num_classes = logits.shape[1]
        probs = logits.softmax(1)
        valid = (target != self.ignore_index).unsqueeze(1)
        one_hot = (
            F.one_hot(target.clamp(0, num_classes - 1), num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        probs = probs * valid
        one_hot = one_hot * valid
        inter = (probs * one_hot).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2 * inter + 1.0) / (denom + 1.0)
        return 1.0 - dice.mean()

    def _focal(self, logits, target):
        ce = F.cross_entropy(
            logits, target, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce.clamp(min=1e-12))
        focal = ((1.0 - pt) ** self.focal_gamma) * ce
        mask = (target != self.ignore_index).float()
        return (focal * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(self, preds, batch):
        mask = batch[1]
        if preds.shape[-2:] != mask.shape[-2:]:
            preds = F.interpolate(
                preds, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = self.ce(preds, mask.long())
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self._dice(preds, mask.long())
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self._focal(preds, mask.long())
        return {"loss": loss}


class LaneSegMetric(object):
    """Foreground (lane) IoU + per-class mIoU."""

    def __init__(
        self, num_classes=None, main_indicator="lane_IoU", ignore_index=255, **kwargs
    ):
        self.num_classes = num_classes
        self.main_indicator = main_indicator
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self):
        self.inter_fg = 0
        self.union_fg = 0
        self.conf = None
        self.correct = 0
        self.total = 0

    def __call__(self, post_result, batch):
        pred = post_result
        target = batch[1]
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred)
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)
        pred = pred.detach().cpu().numpy().astype(np.int64)
        target = target.detach().cpu().numpy().astype(np.int64)
        valid = target != self.ignore_index
        p, t = pred[valid], target[valid]
        if p.size == 0:
            return
        fg_p = p > 0
        fg_t = t > 0
        self.inter_fg += int(np.logical_and(fg_p, fg_t).sum())
        self.union_fg += int(np.logical_or(fg_p, fg_t).sum())
        n_cls = self.num_classes or int(max(p.max(), t.max())) + 1
        if self.conf is None:
            self.conf = np.zeros((n_cls, n_cls), dtype=np.int64)
        elif self.conf.shape[0] < n_cls:
            pad = n_cls - self.conf.shape[0]
            self.conf = np.pad(self.conf, ((0, pad), (0, pad)))
        for ti, pi in zip(t, p):
            if ti < self.conf.shape[0] and pi < self.conf.shape[1]:
                self.conf[ti, pi] += 1
        self.correct += int((p == t).sum())
        self.total += int(p.size)

    def get_metric(self):
        lane_iou = (
            float(self.inter_fg / self.union_fg) if self.union_fg > 0 else 0.0
        )
        miou = 0.0
        acc = 0.0
        if self.conf is not None and self.conf.sum() > 0:
            inter = np.diag(self.conf).astype(np.float64)
            union = (
                self.conf.sum(0) + self.conf.sum(1) - np.diag(self.conf)
            ).astype(np.float64)
            present = union > 0
            if present.any():
                miou = float((inter[present] / union[present]).mean())
            acc = float(self.correct / max(self.total, 1))
        return {
            "lane_IoU": lane_iou,
            "mIoU": miou,
            "acc": acc,
            self.main_indicator: lane_iou if self.main_indicator == "lane_IoU" else miou,
        }


class LaneRowPostProcess(object):
    """(B, L, R, W) bin logits -> normalized x in [0, 1] (invalid = -1)."""

    def __init__(self, ignore_value=-1.0, **kwargs):
        self.ignore_value = float(ignore_value)

    def __call__(self, preds, size=None):
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        # (B, L, R, W) or (B, L*R, W)
        if preds.dim() == 3:
            b, lr, w = preds.shape
            # assume L from config is unknown; leave as (B, L*R, W) -> expand later
            bin_idx = preds.argmax(-1)
            bins = max(w - 1, 1)
            return bin_idx.float() / float(bins)
        if preds.dim() != 4:
            raise ValueError("LaneRow expects (B,L,R,W) logits, got {}".format(tuple(preds.shape)))
        bin_idx = preds.argmax(-1)  # (B, L, R)
        bins = max(preds.shape[-1] - 1, 1)
        return bin_idx.float() / float(bins)


class LaneRowMetric(object):
    """Row-wise F1 / accuracy under a pixel threshold (TuSimple-style simplification)."""

    def __init__(self, main_indicator="F1", threshold_px=50, image_width=256, **kwargs):
        self.main_indicator = main_indicator
        self.threshold_px = float(threshold_px)
        self.image_width = float(image_width)
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.hit = 0
        self.total = 0

    def __call__(self, post_result, batch):
        # post: (B, L, R) x in [0,1]; batch: [img, xs, valid]
        pred = post_result
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred)
        xs = batch[1]
        valid = batch[2]
        if not torch.is_tensor(xs):
            xs = torch.as_tensor(xs)
        if not torch.is_tensor(valid):
            valid = torch.as_tensor(valid)
        pred = pred.detach().cpu()
        xs = xs.detach().cpu().float()
        valid = valid.detach().cpu().bool()
        # align lane dim if pred flattened
        if pred.dim() == 2:
            # (B, L*R) not expected in v0
            return
        b, lr, r = pred.shape if pred.dim() == 3 else (0, 0, 0)
        if pred.dim() != 3 or xs.dim() != 3:
            return
        # match shapes: pred (B,L,R) xs (B,L,R)
        if pred.shape != xs.shape:
            min_l = min(pred.shape[1], xs.shape[1])
            min_r = min(pred.shape[2], xs.shape[2])
            pred = pred[:, :min_l, :min_r]
            xs = xs[:, :min_l, :min_r]
            valid = valid[:, :min_l, :min_r]
        thr = self.threshold_px / max(self.image_width, 1.0)
        err = (pred - xs).abs()
        ok = (err <= thr) & valid
        self.hit += int(ok.sum())
        self.total += int(valid.sum())
        # per-lane presence for F1: lane has any valid row -> predict if any row close
        for bi in range(pred.shape[0]):
            for li in range(pred.shape[1]):
                v = valid[bi, li]
                if not bool(v.any()):
                    continue
                # GT lane exists
                rows_ok = ok[bi, li] & v
                # require majority of valid rows hit for TP
                if float(rows_ok.sum()) / max(int(v.sum()), 1) >= 0.5:
                    self.tp += 1
                else:
                    self.fn += 1

    def get_metric(self):
        acc = float(self.hit / max(self.total, 1))
        prec = float(self.tp / max(self.tp + self.fn, 1))  # simplified (no FP lanes)
        rec = float(self.tp / max(self.tp + self.fn, 1))
        f1 = float(2 * prec * rec / max(prec + rec, 1e-12))
        return {
            "F1": f1,
            "acc": acc,
            "precision": prec,
            "recall": rec,
            self.main_indicator: f1 if self.main_indicator == "F1" else acc,
        }


class LaneRowLoss(nn.Module):
    """Cross-entropy over x bins per (lane, row), masked by validity."""

    def __init__(self, num_bins=101, ignore_index=-100, label_smoothing=0.0, **kwargs):
        super().__init__()
        self.num_bins = int(num_bins)
        self.ignore_index = int(ignore_index)
        self.ce = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, label_smoothing=float(label_smoothing)
        )

    def forward(self, preds, batch):
        # preds: (B, L, R, W) logits
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if preds.dim() == 3:
            # (B, L*R, W) -> cannot recover L without config; expect 4D
            raise ValueError("LaneRowLoss expects (B,L,R,W) logits, got {}".format(tuple(preds.shape)))
        xs = batch[1].float()  # (B, L, R) in [0,1]
        valid = batch[2].bool()  # (B, L, R)
        b, l, r, w = preds.shape
        bins = w
        # map x in [0,1] -> bin index
        bin_idx = torch.clamp((xs * (bins - 1)).round().long(), 0, bins - 1)
        target = bin_idx.view(b * l * r)
        target = torch.where(
            valid.view(b * l * r), target,
            torch.full_like(target, self.ignore_index),
        )
        logits = preds.view(b * l * r, w)
        loss = self.ce(logits, target)
        return {"loss": loss}


def build_lane_seg_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "LaneSegLoss")
    if name in ("SemLoss", "CrossEntropy"):
        from torchkiln.sem import SemLoss

        return SemLoss(**cfg)
    if name != "LaneSegLoss":
        raise ValueError("Unknown lane_seg loss: {}".format(name))
    return LaneSegLoss(**cfg)


def build_lane_seg_metric(metric_cfg, num_classes=None):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    cfg.setdefault("num_classes", num_classes)
    return LaneSegMetric(**cfg)


def build_lane_seg_postprocess(pp_cfg):
    from torchkiln.sem import SemPostProcess

    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    return SemPostProcess(**cfg)


def build_lane_row_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "LaneRowLoss")
    if name != "LaneRowLoss":
        raise ValueError("Unknown lane_row loss: {}".format(name))
    return LaneRowLoss(**cfg)


def build_lane_row_metric(metric_cfg, **kwargs):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    cfg.update(kwargs)
    return LaneRowMetric(**cfg)


def build_lane_row_postprocess(pp_cfg):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    return LaneRowPostProcess(**cfg)
