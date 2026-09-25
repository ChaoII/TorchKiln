"""BEV-LaneDet task side: loss / post-process / metric (Paddle3D-aligned)."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BEVLaneDetLoss", "BEVLaneDetPostProcess", "BEVLaneDetMetric",
    "build_lane_bev_loss", "build_lane_bev_postprocess", "build_lane_bev_metric",
]


def _pairwise_dist(a, b):
    return (a.unsqueeze(-2) - b.unsqueeze(-3)).abs().sum(-1)


class _NDPushPullLoss(nn.Module):
    def __init__(self, var_weight, dist_weight, margin_var, margin_dist, ignore_label):
        super().__init__()
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.margin_var = margin_var
        self.margin_dist = margin_dist
        self.ignore_label = ignore_label

    def forward(self, featmap, gt):
        pull, push = [], []
        valid = gt[gt < self.ignore_label]
        C = int(valid.max().item()) if valid.numel() else 0
        for b in range(featmap.shape[0]):
            bfeat = featmap[b]
            bgt = gt[b][0]
            centers = {}
            for i in range(1, C + 1):
                m = bgt == i
                if m.sum() == 0:
                    continue
                pos = bfeat[:, m].t()  # (n, N)
                center = pos.mean(0, keepdim=True)
                centers[i] = center
                pull.append(F.relu(_pairwise_dist(pos, center) - self.margin_var).mean())
            for i in range(1, C + 1):
                for j in range(1, C + 1):
                    if i == j or i not in centers or j not in centers:
                        continue
                    push.append(F.relu(2 * self.margin_dist
                                       - _pairwise_dist(centers[i], centers[j])))
        zero = featmap.mean() * 0.0
        pl = torch.stack(pull).mean() * self.var_weight if pull else zero
        pu = torch.stack([p.mean() for p in push]).mean() * self.dist_weight if push else zero
        return pl + pu


class _IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        t = targets.float()
        num = (outputs * t * mask).sum()
        den = (outputs * mask + t * mask - outputs * t * mask).sum()
        return 1 - num / den.clamp(min=1e-6)


class BEVLaneDetLoss(nn.Module):
    def __init__(self, ignore_index=255, push_pull=True, **kwargs):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.bce_plain = nn.BCELoss()
        self.iou = _IoULoss(ignore_index)
        self.mse = nn.MSELoss()
        self.poopoo = _NDPushPullLoss(1.0, 1.0, 1.0, 5.0, 200) if push_pull else None

    def forward(self, preds, batch):
        seg, emb, off, z, seg2d, emb2d = preds
        gt_seg, gt_inst, gt_off, gt_z = batch[1], batch[2], batch[3], batch[4]
        img_seg, img_inst = batch[5], batch[6]
        loss_seg = self.bce(seg, gt_seg) + self.iou(seg.sigmoid(), gt_seg)
        loss_emb = self.poopoo(emb, gt_inst) if self.poopoo else seg.sum() * 0.0
        loss_off = self.bce_plain(gt_seg * off.sigmoid(), gt_off)
        loss_z = self.mse(gt_seg * z, gt_z)
        total = 3 * loss_seg + 0.5 * loss_emb + 60 * loss_off + 30 * loss_z
        loss_seg2d = self.bce(seg2d, img_seg) + self.iou(seg2d.sigmoid(), img_seg)
        loss_emb2d = self.poopoo(emb2d, img_inst) if self.poopoo else seg.sum() * 0.0
        total = total + 3 * loss_seg2d + 0.5 * loss_emb2d
        return {"loss": total, "loss_seg": (loss_seg + loss_seg2d).detach(),
                "loss_emb": (loss_emb + loss_emb2d).detach()}


class BEVLaneDetPostProcess(object):
    def __init__(self, score_thres=0.5, **kwargs):
        self.score_thres = float(score_thres)

    def __call__(self, preds, **kwargs):
        seg = preds[0] if isinstance(preds, (list, tuple)) else preds
        return (seg.sigmoid() > self.score_thres).float()  # (B,1,H,W)


class BEVLaneDetMetric(object):
    """Binary lane-segmentation F-score (main) + precision/recall."""

    def __init__(self, main_indicator="FScore", score_thres=0.5, **kwargs):
        self.main_indicator = main_indicator
        self.score_thres = float(score_thres)
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0

    def __call__(self, post_result, batch):
        pred = post_result.detach().cpu().numpy() > 0.5
        gt = batch[1].detach().cpu().numpy() > 0.5
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())

    def get_metric(self):
        p = self.tp / max(self.tp + self.fp, 1)
        r = self.tp / max(self.tp + self.fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        return {"FScore": f, "precision": p, "recall": r}


def build_lane_bev_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    cfg.pop("name", None)
    return BEVLaneDetLoss(**cfg)


def build_lane_bev_postprocess(pp_cfg):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    return BEVLaneDetPostProcess(**cfg)


def build_lane_bev_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    return BEVLaneDetMetric(**cfg)
