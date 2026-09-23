"""Pose (keypoints) components: loss / post-process / OKS-mAP metric.

Aligned 1:1 with ultralytics (``v8PoseLoss`` / ``PoseValidator``):

* keypoint decode (inference): ``x = (raw*2 + (anchor-0.5)) * stride``;
* keypoint decode (loss): same but *without* the ``* stride`` (grid units);
* ``KeypointLoss``: OKS-style ``e = d / ((2*sigma)^2 * area * 2)``, masked &
  factor-normalised like cocoeval;
* visibility loss: ``BCEWithLogits`` on raw visibility logits (only when
  ``kpt_shape[1] == 3``);
* metric: ``kpt_iou`` (OKS), ``area = w*h*0.53``, ``sigma = OKS_SIGMA/10`` for
  COCO 17-kpt or ``ones(nk)/nk`` otherwise.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkiln.det.loss import DetLoss, df_loss
from torchkiln.det.metric import _ap_101
from torchkiln.det.ops import (
    bbox_ciou,
    bbox2dist,
    dfl_project,
    dist2bbox,
    make_anchors,
    nms,
    split_head,
    xyxy2xywh,
)
from torchkiln.det.postprocess import DetPostProcess

__all__ = [
    "PoseLoss",
    "PosePostProcess",
    "PoseMetric",
    "build_pose_loss",
    "build_pose_postprocess",
    "build_pose_metric",
    "OKS_SIGMA",
]

# COCO 17-keypoint sigmas (scaled /10), matching ultralytics ``OKS_SIGMA``.
OKS_SIGMA = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
    dtype=np.float32,
) / 10.0


class PoseLoss(DetLoss):
    """Keypoint loss ported from ``ultralytics.v8PoseLoss``."""

    def __init__(
        self,
        num_classes=80,
        kpt_shape=(17, 3),
        strides=(8, 16, 32),
        topk=13,
        alpha=1.0,
        beta=6.0,
        cls_gain=0.5,
        box_gain=7.5,
        dfl_gain=1.5,
        reg_max=1,
        pose_gain=12.0,
        kobj_gain=1.0,
        kpt_oks_sigmas=None,
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            strides=strides,
            topk=topk,
            alpha=alpha,
            beta=beta,
            cls_gain=cls_gain,
            box_gain=box_gain,
            dfl_gain=dfl_gain,
            reg_max=reg_max,
        )
        self.nk = int(kpt_shape[0])
        self.ndim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
        self.pose_gain = float(pose_gain)
        self.kobj_gain = float(kobj_gain)
        self.kpt_shape = [int(kpt_shape[0]), int(kpt_shape[1])]
        if kpt_oks_sigmas is not None:
            sigmas = torch.as_tensor(kpt_oks_sigmas, dtype=torch.float32).flatten()
        elif self.kpt_shape == [17, 3]:
            sigmas = torch.from_numpy(OKS_SIGMA).float()
        else:
            sigmas = torch.ones(self.nk) / self.nk
        self.sigmas = sigmas

    def _split(self, feats):
        return split_head(feats, self.num_classes, self.reg_max, "pose", project=False)

    def _keypoint_loss(self, pred_kpt_extra, anchor_points, stride_tensor, fg, t_gt_idx,
                       t_bboxes, gt_kpt, device):
        """Returns ``(loss_pose, loss_kobj)`` aligned with ultralytics."""
        B, A = fg.shape
        if not bool(fg.any()):
            z = pred_kpt_extra.sum() * 0.0
            return z, z

        raw = pred_kpt_extra.view(B, A, self.nk, self.ndim)
        # decode to grid units (loss path): x*2 + (anchor - 0.5)
        a = anchor_points.unsqueeze(0).unsqueeze(2)  # (1, A, 1, 2)
        xy = raw[..., 0:2] * 2.0 + (a - 0.5)  # (B, A, nk, 2)

        gt_kpt = gt_kpt.to(device).float()  # (B, M, nk, ndim) pixels
        b_idx = torch.arange(B, device=device)[:, None].expand_as(fg)[fg]
        tgt_kpt = gt_kpt[b_idx, t_gt_idx[fg]]  # (N, nk, ndim)
        stride_fg = stride_tensor.squeeze(-1).unsqueeze(0).expand_as(fg)[fg]  # (N,)
        tgt_xy = tgt_kpt[..., 0:2] / stride_fg[:, None, None]
        # ultralytics `calculate_keypoints_loss`: target_bboxes 先除 stride 转网格单位再算 area
        tb = t_bboxes[fg] / stride_fg[:, None]
        area = xyxy2xywh(tb)[:, 2:].prod(1, keepdim=True)  # (N, 1) grid^2

        pred_xy = xy[fg]  # (N, nk, 2) grid units
        if self.ndim == 3:
            kpt_mask = (tgt_kpt[..., 2] != 0).float()  # (N, nk)
        else:
            kpt_mask = torch.ones_like(tgt_kpt[..., 0])

        d = (pred_xy[..., 0] - tgt_xy[..., 0]) ** 2 + (pred_xy[..., 1] - tgt_xy[..., 1]) ** 2  # (N, nk)
        kpt_loss_factor = kpt_mask.shape[1] / (kpt_mask.sum(dim=1) + 1e-9)  # (N,)
        sig = self.sigmas.to(device)
        e = d / ((2.0 * sig).pow(2) * (area + 1e-9) * 2.0)  # (N, nk)
        loss_pose = (kpt_loss_factor.unsqueeze(-1) * ((1.0 - torch.exp(-e)) * kpt_mask)).mean()

        if self.ndim == 3:
            vis = raw[..., 2:3][fg].squeeze(-1)  # raw logits
            loss_kobj = F.binary_cross_entropy_with_logits(vis, kpt_mask)
        else:
            loss_kobj = pred_kpt_extra.sum() * 0.0

        return loss_pose, loss_kobj

    def forward(self, preds, batch):
        pred_dist_raw, pred_scores, pred_kpt_extra = self._split(preds)
        pred_distri = (
            dfl_project(pred_dist_raw, self.reg_max) if self.reg_max > 1 else pred_dist_raw
        )
        anchor_points, stride_tensor = make_anchors(preds, self.strides)
        anchors_px = anchor_points * stride_tensor
        pred_bboxes = dist2bbox(pred_distri, anchor_points, xywh=False) * stride_tensor

        device = pred_scores.device
        targets = batch[1].to(device)
        mask_gt = batch[2].unsqueeze(-1).bool().to(device)
        gt_labels = targets[..., 0:1]
        gt_bboxes = targets[..., 1:5]

        with torch.no_grad():
            t_labels, t_bboxes, t_scores, fg, t_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                pred_bboxes.detach(),
                anchors_px,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        scores_sum = max(float(t_scores.sum()), 1.0)
        loss_cls = self.bce(pred_scores, t_scores).sum() / scores_sum

        if bool(fg.any()):
            weight = t_scores.sum(-1)[fg].detach()
            loss_box = (bbox_ciou(pred_bboxes[fg], t_bboxes[fg]) * weight).sum() / scores_sum
            if self.reg_max > 1:
                b_sz, a_sz = fg.shape
                pd = pred_dist_raw.view(b_sz, a_sz, 4, self.reg_max)[fg].reshape(-1, self.reg_max)
                stride_fg = stride_tensor.squeeze(-1).unsqueeze(0).expand_as(fg)[fg]
                tgt = (t_bboxes[fg] / stride_fg[:, None]).clamp(0, self.reg_max - 1.0 - 1e-3)
                ap_fg = anchor_points.unsqueeze(0).expand(fg.shape[0], -1, -1)[fg]
                target_ltrb = bbox2dist(ap_fg, tgt).clamp(0, self.reg_max - 1.0 - 1e-3)
                per_side = df_loss(pd, target_ltrb.reshape(-1)).view(-1, 4).mean(-1)
                loss_dfl = (per_side * weight).sum() / scores_sum
            else:
                loss_dfl = pred_dist_raw.sum() * 0.0
        else:
            loss_box = pred_bboxes.sum() * 0.0
            loss_dfl = pred_dist_raw.sum() * 0.0

        loss_pose, loss_kobj = self._keypoint_loss(
            pred_kpt_extra, anchor_points, stride_tensor, fg, t_gt_idx, t_bboxes,
            batch[3], device,
        )

        loss = (
            self.box_gain * loss_box
            + self.cls_gain * loss_cls
            + self.dfl_gain * loss_dfl
            + self.pose_gain * loss_pose
            + self.kobj_gain * loss_kobj
        )
        return {
            "loss": loss * int(batch[0].shape[0]),
            "loss_box": loss_box.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_dfl": loss_dfl.detach(),
            "loss_pose": loss_pose.detach(),
            "loss_kobj": loss_kobj.detach(),
        }


# placeholder, replaced by a module-level hook set in build_pose_loss
class PosePostProcess(DetPostProcess):
    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.7,
        max_det=300,
        strides=(8, 16, 32),
        kpt_shape=(17, 3),
        reg_max=1,
        **kwargs
    ):
        super().__init__(
            conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
            strides=strides, box_type="xyxy", reg_max=reg_max,
        )
        self.nk = int(kpt_shape[0])
        self.ndim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3

    def __call__(self, preds):
        num_classes = preds[0].shape[1] - 4 * self.reg_max - self.nk * self.ndim
        distri, scores, kpts_all = split_head(
            preds, num_classes, self.reg_max, "pose"
        )
        scores = scores.sigmoid()
        anchor_points, stride_tensor = make_anchors(preds, self.strides)
        boxes = dist2bbox(distri, anchor_points, xywh=False) * stride_tensor

        # decode keypoints: (raw*2 + (anchor-0.5)) * stride
        raw = kpts_all.view(kpts_all.shape[0], -1, self.nk, self.ndim)
        a = anchor_points.unsqueeze(0).unsqueeze(2)
        xy = (raw[..., 0:2] * 2.0 + (a - 0.5)) * stride_tensor.view(1, -1, 1, 1)
        if self.ndim == 3:
            vis = raw[..., 2:3].sigmoid()
            kpts_all = torch.cat([xy, vis], dim=-1)
        else:
            kpts_all = xy

        results = []
        for b in range(boxes.shape[0]):
            bx, sc, kp = boxes[b], scores[b], kpts_all[b]
            conf, labels = sc.max(-1)
            keep = conf > self.conf_thres
            bx, conf, labels, kp = bx[keep], conf[keep], labels[keep], kp[keep]
            if bx.numel() == 0:
                results.append(
                    {"bboxes": bx.reshape(0, 4), "scores": conf, "labels": labels,
                     "kpts": kp.reshape(0, self.nk, self.ndim)}
                )
                continue
            kept = []
            for cls in labels.unique():
                idx = (labels == cls).nonzero(as_tuple=True)[0]
                sel = nms(bx[idx], conf[idx], self.iou_thres)
                kept.append(idx[sel])
            kept = torch.cat(kept) if kept else torch.empty(0, dtype=torch.long)
            if kept.numel() > self.max_det:
                order = conf[kept].argsort(descending=True)[: self.max_det]
                kept = kept[order]
            results.append(
                {"bboxes": bx[kept], "scores": conf[kept], "labels": labels[kept],
                 "kpts": kp[kept]}
            )
        return results


class PoseMetric(object):
    """Keypoint mAP via ``kpt_iou`` (OKS), matching ultralytics ``PoseValidator``."""

    def __init__(self, kpt_oks_sigmas=None, iou_thresholds=None, main_indicator="mAP50-95", **kwargs):
        self.sigmas = None
        if kpt_oks_sigmas is not None:
            self.sigmas = np.asarray(kpt_oks_sigmas, dtype=np.float32).flatten()
        self.iou_thresholds = (
            list(iou_thresholds)
            if iou_thresholds is not None
            else [round(0.5 + 0.05 * i, 2) for i in range(10)]
        )
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.preds = []
        self.gts = []

    def _sigma_for(self, k):
        if self.sigmas is not None and len(self.sigmas) == k:
            return self.sigmas
        if k == 17:
            return OKS_SIGMA
        return np.ones(k, dtype=np.float32) / k

    def __call__(self, post_result, batch):
        targets = batch[1].detach().cpu().numpy()
        mask = batch[2].detach().cpu().numpy().astype(bool)
        kpts = batch[3].detach().cpu().numpy()
        for i, pred in enumerate(post_result):
            self.preds.append(
                {
                    "scores": pred["scores"].detach().cpu().numpy().astype(np.float32),
                    "labels": pred["labels"].detach().cpu().numpy().astype(np.int64),
                    "kpts": pred["kpts"].detach().cpu().numpy().astype(np.float32),
                }
            )
            m = mask[i]
            self.gts.append(
                {
                    "boxes": targets[i][m][:, 1:5].astype(np.float32),
                    "labels": targets[i][m][:, 0].astype(np.int64),
                    "kpts": kpts[i][m].astype(np.float32),
                }
            )

    def _kpt_iou(self, gt_k, gt_area, pred_k):
        """OKS of ``gt_k (N,nk,3)`` vs ``pred_k (M,nk,3)`` -> ``(N,M)``."""
        N = gt_k.shape[0]
        M = pred_k.shape[0]
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)
        sigma = self._sigma_for(gt_k.shape[1])
        d = (gt_k[:, None, :, 0] - pred_k[None, :, :, 0]) ** 2 + (
            gt_k[:, None, :, 1] - pred_k[None, :, :, 1]
        ) ** 2  # (N, M, nk)
        kpt_mask = gt_k[..., 2] != 0  # (N, nk)
        e = d / ((2.0 * np.asarray(sigma)) ** 2 * (gt_area[:, None, None] + 1e-9) * 2.0)
        return (np.exp(-e) * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + 1e-9)

    @staticmethod
    def _match(oks, thr):
        """Match GTs to preds per image, aligned with ultralytics ``match_predictions``.

        ``oks`` is ``(N_gt, M_pred)``.  Returns an ``(M_pred,)`` bool TP mask for
        the given OKS threshold (classes are already filtered by the caller).
        """
        M = oks.shape[1]
        matches = np.argwhere(oks >= thr)  # (P, 2) as (gt, pred)
        if matches.shape[0] == 0:
            return np.zeros(M, dtype=bool)
        vals = oks[matches[:, 0], matches[:, 1]]
        matches = matches[np.argsort(vals)[::-1]]  # highest OKS first
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct = np.zeros(M, dtype=bool)
        correct[matches[:, 1]] = True
        return correct

    def get_metric(self):
        n_cls = 0
        for g in self.gts:
            if g["labels"].size:
                n_cls = max(n_cls, int(g["labels"].max()) + 1)
        for p in self.preds:
            if p["labels"].size:
                n_cls = max(n_cls, int(p["labels"].max()) + 1)

        per_thr = {}
        for thr in self.iou_thresholds:
            aps = []
            for c in range(n_cls):
                scores, tps, n_gt = [], [], 0
                for i in range(len(self.gts)):
                    g = self.gts[i]
                    sel_g = g["labels"] == c
                    gt_k = g["kpts"][sel_g]
                    gt_b = g["boxes"][sel_g]
                    n_gt += int(sel_g.sum())
                    if gt_b.shape[0]:
                        # area = w*h*0.53 (COCO OKS convention, matching ultralytics)
                        area = (gt_b[:, 2] - gt_b[:, 0]) * (gt_b[:, 3] - gt_b[:, 1]) * 0.53
                    else:
                        area = np.zeros((0,), dtype=np.float32)
                    p = self.preds[i]
                    sel_p = p["labels"] == c
                    if not sel_p.any():
                        continue
                    pb_k = p["kpts"][sel_p]
                    ps = p["scores"][sel_p]
                    order = np.argsort(-ps)
                    pb_k, ps = pb_k[order], ps[order]
                    if gt_k.shape[0]:
                        oks = self._kpt_iou(gt_k, area, pb_k)  # (N, M)
                    else:
                        oks = np.zeros((0, pb_k.shape[0]), dtype=np.float32)
                    correct = self._match(oks, thr)
                    tps.extend(correct.astype(np.int64).tolist())
                    scores.extend(ps.tolist())
                if n_gt == 0:
                    continue
                if not scores:
                    aps.append(0.0)
                    continue
                order = np.argsort(-np.array(scores))
                tp = np.array(tps, dtype=np.float32)[order]
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(1.0 - tp)
                recall = cum_tp / max(n_gt, 1)
                precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
                aps.append(_ap_101(recall, precision))
            per_thr[thr] = float(np.mean(aps)) if aps else 0.0
        return {
            "mAP50": per_thr.get(0.5, 0.0),
            "mAP50-95": float(np.mean(list(per_thr.values()))) if per_thr else 0.0,
        }


def build_pose_loss(loss_cfg, num_classes, kpt_shape=None, reg_max=None, **kwargs):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "PoseLoss")
    if name not in ("PoseLoss", "KeypointLoss"):
        raise ValueError("Unknown pose loss: {}".format(name))
    if kpt_shape is not None and "kpt_shape" not in cfg:
        cfg["kpt_shape"] = kpt_shape
    if reg_max is not None and "reg_max" not in cfg:
        cfg["reg_max"] = reg_max
    return PoseLoss(num_classes=num_classes, **cfg)


def build_pose_postprocess(pp_cfg, kpt_shape=None, reg_max=None):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    if kpt_shape is not None and "kpt_shape" not in cfg:
        cfg["kpt_shape"] = kpt_shape
    if reg_max is not None and "reg_max" not in cfg:
        cfg["reg_max"] = reg_max
    return PosePostProcess(**cfg)


def build_pose_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    return PoseMetric(**cfg)
