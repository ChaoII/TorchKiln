"""Pose (keypoints) components: loss / post-process / OKS-mAP metric.

Keypoint convention: the head predicts per-anchor *grid-unit offsets*; decoding
gives letterboxed pixel coordinates ``(x, y, vis)`` per keypoint.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchx.det.loss import DetLoss
from pytorchx.det.metric import _ap_101
from pytorchx.det.ops import bbox_ciou, dist2bbox, make_anchors, nms, split_head
from pytorchx.det.postprocess import DetPostProcess

__all__ = [
    "PoseLoss",
    "PosePostProcess",
    "PoseMetric",
    "build_pose_loss",
    "build_pose_postprocess",
    "build_pose_metric",
    "COCO_SIGMAS",
]

# COCO 17-keypoint constants
COCO_SIGMAS = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
    dtype=np.float32,
)


class PoseLoss(DetLoss):
    def __init__(
        self,
        num_classes=80,
        kpt_shape=(17, 3),
        kpt_gain=12.0,
        vis_gain=1.0,
        strides=(8, 16, 32),
        topk=13,
        alpha=1.0,
        beta=6.0,
        cls_gain=0.5,
        box_gain=7.5,
        reg_max=1,
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
            reg_max=reg_max,
        )
        self.nk = int(kpt_shape[0])
        self.kpt_gain = float(kpt_gain)
        self.vis_gain = float(vis_gain)

    def _split_full(self, feats):
        dist, cls, kpt = split_head(feats, self.num_classes, self.reg_max, "pose")
        return dist, cls, kpt

    def forward(self, preds, batch):
        pred_distri, pred_scores, pred_kpt = self._split_full(preds)
        anchor_points, stride_tensor = make_anchors(preds, self.strides)
        anchors_px = anchor_points * stride_tensor
        pred_bboxes = dist2bbox(pred_distri, anchor_points, xywh=False) * stride_tensor

        device = pred_scores.device
        targets = batch[1].to(device)
        mask_gt = batch[2].unsqueeze(-1).bool().to(device)
        gt_kpt = batch[3].to(device)  # (B, M, K, 3) letterboxed px
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
        else:
            loss_box = pred_bboxes.sum() * 0.0

        if bool(fg.any()):
            b_idx = torch.arange(pred_scores.shape[0], device=device)[:, None].expand_as(fg)[fg]
            tgt_kpt = gt_kpt[b_idx, t_gt_idx[fg]]  # (N, K, 3)
            B = pred_scores.shape[0]
            anchor_fg = anchor_points.unsqueeze(0).expand(B, -1, -1)[fg]  # (N, 2)
            stride_fg = stride_tensor.unsqueeze(0).expand(B, -1, -1)[fg]  # (N, 1)
            pred = pred_kpt[fg].view(-1, self.nk, 3)
            # offsets -> pixel coords
            xy = (anchor_fg.unsqueeze(1) + pred[..., 0:2]) * stride_fg.unsqueeze(1)
            vis = pred[..., 2]

            visible = (tgt_kpt[..., 2] > 0).float()
            vis_target = tgt_kpt[..., 2].clamp(0, 1)
            loss_vis = (
                self.bce(vis, vis_target) * visible
            ).sum() / max(float(visible.sum()), 1.0)
            if float(visible.sum()) > 0:
                loss_xy = (
                    F.l1_loss(xy, tgt_kpt[..., 0:2], reduction="none").sum(-1) * visible
                ).sum() / max(float(visible.sum()), 1.0)
            else:
                loss_xy = xy.sum() * 0.0
            loss_kpt = self.kpt_gain * loss_xy + self.vis_gain * loss_vis
        else:
            loss_kpt = pred_kpt.sum() * 0.0

        loss = self.box_gain * loss_box + self.cls_gain * loss_cls + loss_kpt
        return {
            "loss": loss,
            "loss_box": loss_box.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_kpt": loss_kpt.detach(),
        }


class PosePostProcess(DetPostProcess):
    def __init__(self, conf_thres=0.25, iou_thres=0.7, max_det=300, strides=(8, 16, 32),
                 kpt_shape=(17, 3), reg_max=1, **kwargs):
        super().__init__(
            conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
            strides=strides, box_type="xyxy", reg_max=reg_max,
        )
        self.nk = int(kpt_shape[0])

    def __call__(self, preds):
        num_classes = preds[0].shape[1] - self.reg_channels - self.nk * 3
        distri, scores, kpts_all = split_head(
            preds, num_classes, self.reg_max, "pose"
        )
        scores = scores.sigmoid()
        anchor_points, stride_tensor = make_anchors(preds, self.strides)
        boxes = dist2bbox(distri, anchor_points, xywh=False) * stride_tensor

        # decode keypoints: anchor offset (grid units) -> pixels
        raw = kpts_all.view(kpts_all.shape[0], -1, self.nk, 3)
        xy = (anchor_points.unsqueeze(0).unsqueeze(2) + raw[..., 0:2]) * stride_tensor.view(
            1, -1, 1, 1
        )
        kpts_all = torch.cat([xy, raw[..., 2:3]], dim=-1)

        results = []
        for b in range(boxes.shape[0]):
            bx, sc, kp = boxes[b], scores[b], kpts_all[b]
            conf, labels = sc.max(-1)
            keep = conf > self.conf_thres
            bx, conf, labels, kp = bx[keep], conf[keep], labels[keep], kp[keep]
            if bx.numel() == 0:
                results.append(
                    {"bboxes": bx.reshape(0, 4), "scores": conf, "labels": labels,
                     "kpts": kp.reshape(0, self.nk, 3)}
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
    """Keypoint mAP: AP over OKS thresholds (COCO-style), 101-point."""

    def __init__(self, kpt_sigmas=None, iou_thresholds=None, main_indicator="mAP50-95", **kwargs):
        sigmas = np.asarray(kpt_sigmas, dtype=np.float32) if kpt_sigmas is not None else None
        self.sigmas = sigmas
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
            return COCO_SIGMAS
        return np.full((k,), 0.1, dtype=np.float32)

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

    @staticmethod
    def _oks(pred_k, gt_k, area, sigmas):
        vis = gt_k[:, 2] > 0
        if vis.sum() == 0:
            return 0.0
        d = ((pred_k[:, 0:2] - gt_k[:, 0:2]) ** 2).sum(-1)
        e = d / (2.0 * area * (sigmas**2) + 1e-9)
        return float((np.exp(-e) * vis).sum() / vis.sum())

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
                    p = self.preds[i]
                    sel_p = p["labels"] == c
                    if not sel_p.any():
                        continue
                    pb_k = p["kpts"][sel_p]
                    ps = p["scores"][sel_p]
                    order = np.argsort(-ps)
                    pb_k, ps = pb_k[order], ps[order]
                    matches = np.zeros(gt_k.shape[0], dtype=bool)
                    for j in range(pb_k.shape[0]):
                        best, best_i = 0.0, -1
                        for gi in range(gt_k.shape[0]):
                            w = gt_b[gi, 2] - gt_b[gi, 0]
                            h = gt_b[gi, 3] - gt_b[gi, 1]
                            area = float(max(w * h, 1.0))
                            sig = self._sigma_for(gt_k.shape[1])
                            oks = self._oks(pb_k[j], gt_k[gi], area, sig)
                            if oks > best:
                                best, best_i = oks, gi
                        if best_i >= 0 and best >= thr and not matches[best_i]:
                            matches[best_i] = True
                            tps.append(1)
                        else:
                            tps.append(0)
                        scores.append(ps[j])
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


def build_pose_loss(loss_cfg, num_classes, kpt_shape=None, reg_max=None):
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
