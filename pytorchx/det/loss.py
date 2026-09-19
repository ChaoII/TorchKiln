"""Detection loss: Task-Aligned Assigner + BCE (cls) + CIoU (box).

Independent implementation of the published task-alignment idea: each ground
truth selects its top-k anchors by an alignment metric
``score^alpha * IoU^beta``, and those anchors are trained towards the matched
box with a soft classification target proportional to the metric.
"""
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchx.det.ops import (
    batch_box_iou,
    batch_ciou,
    bbox2dist,
    bbox_ciou,
    dfl_project,
    dist2bbox,
    make_anchors,
    split_head,
)
from pytorchx.det.rbox import dist2rbox, points_in_rboxes, probiou, rbox2dist

__all__ = ["TaskAlignedAssigner", "DetLoss", "ObbLoss", "points_in_boxes", "df_loss"]


def df_loss(pred_dist, target):
    """Distribution Focal Loss(对齐 ultralytics)。

    ``pred_dist`` 为 ``(N, reg_max)`` 的 logits(已按 fg 展平),
    ``target`` 为 ``(N,)`` 的连续距离(单位:网格)。
    """
    tl = target.long()
    tr = tl + 1
    wl = tr.to(target.dtype) - target
    wr = 1.0 - wl
    left = F.cross_entropy(pred_dist, tl.clamp(0, pred_dist.shape[-1] - 2), reduction="none")
    right = F.cross_entropy(pred_dist, tr.clamp(1, pred_dist.shape[-1] - 1), reduction="none")
    return left * wl + right * wr


def points_in_boxes(points, boxes):
    """``points (A,2)`` inside axis-aligned ``boxes (B,M,4)`` -> bool ``(B,A,M)``."""
    ax = points[None, :, None, 0]
    ay = points[None, :, None, 1]
    return (
        (ax > boxes[:, None, :, 0])
        & (ax < boxes[:, None, :, 2])
        & (ay > boxes[:, None, :, 1])
        & (ay < boxes[:, None, :, 3])
    )


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner, ported 1:1 from ``ultralytics.utils.tal``.

    ``overlaps`` / ``align_metric`` use CIoU (``iou_fn``); OBB passes ``probiou`` +
    ``points_in_rboxes``. ``stride_val`` floors tiny GT sides so anchors can still
    fall inside very small objects. Soft targets are normalised exactly like the
    upstream implementation (align / overlap maxima).
    """

    def __init__(
        self,
        topk=13,
        num_classes=80,
        alpha=1.0,
        beta=6.0,
        stride=(8, 16, 32),
        eps=1e-9,
        iou_fn=None,
        inside_fn=None,
        topk2=None,
    ):
        super().__init__()
        self.topk = int(topk)
        self.topk2 = int(topk2) if topk2 else self.topk
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stride = tuple(int(s) for s in stride)
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = float(eps)
        self.iou_fn = iou_fn or batch_ciou
        self.inside_fn = inside_fn
        self.bs = 1
        self.n_max_boxes = 1

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        self.bs, self.n_max_boxes = gt_bboxes.shape[:2]
        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self._select_highest_overlaps(
            mask_pos, overlaps, align_metric
        )
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask, pd_scores.dtype
        )

        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        overlaps = overlaps * mask_pos
        pos_overlaps = overlaps.amax(dim=-1, keepdim=True)
        align_metric = align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        norm_align_metric = align_metric.amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt):
        mask_in_gts = self._select_candidates_in_gts(anchor_points, gt_bboxes, mask_gt)
        mask = mask_in_gts & mask_gt.bool()
        align_metric, overlaps = self._get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask
        )
        a = align_metric.shape[-1]
        topk_mask = mask_gt.expand(-1, -1, min(self.topk, a)).bool()
        mask_topk = self._select_topk_candidates(align_metric, topk_mask=topk_mask)
        mask_pos = mask_topk * mask_in_gts * mask_gt.bool()
        return mask_pos, align_metric, overlaps

    def _select_candidates_in_gts(self, anchor_points, gt_bboxes, mask_gt):
        """``(B, M, A)`` bool: anchor centres strictly inside each GT box."""
        if self.inside_fn is not None:
            return self.inside_fn(anchor_points, gt_bboxes).permute(0, 2, 1).bool()
        x1, y1, x2, y2 = gt_bboxes.chunk(4, -1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        sv = self.stride_val
        valid = mask_gt.bool()
        w = torch.where((w < sv) & valid, torch.full_like(w, float(sv)), w)
        h = torch.where((h < sv) & valid, torch.full_like(h, float(sv)), h)
        lt = torch.cat([cx - w / 2, cy - h / 2], dim=-1).unsqueeze(2)  # (B, M, 1, 2)
        rb = torch.cat([cx + w / 2, cy + h / 2], dim=-1).unsqueeze(2)
        ax = anchor_points[:, 0][None, None]
        ay = anchor_points[:, 1][None, None]
        return (
            ((ax - lt[..., 0]) > self.eps)
            & ((ay - lt[..., 1]) > self.eps)
            & ((rb[..., 0] - ax) > self.eps)
            & ((rb[..., 1] - ay) > self.eps)
        )

    def _get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        overlaps = self.iou_fn(pd_bboxes, gt_bboxes).permute(0, 2, 1)  # (B, M, A)
        gt_cls = gt_labels.squeeze(-1).long().clamp(min=0)  # (B, M)
        cls_prob = pd_scores.transpose(1, 2).gather(
            1, gt_cls[:, :, None].expand(-1, -1, pd_scores.shape[1])
        )
        align_metric = cls_prob.pow(self.alpha) * overlaps.clamp(min=0).pow(self.beta)
        return align_metric * mask_gt, overlaps * mask_gt

    def _select_topk_candidates(self, metrics, topk_mask=None):
        a = metrics.shape[-1]
        k = min(self.topk, a)
        topk_metrics, topk_idxs = torch.topk(metrics, k, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        else:
            topk_mask = topk_mask[..., :k].expand_as(topk_idxs)
        topk_idxs = topk_idxs.masked_fill(~topk_mask, 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        count.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
        count.masked_fill_(count > 1, 0)
        return count

    def _select_highest_overlaps(self, mask_pos, overlaps, align_metric):
        fg_mask = mask_pos.sum(-2)
        multi = (fg_mask.unsqueeze(1) > 1).expand(-1, self.n_max_boxes, -1)
        max_overlaps_idx = overlaps.argmax(1)
        is_max = torch.zeros_like(mask_pos)
        is_max.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        mask_pos = torch.where(multi, is_max, mask_pos)
        fg_mask = mask_pos.sum(-2)
        if self.topk2 != self.topk:
            am = align_metric * mask_pos
            idx = torch.topk(am, min(self.topk2, am.shape[-1]), dim=-1, largest=True).indices
            tk = torch.zeros_like(mask_pos)
            tk.scatter_(-1, idx, 1.0)
            mask_pos = mask_pos * tk
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, dtype):
        batch_ind = torch.arange(self.bs, device=gt_labels.device)[..., None]
        idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[idx]
        target_labels = target_labels.clamp_(0)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=dtype,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        target_scores = target_scores * (fg_mask[:, :, None] > 0)
        return target_labels, target_bboxes, target_scores


class DetLoss(nn.Module):
    def __init__(
        self,
        num_classes=80,
        strides=(8, 16, 32),
        topk=10,
        alpha=0.5,
        beta=6.0,
        cls_gain=0.5,
        box_gain=7.5,
        dfl_gain=1.5,
        label_smoothing=0.0,
        reg_max=1,
        use_one2one=False,
        one2one_gain=1.0,
        one2one_topk=1,
        **kwargs
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = tuple(int(s) for s in strides)
        self.cls_gain = float(cls_gain)
        self.box_gain = float(box_gain)
        self.dfl_gain = float(dfl_gain)
        self.label_smoothing = float(label_smoothing)
        self.reg_max = int(reg_max)
        self.assigner = TaskAlignedAssigner(
            topk, self.num_classes, alpha, beta, stride=self.strides
        )
        # end-to-end (NMS-free) training: a second, one-to-one assigner
        self.use_one2one = bool(use_one2one)
        self.one2one_gain = float(one2one_gain)
        if self.use_one2one:
            self.assigner_one2one = TaskAlignedAssigner(
                one2one_topk, self.num_classes, alpha, beta, stride=self.strides
            )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # 逆频率类别权重(ultralytics labels_to_class_weights 语义),None=不启用
        self.class_weights = None

    def _split(self, feats):
        dist, cls, _ = split_head(feats, self.num_classes, self.reg_max, "det")
        return dist, cls

    def forward(self, preds, batch):
        if (
            self.use_one2one
            and isinstance(preds, (tuple, list))
            and len(preds) == 2
            and isinstance(preds[0], (list, tuple))
            and isinstance(preds[1], (list, tuple))
        ):
            one2many, one2one = preds
            d1 = self._forward_one(one2many, batch, self.assigner)
            d2 = self._forward_one(one2one, batch, self.assigner_one2one)
            out = {"loss": d1["loss"] + self.one2one_gain * d2["loss"]}
            for key in d1:
                if key != "loss":
                    out[key] = d1[key]
            for key in d2:
                if key != "loss":
                    out["one2one_" + key] = d2[key]
        else:
            out = self._forward_one(preds, batch, self.assigner)
        # 对齐 ultralytics `return loss.sum() * batch_size`
        out["loss"] = out["loss"] * int(batch[0].shape[0])
        return out

    def _forward_one(self, feats, batch, assigner):
        pred_dist_raw, pred_scores, _ = split_head(
            feats, self.num_classes, self.reg_max, "det", project=False
        )
        pred_distri = (
            dfl_project(pred_dist_raw, self.reg_max)
            if self.reg_max > 1
            else pred_dist_raw
        )
        anchor_points, stride_tensor = make_anchors(feats, self.strides)
        anchors_px = anchor_points * stride_tensor
        pred_bboxes = dist2bbox(pred_distri, anchor_points, xywh=False) * stride_tensor

        targets = batch[1]
        mask_gt = batch[2].unsqueeze(-1).bool()
        device = pred_scores.device
        targets = targets.to(device)
        mask_gt = mask_gt.to(device)
        gt_labels = targets[..., 0:1]
        gt_bboxes = targets[..., 1:5]

        with torch.no_grad():
            t_labels, t_bboxes, t_scores, fg, _ = assigner(
                pred_scores.detach().sigmoid(),
                pred_bboxes.detach(),
                anchors_px,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        scores_sum = max(float(t_scores.sum()), 1.0)
        if self.label_smoothing > 0:
            t_scores = torch.where(
                fg.unsqueeze(-1),
                t_scores * (1 - self.label_smoothing)
                + self.label_smoothing / self.num_classes,
                t_scores,
            )
        loss_cls = self.bce(pred_scores, t_scores)
        if self.class_weights is not None:
            loss_cls = loss_cls * self.class_weights
        loss_cls = loss_cls.sum() / scores_sum

        if bool(fg.any()):
            weight = t_scores.sum(-1)[fg].detach()
            loss_box = (bbox_ciou(pred_bboxes[fg], t_bboxes[fg]) * weight).sum() / scores_sum
            if self.reg_max > 1:
                b_sz, a_sz = fg.shape
                pd = pred_dist_raw.view(b_sz, a_sz, 4, self.reg_max)[fg].reshape(
                    -1, self.reg_max
                )
                stride_fg = stride_tensor.squeeze(-1).unsqueeze(0).expand_as(fg)[fg]
                tgt = (t_bboxes[fg] / stride_fg[:, None]).clamp(
                    0, self.reg_max - 1.0 - 1e-3
                )
                ap_fg = anchor_points.unsqueeze(0).expand(fg.shape[0], -1, -1)[fg]
                target_ltrb = bbox2dist(ap_fg, tgt).clamp(
                    0, self.reg_max - 1.0 - 0.01
                )
                w_dfl = weight
                per_side = df_loss(pd, target_ltrb.reshape(-1)).view(-1, 4).mean(-1)
                loss_dfl = (per_side * w_dfl).sum() / scores_sum
            else:
                loss_dfl = pred_dist_raw.sum() * 0.0
        else:
            loss_box = pred_bboxes.sum() * 0.0
            loss_dfl = pred_dist_raw.sum() * 0.0

        loss = (
            self.box_gain * loss_box
            + self.cls_gain * loss_cls
            + self.dfl_gain * loss_dfl
        )
        return {
            "loss": loss,
            "loss_box": loss_box.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_dfl": loss_dfl.detach(),
        }


class ObbLoss(DetLoss):
    """Oriented-box detection loss: TAL + BCE + ``1 - ProbIoU``."""

    def __init__(
        self,
        num_classes=15,
        strides=(8, 16, 32),
        topk=13,
        alpha=1.0,
        beta=6.0,
        cls_gain=0.5,
        box_gain=7.5,
        dfl_gain=1.5,
        reg_max=1,
        reg_layout="ltrb_angle",
        ne=1,
        angle_gain=1.0,
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
        self.assigner = TaskAlignedAssigner(
            topk,
            self.num_classes,
            alpha,
            beta,
            stride=self.strides,
            iou_fn=probiou,
            inside_fn=points_in_rboxes,
        )
        self.ne = int(ne)
        self.reg_layout = reg_layout
        self.dfl_gain = float(dfl_gain)
        self.angle_gain = float(angle_gain)

    def _split(self, feats):
        mode = "obb" if self.reg_layout == "upstream" else "obb_ours"
        dist, cls, _ = split_head(feats, self.num_classes, self.reg_max, mode, self.ne)
        return dist, cls

    def _split_raw(self, feats):
        """Return raw (un-projected) reg, cls logits and angle logits.

        Layout (upstream) per level: ``[reg(4*reg_max), cls(nc), angle(ne)]``.
        """
        rc = 4 * self.reg_max
        raw_dists, scores, angles = [], [], []
        for feat in feats:
            b, c, h, w = feat.shape
            f = feat.view(b, c, h * w).permute(0, 2, 1)
            raw_dists.append(f[..., :rc])
            scores.append(f[..., rc : rc + self.num_classes])
            angles.append(f[..., rc + self.num_classes :])
        return torch.cat(raw_dists, 1), torch.cat(scores, 1), torch.cat(angles, 1)

    def forward(self, preds, batch):
        dist_raw, pred_scores, angle_raw = self._split_raw(preds)
        anchor_points, stride_tensor = make_anchors(preds, self.strides)
        pred_distri = (
            dfl_project(dist_raw, self.reg_max) if self.reg_max > 1 else dist_raw
        )
        pred_bboxes = dist2rbox(
            torch.cat([pred_distri, angle_raw], dim=-1), anchor_points
        )  # grid-unit xywhr

        device = pred_scores.device
        targets = batch[1].to(device)
        mask_gt = batch[2].unsqueeze(-1).bool().to(device)
        gt_labels = targets[..., 0:1]
        gt_rboxes = targets[..., 1:6]

        with torch.no_grad():
            # scale pred boxes to pixels for matching (matches ultralytics)
            bboxes_for_assigner = pred_bboxes.clone().detach()
            bboxes_for_assigner[..., :4] *= stride_tensor
            t_labels, t_bboxes, t_scores, fg, _ = self.assigner(
                pred_scores.detach().sigmoid(),
                bboxes_for_assigner,
                anchor_points * stride_tensor,
                gt_labels,
                gt_rboxes,
                mask_gt,
            )

        scores_sum = max(float(t_scores.sum()), 1.0)
        loss_cls = self.bce(pred_scores, t_scores)
        if self.class_weights is not None:
            loss_cls = loss_cls * self.class_weights
        loss_cls = loss_cls.sum() / scores_sum

        if bool(fg.any()):
            t_bboxes = t_bboxes.clone()
            t_bboxes[..., :4] = t_bboxes[..., :4] / stride_tensor  # back to grid
            weight = t_scores.sum(-1)[fg].detach()
            iou = probiou(pred_bboxes[fg], t_bboxes[fg], floor=0.01)
            loss_box = ((1.0 - iou) * weight).sum() / scores_sum

            # DFL (RotatedBboxLoss)
            if self.reg_max > 1:
                target_ltrb = rbox2dist(
                    t_bboxes[..., :4].contiguous(),
                    anchor_points,
                    t_bboxes[..., 4:5],
                    reg_max=self.reg_max - 1,
                )
                b_sz, a_sz = fg.shape
                pd = dist_raw.view(b_sz, a_sz, 4, self.reg_max)[fg].reshape(-1, self.reg_max)
                per_side = df_loss(pd, target_ltrb[fg].reshape(-1)).view(-1, 4).mean(-1)
                loss_dfl = (per_side * weight).sum() / scores_sum
            else:
                loss_dfl = dist_raw.sum() * 0.0

            # angle loss (v8OBBLoss.calculate_angle_loss)
            w_gt = t_bboxes[..., 2]
            h_gt = t_bboxes[..., 3]
            log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
            scale_weight = torch.exp(-(log_ar ** 2) / (3.0 ** 2))
            delta = pred_bboxes[..., 4] - t_bboxes[..., 4]
            delta_wrapped = delta - torch.round(delta / math.pi) * math.pi
            ang_loss = (torch.sin(2 * delta_wrapped[fg]) ** 2) * scale_weight[fg] * weight
            loss_angle = ang_loss.sum() / scores_sum
        else:
            loss_box = pred_bboxes.sum() * 0.0
            loss_dfl = dist_raw.sum() * 0.0
            loss_angle = pred_bboxes.sum() * 0.0

        loss = (
            self.box_gain * loss_box
            + self.cls_gain * loss_cls
            + self.dfl_gain * loss_dfl
            + self.angle_gain * loss_angle
        )
        return {
            "loss": loss * int(batch[0].shape[0]),
            "loss_box": loss_box.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_dfl": loss_dfl.detach(),
            "loss_angle": loss_angle.detach(),
        }
