"""Instance segmentation components: mask loss / post-process / mask-mAP metric."""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchx.det.loss import DetLoss, df_loss
from pytorchx.det.metric import DetMetric, _ap_101
from pytorchx.det.ops import (
    bbox2dist,
    bbox_ciou,
    dfl_project,
    dist2bbox,
    make_anchors,
    nms,
    split_head,
)
from pytorchx.det.postprocess import DetPostProcess

__all__ = [
    "SegLoss",
    "SegPostProcess",
    "SegMetric",
    "build_seg_loss",
    "build_seg_postprocess",
    "build_seg_metric",
]




def _crop_mask(masks, boxes):
    """把 loss 限制在 GT 框内(对齐 ultralytics crop_mask)。"""
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def _single_mask_loss(gt_mask, pred, proto, xyxy, area):
    """对齐 ultralytics v8SegmentationLoss.single_mask_loss。

    ``pred_mask = einsum('in,nhw->ihw', coeff, proto)``;BCE **只在 GT 框内**统计,
    按每实例的归一化框面积归一 —— 否则整图平均会把损失稀释到接近 0(掩码学不出来)。
    """
    pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
    loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
    return (_crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()


class SegLoss(DetLoss):
    """Detection loss + mask (BCE) loss on the matched anchors."""

    def __init__(
        self,
        num_classes=80,
        nm=32,
        mask_gain=None,
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
        self.nm = int(nm)
        # 对齐原版:v8SegmentationLoss 里 seg 损失乘 self.hyp.box(=box_gain, 默认 7.5)
        self.mask_gain = (
            float(mask_gain) if mask_gain is not None else float(box_gain)
        )
        self.bce_mask = nn.BCEWithLogitsLoss(reduction="none")

    def _split_full(self, feats):
        dist, cls, coeff = split_head(
            feats, self.num_classes, self.reg_max, "seg", project=False
        )
        return dist, cls, coeff

    def forward(self, preds, batch):
        feats = preds["feats"]
        protos = preds["protos"]
        pred_dist_raw, pred_scores, pred_coeff = self._split_full(feats)
        pred_distri = (
            dfl_project(pred_dist_raw, self.reg_max)
            if self.reg_max > 1
            else pred_dist_raw
        )
        anchor_points, stride_tensor = make_anchors(feats, self.strides)
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
                pd = pred_dist_raw.view(b_sz, a_sz, 4, self.reg_max)[fg].reshape(
                    -1, self.reg_max
                )
                stride_fg = stride_tensor.squeeze(-1).unsqueeze(0).expand_as(fg)[fg]
                tgt = (t_bboxes[fg] / stride_fg[:, None]).clamp(
                    0, self.reg_max - 1.0 - 1e-3
                )
                ap_fg = anchor_points.unsqueeze(0).expand(fg.shape[0], -1, -1)[fg]
                target_ltrb = bbox2dist(ap_fg, tgt).clamp(
                    0, self.reg_max - 1.0 - 1e-3
                )
                w_dfl = weight
                per_side = df_loss(pd, target_ltrb.reshape(-1)).view(-1, 4).mean(-1)
                loss_dfl = (per_side * w_dfl).sum() / scores_sum
            else:
                loss_dfl = pred_dist_raw.sum() * 0.0
        else:
            loss_box = pred_bboxes.sum() * 0.0
            loss_dfl = pred_dist_raw.sum() * 0.0

        # ---- mask loss on the foreground anchors ----
        if bool(fg.any()) and batch[3] is not None:
            b_idx = torch.arange(pred_scores.shape[0], device=device)[:, None].expand_as(fg)[fg]
            mh, mw = protos.shape[-2:]
            imgsz_hw = torch.tensor(
                [feats[0].shape[-2] * self.strides[0], feats[0].shape[-1] * self.strides[0]],
                device=device, dtype=protos.dtype,
            )
            _gt_all = batch[3].to(device)
            loss_mask = 0.0
            for b in range(protos.shape[0]):
                sel = fg[b]
                if not bool(sel.any()):
                    continue
                gidx = t_gt_idx[b][sel]
                gm = _gt_all[b].float()[gidx]
                bb = t_bboxes[b][sel].to(protos.dtype)
                bb_norm = bb / imgsz_hw[[1, 0, 1, 0]]
                area = ((bb_norm[:, 2] - bb_norm[:, 0]) *
                        (bb_norm[:, 3] - bb_norm[:, 1])).clamp(min=1e-6)
                xyxy_m = bb_norm * torch.tensor(
                    [mw, mh, mw, mh], device=device, dtype=protos.dtype
                )
                loss_mask = loss_mask + _single_mask_loss(
                    gm, pred_coeff[b][sel], protos[b], xyxy_m, area
                )
            loss_mask = loss_mask / max(int(round(float(fg.sum()))), 1)
        else:
            loss_mask = pred_coeff.sum() * 0.0

        loss = (
            self.box_gain * loss_box
            + self.cls_gain * loss_cls
            + self.dfl_gain * loss_dfl
            + self.mask_gain * loss_mask
        )
        return {
            "loss": loss * int(batch[0].shape[0]),
            "loss_box": loss_box.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_dfl": loss_dfl.detach(),
            "loss_mask": loss_mask.detach(),
        }


class SegPostProcess(DetPostProcess):
    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.7,
        max_det=300,
        strides=(8, 16, 32),
        nm=32,
        mask_thres=0.5,
        reg_max=1,
        **kwargs
    ):
        super().__init__(
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            strides=strides,
            box_type="xyxy",
            reg_max=reg_max,
        )
        self.nm = int(nm)
        self.mask_thres = float(mask_thres)

    def __call__(self, preds):
        feats = preds["feats"]
        protos = preds["protos"]
        num_classes = feats[0].shape[1] - self.reg_channels - self.nm
        distri, scores, coeffs = split_head(
            feats, num_classes, self.reg_max, "seg"
        )
        coeffs = coeffs
        anchor_points, stride_tensor = make_anchors(feats, self.strides)
        boxes = dist2bbox(distri, anchor_points, xywh=False) * stride_tensor

        h, w = protos.shape[-2:]
        imgsz_h = feats[0].shape[-2] * self.strides[0]
        imgsz_w = feats[0].shape[-1] * self.strides[0]
        proto_flat = protos.flatten(2)  # (B, nm, h*w)
        results = []
        for b in range(boxes.shape[0]):
            bx, sc, cf = boxes[b], scores[b].sigmoid(), coeffs[b]
            conf, labels = sc.max(-1)
            keep = conf > self.conf_thres
            bx, conf, labels, cf = bx[keep], conf[keep], labels[keep], cf[keep]
            if bx.numel() == 0:
                results.append(
                    {
                        "bboxes": bx.reshape(0, 4),
                        "scores": conf,
                        "labels": labels,
                        "masks": torch.zeros((0, imgsz_h, imgsz_w), dtype=torch.bool),
                    }
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
            masks = None
            if kept.numel():
                proto_b = proto_flat[b].unsqueeze(0).expand(kept.numel(), -1, -1)
                m = torch.bmm(cf[kept].unsqueeze(1), proto_b).squeeze(1)
                logits = m.view(-1, h, w)
                # 对齐 ultralytics process_mask(upsample=True):先在 logits 上双线性
                # 上采样到原图分辨率,再阈值 0 二值化,最后按预测框裁剪。
                logits_up = F.interpolate(
                    logits.unsqueeze(1),
                    size=(imgsz_h, imgsz_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                bin_m = (logits_up > 0.0).float()
                masks = _crop_mask(bin_m, bx[kept]) > 0.5
            results.append(
                {
                    "bboxes": bx[kept],
                    "scores": conf[kept],
                    "labels": labels[kept],
                    "masks": masks if masks is not None else torch.zeros((0, imgsz_h, imgsz_w), dtype=torch.bool),
                }
            )
        return results


class SegMetric(DetMetric):
    """Box mAP (xyxy) + mask mAP, both 101-point interpolated."""

    def __init__(self, iou_thresholds=None, main_indicator="mask_mAP50-95", **kwargs):
        super().__init__(
            iou_thresholds=iou_thresholds, main_indicator=main_indicator, box_format="xyxy"
        )
        self.pred_masks = []
        self.gt_masks = []

    def reset(self):
        super().reset()
        self.pred_masks = []
        self.gt_masks = []

    def __call__(self, post_result, batch):
        super().__call__(post_result, batch)
        gt_masks = batch[3]
        mask_valid = batch[2].detach().cpu().numpy().astype(bool)
        idx_map = None
        if torch.is_tensor(gt_masks) and gt_masks.dim() == 3:
            idx_map = gt_masks.detach().cpu().numpy()  # (B,H,W) 实例索引图
        for i, pred in enumerate(post_result):
            m = pred.get("masks")
            if m is not None and torch.is_tensor(m):
                m = m.detach().cpu().numpy().astype(bool)
            self.pred_masks.append(m)
            if gt_masks is None:
                self.gt_masks.append(None)
            elif idx_map is not None:
                inst = idx_map[i]
                n = int(mask_valid[i].sum())
                if n:
                    gms = np.stack([inst == (k + 1) for k in range(n)], axis=0)
                else:
                    gms = np.zeros((0,) + inst.shape, dtype=bool)
                self.gt_masks.append(gms)
            else:
                self.gt_masks.append(
                    gt_masks[i].detach().cpu().numpy().astype(bool)[mask_valid[i]]
                )

    @staticmethod
    def _mask_iou(pred, gts):
        if gts.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        p = pred.reshape(-1)
        inter = (gts.reshape(gts.shape[0], -1) & p).sum(1).astype(np.float64)
        union = (gts.reshape(gts.shape[0], -1) | p).sum(1).astype(np.float64)
        return (inter / (union + 1e-9)).astype(np.float32)

    def _ap_for(self, use_masks):
        num_images = len(self.gts)
        n_classes = 0
        for g in self.gts:
            if g.shape[0]:
                n_classes = max(n_classes, int(g[:, 0].max()) + 1)
        for p in self.preds:
            if p["labels"].size:
                n_classes = max(n_classes, int(p["labels"].max()) + 1)

        per_thr = {}
        for thr in self.iou_thresholds:
            aps = []
            for c in range(n_classes):
                scores, tps, n_gt = [], [], 0
                for i in range(num_images):
                    g = self.gts[i]
                    sel_gt = g[:, 0] == c if g.shape[0] else np.zeros((0,), dtype=bool)
                    gt_c = g[sel_gt][:, 1:5] if g.shape[0] else g.reshape(0, 4)
                    n_gt += gt_c.shape[0]

                    gt_masks_c = None
                    if use_masks:
                        gm = self.gt_masks[i]
                        gt_masks_c = (
                            gm[sel_gt]
                            if gm is not None
                            else np.zeros((0, 1, 1), dtype=bool)
                        )

                    p = self.preds[i]
                    sel_p = p["labels"] == c
                    if not sel_p.any():
                        continue
                    pb, ps = p["bboxes"][sel_p], p["scores"][sel_p]
                    pm = None
                    if use_masks:
                        pm_all = self.pred_masks[i]
                        if pm_all is not None and pm_all.shape[0] == p["labels"].shape[0]:
                            pm = pm_all[sel_p]

                    order = np.argsort(-ps)
                    pb, ps = pb[order], ps[order]
                    if pm is not None:
                        pm = pm[order]

                    matches = np.zeros(gt_c.shape[0], dtype=bool)
                    for j in range(pb.shape[0]):
                        if use_masks:
                            ious = np.array(self._mask_iou(pm[j], gt_masks_c), dtype=np.float32)
                        else:
                            ious = np.array(self._iou(pb[j], gt_c), dtype=np.float32)
                        # 对齐 ultralytics:只在未匹配 GT 中找最优
                        if ious.size:
                            ious[matches] = -1.0
                            k = int(ious.argmax())
                        else:
                            k = -1
                        if k >= 0 and ious[k] > thr:
                            matches[k] = True
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
        return per_thr

    def get_metric(self):
        box = self._ap_for(use_masks=False)
        mask = self._ap_for(use_masks=True)
        metrics = {
            "box_mAP50": box.get(0.5, 0.0),
            "box_mAP50-95": float(np.mean(list(box.values()))) if box else 0.0,
            "mask_mAP50": mask.get(0.5, 0.0),
            "mask_mAP50-95": float(np.mean(list(mask.values()))) if mask else 0.0,
        }
        return metrics


def build_seg_loss(loss_cfg, num_classes, nm=None, reg_max=None):
    cfg = dict(loss_cfg or {})
    name = cfg.pop("name", "SegLoss")
    if name not in ("SegLoss", "SegmentationLoss"):
        raise ValueError("Unknown seg loss: {}".format(name))
    if nm is not None and "nm" not in cfg:
        cfg["nm"] = nm
    if reg_max is not None and "reg_max" not in cfg:
        cfg["reg_max"] = reg_max
    return SegLoss(num_classes=num_classes, **cfg)


def build_seg_postprocess(pp_cfg, nm=None, reg_max=None):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    if nm is not None and "nm" not in cfg:
        cfg["nm"] = nm
    if reg_max is not None and "reg_max" not in cfg:
        cfg["reg_max"] = reg_max
    return SegPostProcess(**cfg)


def build_seg_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    return SegMetric(**cfg)
