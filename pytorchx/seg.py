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
    """Box mAP (xyxy) + mask mAP, both 101-point interpolated.

    流式实现:每张图匹配后立即释放掩码,只累积 (score, tp) 计数,
    避免在 conf 较低、检测数较多时把全部 (N,H,W) 掩码驻留内存导致 OOM。
    """

    def __init__(self, iou_thresholds=None, main_indicator="mask_mAP50-95", **kwargs):
        super().__init__(
            iou_thresholds=iou_thresholds, main_indicator=main_indicator, box_format="xyxy"
        )
        self.reset()

    def reset(self):
        super().reset()
        self._acc = {"box": {}, "mask": {}}

    @staticmethod
    def _box_iou_mat(pb, gt):
        """(N,4) vs (M,4) xyxy -> (N,M) IoU matrix."""
        if pb.shape[0] == 0 or gt.shape[0] == 0:
            return np.zeros((pb.shape[0], gt.shape[0]), np.float32)
        a = pb[:, None, :]
        b = gt[None, :, :]
        xx1 = np.maximum(a[:, :, 0], b[:, :, 0]); yy1 = np.maximum(a[:, :, 1], b[:, :, 1])
        xx2 = np.minimum(a[:, :, 2], b[:, :, 2]); yy2 = np.minimum(a[:, :, 3], b[:, :, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = np.maximum(0.0, a[:, :, 2] - a[:, :, 0]) * np.maximum(0.0, a[:, :, 3] - a[:, :, 1])
        area_b = np.maximum(0.0, b[:, :, 2] - b[:, :, 0]) * np.maximum(0.0, b[:, :, 3] - b[:, :, 1])
        return (inter / (area_a + area_b - inter + 1e-9)).astype(np.float32)

    @staticmethod
    def _mask_iou_mat(pm, gm):
        """(N,H,W) vs (M,H,W) bool -> (N,M) mask IoU matrix."""
        if pm.shape[0] == 0 or gm.shape[0] == 0:
            return np.zeros((pm.shape[0], gm.shape[0]), np.float32)
        pf = pm.reshape(pm.shape[0], -1).astype(np.float32)
        gf = gm.reshape(gm.shape[0], -1).astype(np.float32)
        inter = pf @ gf.T
        union = pf.sum(1)[:, None] + gf.sum(1)[None, :] - inter
        return (inter / (union + 1e-9)).astype(np.float32)

    @staticmethod
    def _greedy(iou, order, thr):
        """在未匹配 GT 中贪心匹配,返回逐个预测的 tp 列表(对齐 ultralytics)。"""
        matches = np.zeros(iou.shape[1], dtype=bool)
        t = []
        for j in order:
            row = iou[j]
            if row.size:
                row = row.copy()
                row[matches] = -1.0
                k = int(row.argmax())
            else:
                k = -1
            if k >= 0 and row[k] > thr:
                matches[k] = True
                t.append(1)
            else:
                t.append(0)
        return t

    def __call__(self, post_result, batch):
        targets = batch[1].detach().cpu().numpy()
        valid = batch[2].detach().cpu().numpy().astype(bool)
        gt_arr = batch[3]
        is_inst = torch.is_tensor(gt_arr) and gt_arr.dim() == 3
        inst_np = gt_arr.detach().cpu().numpy() if is_inst else None
        for i, pred in enumerate(post_result):
            g = targets[i][valid[i]].astype(np.float32)  # (N, 1+box_dim)
            H, W = 0, 0
            if is_inst:
                instn = inst_np[i]
                H, W = instn.shape
                n = int(valid[i].sum())
                if n:
                    gm = np.stack([instn == (k + 1) for k in range(n)], axis=0).astype(bool)
                else:
                    gm = np.zeros((0, H, W), dtype=bool)
            elif gt_arr is not None:
                gm_arr = gt_arr[i].detach().cpu().numpy().astype(bool)
                H, W = gm_arr.shape[1:]
                gm = gm_arr[valid[i]]
            else:
                gm = np.zeros((0, H, W), dtype=bool)
            pb = pred["bboxes"].detach().cpu().numpy().astype(np.float32)
            ps = pred["scores"].detach().cpu().numpy().astype(np.float32)
            pl = pred["labels"].detach().cpu().numpy().astype(np.int64)
            pm = pred.get("masks")
            pm_np = None
            if pm is not None and torch.is_tensor(pm):
                pm_np = pm.detach().cpu().numpy().astype(bool)
                if pm_np.shape[0] != pl.shape[0]:
                    pm_np = None
            if pm_np is None and pl.size and pb.shape[0]:
                H, W = pm.shape[-2:]
                pm_np = np.zeros((pb.shape[0], H, W), dtype=bool)
            n_pred = pb.shape[0]
            if n_pred == 0:
                continue
            classes = set(int(x) for x in pl)
            if g.shape[0]:
                classes |= set(int(x) for x in g[:, 0])
            for c in classes:
                sel_p = pl == c
                pb_c = pb[sel_p]; ps_c = ps[sel_p]
                pm_c = pm_np[sel_p] if pm_np is not None and pm_np.shape[0] == pl.shape[0] else None
                sel_gt = g[:, 0] == c if g.shape[0] else np.zeros((0,), dtype=bool)
                gt_c = g[sel_gt][:, 1:5] if g.shape[0] else np.zeros((0, 4), np.float32)
                gm_c = gm[sel_gt] if gm.shape[0] else np.zeros((0,) + gm.shape[1:], bool)
                B = self._box_iou_mat(pb_c, gt_c)
                M = self._mask_iou_mat(pm_c, gm_c) if pm_c is not None else B
                order = np.argsort(-ps_c)
                ngt = int(sel_gt.sum())
                for thr in self.iou_thresholds:
                    eb = self._acc["box"].setdefault(c, {}).setdefault(thr, {"s": [], "t": [], "n": 0})
                    eb["n"] += ngt
                    eb["s"].extend(ps_c.tolist())
                    eb["t"].extend(self._greedy(B, order, thr))
                    em = self._acc["mask"].setdefault(c, {}).setdefault(thr, {"s": [], "t": [], "n": 0})
                    em["n"] += ngt
                    em["s"].extend(ps_c.tolist())
                    em["t"].extend(self._greedy(M, order, thr))

    def _ap(self, kind):
        per_thr = {}
        for thr in self.iou_thresholds:
            aps = []
            for c, d in self._acc[kind].items():
                e = d.get(thr)
                if e is None:
                    continue
                n_gt = e["n"]
                if n_gt == 0:
                    continue
                s = np.array(e["s"], dtype=np.float32)
                t = np.array(e["t"], dtype=np.float32)
                order = np.argsort(-s)
                tp = t[order]
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(1.0 - tp)
                recall = cum_tp / max(n_gt, 1)
                precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
                aps.append(_ap_101(recall, precision))
            per_thr[thr] = float(np.mean(aps)) if aps else 0.0
        return per_thr

    def get_metric(self):
        box = self._ap("box")
        mask = self._ap("mask")
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
