"""Plate detection task: yolov5-face style loss / post-process / metric / adapter.

Faithful port of upstream (``Chinese_license_plate_detection_recognition``):

* ``utils/loss.py::compute_loss`` + ``build_targets`` (anchor matching with
  ``anchor_t``, BCE obj/cls, CIoU box, ``WingLoss`` on the 4 plate corners),
* ``models/yolo.py::Detect`` decode (see :class:`pytorchx.nn.plate.PlateDetect`),
* hyper parameters from ``data/hyp.scratch.yaml``
  (``box 0.05 / cls 0.5 / obj 1.0 / landmark 0.005 / cls_pw 1.0 / obj_pw 1.0 /
  anchor_t 4.0 / gr 1.0``, ``balance [4.0, 1.0, 0.4]``).
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from pytorchx.det.metric import DetMetric
from pytorchx.det.ops import bbox_ciou, nms

__all__ = [
    "PlateDetLoss",
    "PlateDetPostProcess",
    "PlateDetMetric",
    "PlateDetTask",
    "build_plate_det_loss",
    "build_plate_det_postprocess",
    "build_plate_det_metric",
]


class WingLoss(nn.Module):
    """Wing loss used for the plate corners (upstream ``LandmarksLoss``)."""

    def __init__(self, w=10.0, e=2.0):
        super().__init__()
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1.0):
        weight = torch.ones_like(t)
        weight[torch.where(t == -1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (
            abs_diff - self.C
        )
        return y.sum()


class LandmarksLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.loss_fcn = WingLoss()
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred * mask, truel * mask)
        return loss / (torch.sum(mask) + 10e-14)


class PlateDetLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        anchors,
        strides=(8, 16, 32),
        kpt_label=4,
        box=0.05,
        obj=1.0,
        cls=0.5,
        landmark=0.005,
        cls_pw=1.0,
        obj_pw=1.0,
        anchor_t=4.0,
        gr=1.0,
        fl_gamma=0.0,
        label_smoothing=0.0,
        balance=(4.0, 1.0, 0.4),
        **kwargs
    ):
        super().__init__()
        self.nc = int(num_classes)
        self.kpt_label = int(kpt_label)
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.register_buffer(
            "stride", torch.tensor([float(s) for s in strides]).view(-1, 1, 1)
        )
        self.register_buffer(
            "anchors",
            torch.tensor(anchors).float().view(self.nl, -1, 2) / self.stride,
        )
        self.box = float(box)
        self.obj = float(obj)
        self.cls = float(cls)
        self.landmark = float(landmark)
        self.cls_pw = float(cls_pw)
        self.obj_pw = float(obj_pw)
        self.anchor_t = float(anchor_t)
        self.gr = float(gr)
        self.fl_gamma = float(fl_gamma)
        self.label_smoothing = float(label_smoothing or 0.0)
        self.balance = list(balance)
        self.landmarks_loss = LandmarksLoss(1.0)
        self._landmarks = LandmarksLoss(1.0)

    # ------------------------------------------------------------------ #
    def _targets_from_batch(self, batch, device):
        boxes = batch[1].to(device).float()  # (B, max, 5) cls,x1,y1,x2,y2
        valid = batch[2].to(device)
        kpts = batch[3].to(device).float()  # (B, max, k, dim)
        nl = kpts.shape[2]
        keep = valid.bool()
        b_idx, l_idx = keep.nonzero(as_tuple=True)
        if b_idx.numel():
            bb = boxes[b_idx, l_idx]
            kk = kpts[b_idx, l_idx][..., :2].reshape(-1, 2 * nl)
            cx = (bb[:, 1] + bb[:, 3]) * 0.5
            cy = (bb[:, 2] + bb[:, 4]) * 0.5
            w = bb[:, 3] - bb[:, 1]
            h = bb[:, 4] - bb[:, 2]
            targets = torch.cat(
                [
                    b_idx.float()[:, None],
                    bb[:, 0:1],
                    cx[:, None],
                    cy[:, None],
                    w[:, None],
                    h[:, None],
                    kk,
                ],
                1,
            )
        else:
            targets = torch.zeros((0, 6 + 2 * nl), device=device)
        return targets

    def build_targets(self, p, batch):
        device = p[0].device
        targets = self._targets_from_batch(batch, device)
        na, nt = self.na, targets.shape[0]
        nl = self.kpt_label
        tcls, tbox, indices, anch, landmarks, lmks_mask = [], [], [], [], [], []
        gain = torch.ones(7 + 2 * nl, device=device)  # +1 = anchor index
        ai = (
            torch.arange(na, device=device)
            .float()
            .view(na, 1)
            .repeat(1, nt)
        )
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        g = 0.5
        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=device
        ).float() * g

        for i in range(self.nl):
            anchors, shape = self.anchors[i].to(device), p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]].to(device)
            gain[6 : 6 + 2 * nl] = torch.tensor(shape)[[3, 2] * nl].to(device)
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                jj, kk = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                ll, mm = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                jj = torch.stack((torch.ones_like(jj), jj, kk, ll, mm))
                t = t.repeat((5, 1, 1))[jj]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[jj]
            else:
                t = targets[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, -1].long()
            indices.append(
                (b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1))
            )
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
            lks = t[:, 6 : 6 + 2 * nl].clone()
            lks_mask = torch.where(
                lks < 0, torch.full_like(lks, 0.0), torch.full_like(lks, 1.0)
            )
            for k in range(nl):
                lks[:, 2 * k : 2 * k + 2] = lks[:, 2 * k : 2 * k + 2] - gij
            lmks_mask.append(lks_mask)
            landmarks.append(lks)
        return tcls, tbox, indices, anch, landmarks, lmks_mask

    # ------------------------------------------------------------------ #
    def forward(self, preds, batch):
        if isinstance(preds, dict):
            preds = preds.get("feats", preds)
        preds = list(preds)
        # heads may return `(decoded_concat, per_level_maps)` when in eval mode
        if (
            len(preds) == 2
            and torch.is_tensor(preds[0])
            and isinstance(preds[1], (list, tuple))
            and preds[1]
            and torch.is_tensor(preds[1][0])
        ):
            preds = list(preds[1])
        device = preds[0].device
        tcls, tbox, indices, anchors, tlandmarks, lmks_mask = self.build_targets(
            preds, batch
        )
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lmark = torch.zeros(1, device=device)

        eps = self.label_smoothing
        cp, cn = 1.0 - 0.5 * eps, 0.5 * eps
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.cls_pw], device=device)
        )
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.obj_pw], device=device)
        )
        no = len(preds)
        balance = self.balance[:no]
        cls_start = 5 + 2 * self.kpt_label

        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_ciou(pbox, tbox[i])
                lbox += (1.0 - iou).mean()
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(
                    0
                ).type(tobj.dtype)
                if self.nc > 1:
                    t = torch.full_like(ps[:, cls_start:], cn, device=device)
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, cls_start:], t)
                pl = ps[:, 5:cls_start].clone()
                for k in range(self.kpt_label):
                    pl[:, 2 * k : 2 * k + 2] = (
                        pl[:, 2 * k : 2 * k + 2] * anchors[i]
                    )
                lmark += self._landmarks(pl, tlandmarks[i], lmks_mask[i])
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]

        s = 3.0 / no
        lbox = lbox * self.box * s
        lobj = lobj * self.obj * s * (1.4 if no == 4 else 1.0)
        lcls = lcls * self.cls * s
        lmark = lmark * self.landmark * s
        bs = preds[0].shape[0]
        loss = lbox + lobj + lcls + lmark
        return {
            "loss": loss * bs,
            "loss_box": (lbox * bs).detach(),
            "loss_obj": (lobj * bs).detach(),
            "loss_cls": (lcls * bs).detach(),
            "loss_landmark": (lmark * bs).detach(),
        }


class PlateDetPostProcess(object):
    """Anchor-head decode + per-class NMS (predictions stay in letterbox coords)."""

    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        num_classes=2,
        kpt_label=4,
        **kwargs
    ):
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.nc = int(num_classes)
        self.kpt_label = int(kpt_label)

    def __call__(self, preds):
        cat = preds[0] if isinstance(preds, (tuple, list)) else preds
        if isinstance(cat, dict):
            cat = cat.get("preds")
        kp = self.kpt_label
        cls_start = 5 + 2 * kp
        results = []
        for b in range(cat.shape[0]):
            p = cat[b]
            obj = p[:, 4]
            cls_scores = p[:, cls_start : cls_start + self.nc]
            conf, labels = cls_scores.max(-1)
            conf = conf * obj
            keep = conf > self.conf_thres
            p, conf, labels = p[keep], conf[keep], labels[keep]
            if p.numel() == 0:
                results.append(
                    {
                        "bboxes": torch.zeros((0, 4)),
                        "scores": torch.zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.long),
                        "kpts": torch.zeros((0, kp, 2)),
                    }
                )
                continue
            xy, wh = p[:, 0:2], p[:, 2:4]
            xyxy = torch.cat([xy - wh / 2, xy + wh / 2], 1)
            kpts = p[:, 5:cls_start].reshape(-1, kp, 2)
            kept = []
            for c in labels.unique():
                idx = (labels == c).nonzero(as_tuple=True)[0]
                sel = nms(xyxy[idx], conf[idx], self.iou_thres)
                if torch.is_tensor(sel):
                    sel = sel.to(device=idx.device, dtype=torch.long)
                else:
                    sel = torch.as_tensor(
                        np.asarray(sel), dtype=torch.long, device=idx.device
                    )
                if sel.numel():
                    kept.append(idx[sel])
            kept = (
                torch.cat(kept)
                if kept
                else torch.zeros(0, dtype=torch.long, device=xyxy.device)
            )
            if kept.numel() > self.max_det:
                order = conf[kept].argsort(descending=True)[: self.max_det]
                kept = kept[order]
            results.append(
                {
                    "bboxes": xyxy[kept].detach(),
                    "scores": conf[kept].detach(),
                    "labels": labels[kept].detach(),
                    "kpts": kpts[kept].detach(),
                }
            )
        return results


class PlateDetMetric(DetMetric):
    """Box mAP (from :class:`DetMetric`) plus plate-corner error (NME / pixels)."""

    def __init__(self, main_indicator="mAP50-95", **kwargs):
        super().__init__(main_indicator=main_indicator, **kwargs)
        self._kpt_preds = []
        self._kpt_gts = []

    def reset(self):
        super().reset()
        self._kpt_preds = []
        self._kpt_gts = []

    def __call__(self, post_result, batch):
        super().__call__(post_result, batch)
        kpts = batch[3].detach().cpu().numpy()
        valid = batch[2].detach().cpu().numpy().astype(bool)
        for i, pred in enumerate(post_result):
            self._kpt_preds.append(
                {
                    "bboxes": pred["bboxes"].detach().cpu().numpy().astype(np.float32),
                    "labels": pred["labels"].detach().cpu().numpy().astype(np.int64),
                    "kpts": pred["kpts"].detach().cpu().numpy().astype(np.float32),
                }
            )
            self._kpt_gts.append(
                {
                    "bboxes": batch[1][i].detach().cpu().numpy()[valid[i]],
                    "kpts": kpts[i][valid[i]],
                }
            )

    @staticmethod
    def _iou_pair(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def get_metric(self):
        metrics = super().get_metric()
        errs, px = [], []
        for pred, gt in zip(self._kpt_preds, self._kpt_gts):
            for j in range(gt["bboxes"].shape[0]):
                gb = gt["bboxes"][j][1:5]
                diag = float(np.hypot(gb[2] - gb[0], gb[3] - gb[1])) + 1e-9
                best, bi = 0.0, -1
                for k in range(pred["bboxes"].shape[0]):
                    iou = self._iou_pair(pred["bboxes"][k], gb)
                    if iou > best:
                        best, bi = iou, k
                if bi >= 0 and best >= 0.5:
                    d = np.linalg.norm(
                        pred["kpts"][bi][:, :2] - gt["kpts"][j][:, :2], axis=1
                    )
                    errs.append(float(d.mean()) / diag)
                    px.append(float(d.mean()))
        if errs:
            metrics["kpt_nme"] = float(np.mean(errs))
            metrics["kpt_px"] = float(np.mean(px))
        return metrics


def _head_of(model):
    """Locate the landmark head on a graph model (``model.model[-1]``) or a wrapper."""
    if model is None:
        return None
    for attr in ("head", "_heads"):
        h = getattr(model, attr, None)
        if isinstance(h, nn.Module) and hasattr(h, "anchors"):
            return h
        if isinstance(h, (list, tuple)) and h and hasattr(h[0], "anchors"):
            return h[0]
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        try:
            last = list(inner)[-1]
        except Exception:
            last = None
        if last is not None and hasattr(last, "anchors"):
            return last
    return None


def _anchor_kwargs(config, model=None):
    arch = (config or {}).get("Architecture") or {}
    head = arch.get("Head") or {}
    anchors = head.get("anchors")
    kpt_label = int(head.get("kpt_label", 4))
    nc = int(head.get("num_classes", 2))
    h = _head_of(model)
    if h is not None:
        pairs = (h.anchors.cpu() * h.stride.cpu().view(-1, 1, 1)).tolist()
        # flat [w1,h1,w2,h2,...] per level (YAML / upstream convention)
        anchors = [[c for pair in lvl for c in pair] for lvl in pairs]
        kpt_label = int(getattr(h, "kpt_label", kpt_label))
        nc = int(getattr(h, "nc", nc))
    return nc, anchors, kpt_label


def build_plate_det_loss(loss_cfg, model=None, num_classes=2, anchors=None, kpt_label=4):
    cfg = dict(loss_cfg or {})
    cfg.pop("name", None)
    cfg.pop("strides", None)
    strides = cfg.pop("strides", (8, 16, 32))
    nc, anchors, kpt_label = _anchor_kwargs(
        {"Architecture": {"Head": {"num_classes": num_classes, "anchors": anchors, "kpt_label": kpt_label}}},
        model,
    )
    if cfg.get("anchors"):
        anchors = cfg.pop("anchors")
    return PlateDetLoss(
        num_classes=nc,
        anchors=anchors,
        strides=strides,
        kpt_label=kpt_label,
        **cfg
    )


def build_plate_det_postprocess(pp_cfg, model=None, num_classes=2, kpt_label=4):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    cfg.pop("strides", None)
    cfg.setdefault("num_classes", num_classes)
    cfg.setdefault("kpt_label", kpt_label)
    return PlateDetPostProcess(**cfg)


def build_plate_det_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    cfg.pop("strides", None)
    return PlateDetMetric(**cfg)


class PlateDetTask(TaskAdapter):
    """Licence-plate detection (``Architecture.task: plate_det``)."""

    name = "plate_det"

    def _head(self, config, model=None):
        arch = config.get("Architecture") or {}
        return arch.get("Head") or {}

    def build_post_process(self, config):
        return build_plate_det_postprocess(
            config.get("PostProcess"),
            num_classes=int(self._head(config).get("num_classes", 2)),
            kpt_label=int(self._head(config).get("kpt_label", 4)),
        )

    def build_model(self, config, post_process):
        from pytorchx.models import build_arch_model

        return build_arch_model(config["Architecture"], "plate_det")

    def build_loss(self, config, model):
        return build_plate_det_loss(config.get("Loss"), model=model)

    def build_metric(self, config):
        return build_plate_det_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from pytorchx.data.pose import PoseDataset

        train_ds = PoseDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = PoseDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from pytorchx.data.pose import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from pytorchx.data.pose import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        metric(post_process(model(images)), batch)

    def summary_lines(self, config, global_config, post_process):
        head = self._head(config)
        ds = config.get("Train", {}).get("dataset") or {}
        return [
            "task=plate_det yaml={} num_classes={} kpt_label={} imgsz={}".format(
                (config.get("Architecture") or {}).get("yaml_file"),
                head.get("num_classes"),
                head.get("kpt_label", 4),
                (ds.get("transform") or {}).get("image_size"),
            )
        ]
