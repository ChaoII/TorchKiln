"""3D detection components (CenterPoint-style): loss / metric / postprocess.

v0 scope (per AUTONOMOUS_DRIVING §4):
- heatmap focal classification + L1 box regression at object centers
- BEV rotated IoU AP (probiou) + BEV NMS (reuses ``det.rbox``)
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Det3DLoss",
    "Det3DMetric",
    "Det3DPostProcess",
    "build_det3d_loss",
    "build_det3d_metric",
    "build_det3d_postprocess",
    "gaussian_radius",
    "draw_gaussian",
]


def gaussian_radius(sz, min_overlap=0.7):
    """CenterNet gaussian radius for a square of size ``sz`` (approx)."""
    return max(int(sz), 1)


def draw_gaussian(heatmap, center, radius, k=1.0):
    """Draw a 2D gaussian blob on ``heatmap`` (H,W) numpy float array."""
    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ys = np.arange(radius * 2 + 1) - radius
    xs = np.arange(radius * 2 + 1) - radius
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    left, right = min(x, radius), min(w - 1 - x, radius)
    top, bottom = min(y, radius), min(h - 1 - y, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    hm_reg = heatmap[
        y - top : y + bottom + 1, x - left : x + right + 1
    ]
    g_reg = g[
        radius - top : radius + bottom + 1, radius - left : radius + right + 1
    ]
    np.maximum(hm_reg, g_reg * k, out=hm_reg)


class Det3DLoss(nn.Module):
    """Heatmap focal + L1 regression on CenterPoint-style heads.

    ``preds``: dict with ``heatmap (B,nc,H,W)`` and ``reg (B,8,H,W)``
    (or a tuple/list ``(hm, reg)``).
    ``batch``: ``[bev, targets(B,M,8), mask(B,M)]`` where targets are
    ``[cls, x, y, z, l, w, h, yaw]`` in LiDAR frame.
    """

    def __init__(
        self,
        num_classes=3,
        pc_range=None,
        pillar_size=None,
        focal_alpha=2.0,
        focal_beta=4.0,
        box_weight=0.25,
        hm_weight=1.0,
        gaussian_radius=2,
        **kwargs,
    ):
        super().__init__()
        self.nc = int(num_classes)
        self.focal_alpha = float(focal_alpha)
        self.focal_beta = float(focal_beta)
        self.box_weight = float(box_weight)
        self.hm_weight = float(hm_weight)
        self.gauss_r = int(gaussian_radius)
        if pc_range is None:
            pc_range = [-16, -16, -3, 16, 16, 3]
        if isinstance(pc_range[0], (list, tuple)):
            (x0, x1), (y0, y1), (z0, z1) = pc_range
            self.xmin, self.xmax = float(x0), float(x1)
            self.ymin, self.ymax = float(y0), float(y1)
            self.zmin, self.zmax = float(z0), float(z1)
        else:
            v = [float(t) for t in pc_range]
            self.xmin, self.xmax = v[0], v[3]
            self.ymin, self.ymax = v[1], v[4]
            self.zmin, self.zmax = v[2], v[5]
        if pillar_size is None:
            pillar_size = [0.5, 0.5, 4.0]
        ps = list(pillar_size) if isinstance(pillar_size, (list, tuple)) else [pillar_size] * 2
        if len(ps) == 2:
            ps.append(4.0)
        self.px, self.py = float(ps[0]), float(ps[1])

    @staticmethod
    def _split(preds):
        if isinstance(preds, dict):
            return preds["heatmap"], preds["reg"]
        if isinstance(preds, (list, tuple)) and len(preds) >= 2:
            return preds[0], preds[1]
        raise TypeError("Det3DLoss expects dict/tuple (heatmap, reg), got %r" % type(preds))

    def _make_targets(self, targets, mask, hm_size):
        """Build heatmap / reg targets for a batch."""
        b, ny, nx = hm_size
        hm = torch.zeros(b, self.nc, ny, nx, device=targets.device, dtype=torch.float32)
        reg = torch.zeros(b, 8, ny, nx, device=targets.device, dtype=torch.float32)
        pos = torch.zeros(b, ny, nx, device=targets.device, dtype=torch.float32)
        for i in range(b):
            m = mask[i] > 0.5
            boxes = targets[i][m]
            for box in boxes:
                cls = int(box[0].item())
                if cls < 0 or cls >= self.nc:
                    continue
                x, y = float(box[1]), float(box[2])
                z = float(box[3])
                l, w, h = float(box[4]), float(box[5]), float(box[6])
                yaw = float(box[7])
                cx = (x - self.xmin) / self.px
                cy = (y - self.ymin) / self.py
                ix, iy = int(cx), int(cy)
                if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
                    continue
                # draw on a CPU clone then write back (device-safe)
                blob = np.zeros((ny, nx), dtype=np.float32)
                draw_gaussian(blob, (ix, iy), self.gauss_r)
                hm[i, cls] = torch.maximum(
                    hm[i, cls], torch.from_numpy(blob).to(hm.device)
                )
                dx = cx - ix
                dy = cy - iy
                reg[i, 0, iy, ix] = dx
                reg[i, 1, iy, ix] = dy
                reg[i, 2, iy, ix] = z
                reg[i, 3, iy, ix] = np.log(max(l, 1e-3))
                reg[i, 4, iy, ix] = np.log(max(w, 1e-3))
                reg[i, 5, iy, ix] = np.log(max(h, 1e-3))
                reg[i, 6, iy, ix] = np.sin(yaw)
                reg[i, 7, iy, ix] = np.cos(yaw)
                pos[i, iy, ix] = 1.0
        return hm, reg, pos

    def forward(self, preds, batch):
        hm_pred, reg_pred = self._split(preds)
        targets = batch[1]
        mask = batch[2]
        ny, nx = hm_pred.shape[-2:]
        hm_t, reg_t, pos = self._make_targets(targets, mask, (hm_pred.shape[0], ny, nx))

        # CenterNet focal loss on sigmoid heatmap. Normalize by the number of
        # positive centres (not by all cells) — otherwise the loss is diluted by
        # ~H*W and the heatmap head barely learns.
        pred = hm_pred.sigmoid()
        pos_mask = pos.unsqueeze(1).expand_as(hm_t)
        neg_weight = torch.pow(1.0 - hm_t, self.focal_beta)
        pos_loss = -torch.log(pred.clamp(min=1e-6)) * torch.pow(1.0 - pred, self.focal_alpha)
        neg_loss = -torch.log((1.0 - pred).clamp(min=1e-6)) * torch.pow(
            pred, self.focal_alpha
        ) * neg_weight
        n_pos = pos_mask.sum().clamp(min=1.0)
        loss_hm = (pos_loss * pos_mask + neg_loss * (1.0 - pos_mask)).sum() / n_pos
        loss_hm = loss_hm * self.hm_weight

        # L1 on 8-dof at positive cells only
        pos_e = pos.unsqueeze(1)
        n_pos = pos_e.sum().clamp(min=1.0)
        loss_box = ((reg_pred - reg_t).abs() * pos_e).sum() / n_pos

        loss = loss_hm + self.box_weight * loss_box
        return {
            "loss": loss,
            "loss_cls": loss_hm.detach(),
            "loss_box": loss_box.detach(),
        }


class Det3DPostProcess(object):
    """Peak-extract heatmap + decode boxes + BEV rotated NMS."""

    def __init__(
        self,
        score_thres=0.1,
        nms_thres=0.2,
        max_det=100,
        pc_range=None,
        pillar_size=None,
        **kwargs,
    ):
        self.score_thres = float(score_thres)
        self.nms_thres = float(nms_thres)
        self.max_det = int(max_det)
        # lazy import of range parse to avoid cycle
        from torchkiln.data.pc import _as_range, _as_pillar

        xr = _as_range(pc_range)
        self.xmin, self.xmax = xr[0], xr[1]
        self.ymin, self.ymax = xr[2], xr[3]
        self.zmin, self.zmax = xr[4], xr[5]
        ps = _as_pillar(pillar_size)
        self.px, self.py = ps[0], ps[1]

    @staticmethod
    def _split(preds):
        if isinstance(preds, dict):
            return preds["heatmap"], preds["reg"]
        if isinstance(preds, (list, tuple)) and len(preds) >= 2:
            return preds[0], preds[1]
        raise TypeError("Det3DPostProcess expects dict/tuple (heatmap, reg)")

    def __call__(self, preds, **kwargs):
        hm, reg = self._split(preds)
        if torch.is_tensor(hm):
            hm = hm.detach()
            reg = reg.detach()
        scores = hm.sigmoid()
        b, nc, ny, nx = scores.shape
        # 3×3 max-pool peak extraction
        pooled = F.max_pool2d(scores, kernel_size=3, stride=1, padding=1)
        peak = (scores == pooled) & (scores >= self.score_thres)

        outs = []
        for i in range(b):
            # nonzero on (C,H,W) -> (channel, y, x)
            cs, ys, xs = torch.nonzero(peak[i], as_tuple=True)
            sc = scores[i, cs, ys, xs]
            order = torch.argsort(sc, descending=True)[: self.max_det]
            ys, xs, cs, sc = ys[order], xs[order], cs[order], sc[order]
            if sc.numel() == 0:
                outs.append(
                    {
                        "bboxes": np.zeros((0, 5), dtype=np.float32),
                        "scores": np.zeros((0,), dtype=np.float32),
                        "labels": np.zeros((0,), dtype=np.int64),
                    }
                )
                continue
            r = reg[i, :, ys, xs]  # (N, 8)
            dx, dy = r[0], r[1]
            x = (xs.float() + dx) * self.px + self.xmin
            y = (ys.float() + dy) * self.py + self.ymin
            z = r[2]
            l = r[3].exp().clamp(0.1, 50.0)
            w = r[4].exp().clamp(0.1, 50.0)
            h = r[5].exp().clamp(0.1, 50.0)
            yaw = torch.atan2(r[6], r[7])
            # BEV xywhr: (cx, cy, length, width, yaw) — length along yaw
            boxes = torch.stack([x, y, l, w, yaw], dim=1)  # (N,5)
            labels = cs.cpu().numpy().astype(np.int64)
            scores_np = sc.cpu().numpy().astype(np.float32)

            # BEV rotated NMS
            from torchkiln.det.rbox import nms_rotated

            keep = nms_rotated(
                boxes, torch.from_numpy(scores_np).to(boxes.device), self.nms_thres
            )
            keep = keep.cpu().numpy()
            boxes_np = boxes[keep].cpu().numpy().astype(np.float32)
            outs.append(
                {
                    "bboxes": boxes_np,
                    "scores": scores_np[keep],
                    "labels": labels[keep],
                }
            )
        return outs


class Det3DMetric(object):
    """BEV rotated-IoU AP (probiou); reuses 2D DetMetric AP bookkeeping."""

    def __init__(self, iou_thresholds=None, main_indicator="mAP", names=None, **kwargs):
        from torchkiln.det.metric import DetMetric

        thr = list(iou_thresholds) if iou_thresholds else [0.5, 0.7]
        self._inner = DetMetric(
            iou_thresholds=thr, main_indicator=main_indicator, box_format="xywhr"
        )
        self.main_indicator = main_indicator
        self.names = names

    def reset(self):
        self._inner.reset()

    def __call__(self, post_result, batch):
        # GT batch[1] is (B,M,8) [cls,x,y,z,l,w,h,yaw]; DetMetric wants
        # (B,M,1+dim) with dim=5 for xywhr: [cls,x,y,l,w,yaw] (drop z,h).
        targets = batch[1]
        mask = batch[2]
        if torch.is_tensor(targets):
            t8 = targets.detach().cpu().numpy()
            m = mask.detach().cpu().numpy()
        else:
            t8, m = np.asarray(targets), np.asarray(mask)
        bev = np.concatenate(
            [t8[..., 0:1], t8[..., 1:3], t8[..., 4:5], t8[..., 5:6], t8[..., 7:8]],
            axis=-1,
        )  # (B,M,6) [cls,x,y,l,w,yaw]
        # DetMetric expects torch tensors for preds (bboxes/scores/labels)
        # and GT (targets/mask) at indices 1/2.
        pred_t = []
        for p in post_result:
            pred_t.append(
                {
                    "bboxes": torch.as_tensor(np.asarray(p["bboxes"]), dtype=torch.float32),
                    "scores": torch.as_tensor(np.asarray(p["scores"]), dtype=torch.float32),
                    "labels": torch.as_tensor(np.asarray(p["labels"]), dtype=torch.int64),
                }
            )
        batch_bev = [
            None,
            torch.as_tensor(bev, dtype=torch.float32),
            torch.as_tensor(m.astype(np.float32), dtype=torch.float32),
        ]
        self._inner(pred_t, batch_bev)

    def get_metric(self):
        raw = self._inner.get_metric()
        # rename to design keys: mAP50 / mAP
        m = {
            "mAP": raw.get("mAP50-95", 0.0),
            "mAP50": raw.get("mAP50", 0.0),
            "mAP70": raw.get("mAP75", 0.0) if 0.75 in (0.5, 0.7) else raw.get("mAP75", 0.0),
        }
        # prefer threshold 0.7 if present under mAP75-like keys; DetMetric stores
        # mAP50 / mAP50-95 / mAP75 — for thr=[0.5,0.7] mAP50-95 is mean of both.
        if abs(m["mAP"] - raw.get("mAP50-95", 0.0)) < 1e-12:
            # expose mean as mAP (design main_indicator)
            pass
        return m


def build_det3d_loss(loss_cfg, num_classes=None, pc_range=None, pillar_size=None):
    cfg = dict(loss_cfg or {})
    cfg.pop("name", None)
    if num_classes is not None:
        cfg.setdefault("num_classes", num_classes)
    if pc_range is not None:
        cfg.setdefault("pc_range", pc_range)
    if pillar_size is not None:
        cfg.setdefault("pillar_size", pillar_size)
    return Det3DLoss(**cfg)


def build_det3d_metric(metric_cfg, num_classes=None, names=None):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    cfg.setdefault("names", names)
    return Det3DMetric(**cfg)


def build_det3d_postprocess(pp_cfg, pc_range=None, pillar_size=None):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    if pc_range is not None:
        cfg.setdefault("pc_range", pc_range)
    if pillar_size is not None:
        cfg.setdefault("pillar_size", pillar_size)
    return Det3DPostProcess(**cfg)
