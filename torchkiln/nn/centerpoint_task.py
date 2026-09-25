"""CenterPoint-Pillars task side: target generation, loss, post-process, metric.

Bridges the framework ``det3d`` data format ``[cls, x, y, z, l, w, h, yaw]``
(``yaw`` = box length direction, = KITTI ``ry + pi/2``) to the Paddle3D
CenterPoint convention ``(x, y, z, w, l, h, ry)`` with ``ry = yaw - pi/2``.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CenterPointLoss",
    "CenterPointPostProcess",
    "CenterPointMetric",
    "build_centerpoint_loss",
    "build_centerpoint_postprocess",
    "build_centerpoint_metric",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _draw_gaussian(heatmap, center, radius, k=1.0):
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return
    ys = np.arange(diameter) - radius
    xs = np.arange(diameter) - radius
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
    left, right = min(x, radius), min(w - 1 - x, radius)
    top, bottom = min(y, radius), min(h - 1 - y, radius)
    np.maximum(
        heatmap[y - top:y + bottom + 1, x - left:x + right + 1],
        g[radius - top:radius + bottom + 1, radius - left:radius + right + 1] * k,
        out=heatmap[y - top:y + bottom + 1, x - left:x + right + 1],
    )


def _gaussian_radius(height, width, min_overlap=0.5):
    a1, b1 = 1, (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))) / 2
    r2 = 4 * min_overlap ** 2 * width * height
    r2 = np.sqrt(max(r2, 0.0)) / 4
    return max(r1, r2, 1.0)


def _gather_feat(feat, ind):
    """feat (B, C, H, W) or (B, L, C); ind (B, M) -> (B, M, C)."""
    if feat.dim() == 4:
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
    B, L, C = feat.shape
    ind = ind.unsqueeze(2).expand(B, ind.shape[1], C)
    return feat.gather(1, ind)


class _FastFocalLoss(nn.Module):
    def forward(self, out, target, ind, mask, cat):
        mask = mask.float()
        neg = (torch.log(1 - out) * out.pow(2) * (1 - target).pow(4)).sum()
        pos_pred = _gather_feat(out, ind)  # B,M,C
        cls_idx = cat.unsqueeze(2).clamp(0, pos_pred.shape[2] - 1)
        pos_pred = pos_pred.gather(2, cls_idx).squeeze(2)  # B,M
        num_pos = mask.sum()
        pos = (torch.log(pos_pred.clamp(min=1e-6)) * (1 - pos_pred).pow(2) * mask).sum()
        if num_pos.item() == 0:
            return -neg
        return -(pos + neg) / num_pos


class _RegLoss(nn.Module):
    def forward(self, output, mask, ind, target):
        pred = _gather_feat(output, ind)  # B,M,C
        mask = mask.float().unsqueeze(2)
        loss = F.l1_loss(pred * mask, target * mask, reduction="none")
        loss = loss / (mask.sum() + 1e-4)
        return loss.sum(dim=0).sum(dim=0)  # (8,) per code dimension


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
class CenterPointLoss(nn.Module):
    def __init__(self, num_classes=(1, 2), class_ids=None, point_cloud_range=None,
                 voxel_size=None, down_ratio=2, gaussian_overlap=0.1,
                 max_objs=500, min_radius=2, weight=2.5,
                 code_weights=(1.0,) * 8, **kwargs):
        super().__init__()
        self.num_classes = tuple(num_classes)
        # class_ids[task] = list of global class ids for that task
        self.class_ids = class_ids or [list(range(n)) for n in num_classes]
        pc = list(point_cloud_range or [0, -39.68, -3, 69.12, 39.68, 1])
        self.xmin, self.ymin = float(pc[0]), float(pc[1])
        vs = list(voxel_size or [0.16, 0.16, 4.0])
        self.vx, self.vy = float(vs[0]), float(vs[1])
        self.down_ratio = int(down_ratio)
        self.gaussian_overlap = float(gaussian_overlap)
        self.max_objs = int(max_objs)
        self.min_radius = int(min_radius)
        self.weight = float(weight)
        self.code_weights = torch.tensor(code_weights, dtype=torch.float32)
        self.crit = _FastFocalLoss()
        self.crit_reg = _RegLoss()

    def _targets(self, labels, mask, hm_size, task_id):
        b, ny, nx = hm_size
        n_cls = self.num_classes[task_id]
        ids = self.class_ids[task_id]
        hm = torch.zeros(b, n_cls, ny, nx, device=labels.device)
        tbox = torch.zeros(b, self.max_objs, 8, device=labels.device)
        idx = torch.zeros(b, self.max_objs, dtype=torch.long, device=labels.device)
        tmask = torch.zeros(b, self.max_objs, device=labels.device)
        tlabel = torch.zeros(b, self.max_objs, dtype=torch.long, device=labels.device)
        for i in range(b):
            m = mask[i] > 0.5
            boxes = labels[i][m]
            keep = [k for k in range(boxes.shape[0]) if int(boxes[k, 0].item()) in ids]
            boxes = boxes[keep]
            n = min(boxes.shape[0], self.max_objs)
            for j in range(n):
                gid = int(boxes[j, 0].item())
                c = ids.index(gid)
                x, y, z = (float(boxes[j, 1]), float(boxes[j, 2]), float(boxes[j, 3]))
                l, w, h, yaw = (float(boxes[j, 4]), float(boxes[j, 5]),
                                float(boxes[j, 6]), float(boxes[j, 7]))
                ry = yaw - np.pi / 2.0
                ws = w / self.vx / self.down_ratio
                ls = l / self.vy / self.down_ratio
                if ws <= 0 or ls <= 0:
                    continue
                radius = max(self.min_radius, int(_gaussian_radius(ls, ws, self.gaussian_overlap)))
                cx = (x - self.xmin) / self.vx / self.down_ratio
                cy = (y - self.ymin) / self.vy / self.down_ratio
                ci = np.array([cx, cy], np.float32)
                cint = ci.astype(np.int32)
                if not (0 <= cint[0] < nx and 0 <= cint[1] < ny):
                    continue
                blob = hm[i, c].cpu().numpy()
                _draw_gaussian(blob, ci, radius)
                hm[i, c] = torch.from_numpy(blob).to(hm.device)
                tlabel[i, j] = c
                idx[i, j] = cint[1] * nx + cint[0]
                tmask[i, j] = 1
                tbox[i, j] = torch.tensor(
                    [ci[0] - cint[0], ci[1] - cint[1], z,
                     np.log(w), np.log(l), np.log(h), np.sin(ry), np.cos(ry)],
                    device=labels.device)
        return hm, tbox, idx, tmask, tlabel

    def forward(self, preds, batch):
        labels, mask = batch[1], batch[2]
        hm0 = preds[0]["hm"]
        b, _, ny, nx = hm0.shape
        losses = []
        hm_losses, loc_losses = [], []
        for tid, pred in enumerate(preds):
            hm_t, tbox, idx, tmask, tlabel = self._targets(labels, mask, (b, ny, nx), tid)
            hm = pred["hm"].sigmoid().clamp(1e-4, 1 - 1e-4)
            hm_loss = self.crit(hm, hm_t, idx, tmask, tlabel)
            box = torch.cat([pred["reg"], pred["height"], pred["dim"], pred["rot"]], dim=1)
            box_loss = self.crit_reg(box, tmask, idx, tbox)
            cw = self.code_weights.to(box_loss.device)
            loc_loss = (box_loss * cw).sum()
            losses.append(hm_loss + self.weight * loc_loss)
            hm_losses.append(hm_loss.detach())
            loc_losses.append(loc_loss.detach())
        return {
            "loss": sum(losses),
            "loss_cls": sum(hm_losses),
            "loss_box": sum(loc_losses),
        }


# --------------------------------------------------------------------------- #
# post-process / metric
# --------------------------------------------------------------------------- #
class CenterPointPostProcess(object):
    def __init__(self, num_classes=(1, 2), class_ids=None, point_cloud_range=None,
                 voxel_size=None, down_ratio=2, score_thres=0.1, nms_thres=0.1,
                 max_det=100, **kwargs):
        self.num_classes = tuple(num_classes)
        self.class_ids = class_ids or [list(range(n)) for n in num_classes]
        pc = list(point_cloud_range or [0, -39.68, -3, 69.12, 39.68, 1])
        self.pc_range = pc
        vs = list(voxel_size or [0.16, 0.16, 4.0])
        self.vs = vs
        self.dr = int(down_ratio)
        self.score_thres = float(score_thres)
        self.nms_thres = float(nms_thres)
        self.max_det = int(max_det)

    @torch.no_grad()
    def __call__(self, preds, **kwargs):
        from torchkiln.det.rbox import nms_rotated

        pc = torch.tensor(self.pc_range)
        outs = []
        for i in range(len(preds[0]["hm"])):
            boxes, scores, labels = [], [], []
            for tid, pred in enumerate(preds):
                hm = pred["hm"][i:i + 1].sigmoid()
                dim = pred["dim"][i:i + 1].exp()
                rot = torch.atan2(pred["rot"][i:i + 1, 0:1], pred["rot"][i:i + 1, 1:2])
                B, _, H, W = hm.shape
                ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W),
                                        indexing="ij")
                xs = (xs[None, None].float().to(hm.device) + pred["reg"][i:i + 1, 0:1]) \
                    * self.dr * self.vs[0] + self.pc_range[0]
                ys = (ys[None, None].float().to(hm.device) + pred["reg"][i:i + 1, 1:2]) \
                    * self.dr * self.vs[1] + self.pc_range[1]
                box = torch.cat([xs, ys, pred["height"][i:i + 1], dim, rot], 1)
                box = box.permute(0, 2, 3, 1).reshape(1, -1, 7)  # x,y,z,w,l,h,ry
                score, lab = hm.max(dim=1)
                box = box[0]; score = score.reshape(-1); lab = lab.reshape(-1)
                ctr = box[:, :3]
                msk = score > self.score_thres
                bx, sc, lb = box[msk], score[msk], lab[msk]
                if sc.numel() == 0:
                    continue
                keep = nms_rotated(bx[:, [0, 1, 4, 3, 6]], sc, self.nms_thres).cpu().numpy()
                ids = self.class_ids[tid]
                for t in keep:
                    b = bx[t].cpu().numpy()
                    s = float(sc[t])
                    c_local = int(lb[t])
                    if c_local >= len(ids):
                        continue
                    # framework bev box: [x, y, l, w, yaw], yaw = ry + pi/2
                    boxes.append([b[0], b[1], b[4], b[3], b[6] + np.pi / 2.0])
                    scores.append(s)
                    labels.append(ids[c_local])
            if boxes:
                boxes = np.asarray(boxes, np.float32)
                scores = np.asarray(scores, np.float32)
                labels = np.asarray(labels, np.int64)
                if boxes.shape[0] > self.max_det:
                    o = np.argsort(-scores)[: self.max_det]
                    boxes, scores, labels = boxes[o], scores[o], labels[o]
            else:
                boxes = np.zeros((0, 5), np.float32)
                scores = np.zeros((0,), np.float32)
                labels = np.zeros((0,), np.int64)
            outs.append({"bboxes": boxes, "scores": scores, "labels": labels})
        return outs


class CenterPointMetric(object):
    def __init__(self, iou_thresholds=None, main_indicator="mAP", names=None, **kwargs):
        from torchkiln.det.metric import DetMetric

        thr = list(iou_thresholds) if iou_thresholds else [0.5, 0.7]
        self._inner = DetMetric(iou_thresholds=thr, main_indicator=main_indicator,
                                box_format="xywhr")
        self.main_indicator = main_indicator
        self.names = names

    def reset(self):
        self._inner.reset()

    def __call__(self, post_result, batch):
        # batch[1] is (B,M,8) [cls,x,y,z,l,w,h,yaw]; DetMetric wants (B,M,6)
        # bev [cls,x,y,l,w,yaw] (indices 0,1,2,4,5,7).
        t8 = batch[1].detach().cpu().numpy()
        m = batch[2].detach().cpu().numpy()
        bev = np.concatenate(
            [t8[..., 0:1], t8[..., 1:3], t8[..., 4:5], t8[..., 5:6], t8[..., 7:8]],
            axis=-1,
        )
        pred_t = [
            {"bboxes": torch.as_tensor(np.asarray(p["bboxes"]), dtype=torch.float32),
             "scores": torch.as_tensor(np.asarray(p["scores"]), dtype=torch.float32),
             "labels": torch.as_tensor(np.asarray(p["labels"]), dtype=torch.int64)}
            for p in post_result
        ]
        batch_bev = [None, torch.as_tensor(bev, dtype=torch.float32),
                     torch.as_tensor(m.astype(np.float32))]
        self._inner(pred_t, batch_bev)

    def get_metric(self):
        raw = self._inner.get_metric()
        return {
            "mAP": raw.get("mAP50-95", 0.0),
            "mAP50": raw.get("mAP50", 0.0),
            "mAP70": raw.get("mAP75", 0.0),
        }


def _resolve_classes(arch, ds_cfg):
    """Return (num_class tuple, class_ids per task) from config."""
    head = arch.get("Head") or {}
    tasks = head.get("tasks")
    if tasks:
        num_class = tuple(int(t.get("num_class", len(t.get("class_names", [])))) for t in tasks)
        if all(t.get("class_names") for t in tasks):
            names = list((ds_cfg or {}).get("names") or [])
            idx = {n: i for i, n in enumerate(names)}
            class_ids = [[idx[n] for n in t["class_names"]] for t in tasks]
        else:
            class_ids = None
        return num_class, class_ids
    nc = int(head.get("num_classes", 3))
    num_class = (1, nc - 1) if nc > 1 else (1,)
    return num_class, None


def build_centerpoint_loss(loss_cfg, arch, ds_cfg):
    cfg = dict(loss_cfg or {})
    cfg.pop("name", None)
    num_class, class_ids = _resolve_classes(arch, ds_cfg)
    cfg.setdefault("num_classes", num_class)
    cfg.setdefault("class_ids", class_ids)
    cfg.setdefault("point_cloud_range", (ds_cfg or {}).get("point_cloud_range")
                   or (arch.get("Head") or {}).get("point_cloud_range"))
    cfg.setdefault("voxel_size", (arch.get("Head") or {}).get("voxel_size"))
    cfg.pop("names", None)
    return CenterPointLoss(**cfg)


def build_centerpoint_postprocess(pp_cfg, arch, ds_cfg):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    num_class, class_ids = _resolve_classes(arch, ds_cfg)
    cfg.setdefault("num_classes", num_class)
    cfg.setdefault("class_ids", class_ids)
    cfg.setdefault("point_cloud_range", (ds_cfg or {}).get("point_cloud_range")
                   or (arch.get("Head") or {}).get("point_cloud_range"))
    cfg.setdefault("voxel_size", (arch.get("Head") or {}).get("voxel_size"))
    cfg.pop("names", None)
    return CenterPointPostProcess(**cfg)


def build_centerpoint_metric(metric_cfg, arch, ds_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    cfg.setdefault("names", (ds_cfg or {}).get("names"))
    return CenterPointMetric(**cfg)
