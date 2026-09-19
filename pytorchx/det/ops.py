"""Detection building blocks: box math, IoU/CIoU, NMS, letterbox, anchors.

Independent implementation from the standard formulations.
"""
from __future__ import absolute_import

import cv2
import numpy as np
import torch

__all__ = [
    "xywh2xyxy",
    "xyxy2xywh",
    "box_iou",
    "batch_box_iou",
    "bbox_ciou",
    "make_anchors",
    "dist2bbox",
    "bbox2dist",
    "dfl_project",
    "split_head",
    "nms",
    "letterbox",
]


def xywh2xyxy(x):
    """(..., 4) center-x, center-y, w, h -> x1, y1, x2, y2."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.array(x, copy=True)
    dw = y[..., 2] / 2
    dh = y[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.array(x, copy=True)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def box_iou(box1, box2, eps=1e-7):
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    area1 = (a2 - a1).clamp(0).prod(2)
    area2 = (b2 - b1).clamp(0).prod(2)
    union = area1 + area2 - inter + eps
    return inter / union


def batch_box_iou(box1, box2, eps=1e-7):
    """IoU between ``(B, N, 4)`` and ``(B, M, 4)`` xyxy boxes -> ``(B, N, M)``."""
    (a1x, a1y, a2x, a2y) = box1.unsqueeze(2).chunk(4, -1)
    (b1x, b1y, b2x, b2y) = box2.unsqueeze(1).chunk(4, -1)
    inter = (torch.min(a2x, b2x) - torch.max(a1x, b1x)).clamp(0) * (
        torch.min(a2y, b2y) - torch.max(a1y, b1y)
    ).clamp(0)
    area1 = (a2x - a1x).clamp(0) * (a2y - a1y).clamp(0)
    area2 = (b2x - b1x).clamp(0) * (b2y - b1y).clamp(0)
    return (inter / (area1 + area2 - inter + eps)).squeeze(-1)


def batch_ciou(box1, box2, eps=1e-7):
    """CIoU between ``(B, N, 4)`` and ``(B, M, 4)`` xyxy -> ``(B, N, M)`` (clamped >=0).

    对齐 ultralytics ``TaskAlignedAssigner.iou_calculation``:对齐度量用 CIoU 而非 IoU。
    """
    (a1x, a1y, a2x, a2y) = box1.unsqueeze(2).chunk(4, -1)
    (b1x, b1y, b2x, b2y) = box2.unsqueeze(1).chunk(4, -1)
    inter = (torch.min(a2x, b2x) - torch.max(a1x, b1x)).clamp(0) * (
        torch.min(a2y, b2y) - torch.max(a1y, b1y)
    ).clamp(0)
    w1, h1 = (a2x - a1x).clamp(0), (a2y - a1y).clamp(0)
    w2, h2 = (b2x - b1x).clamp(0), (b2y - b1y).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.max(a2x, b2x) - torch.min(a1x, b1x)
    ch = torch.max(a2y, b2y) - torch.min(a1y, b1y)
    c2 = cw**2 + ch**2 + eps
    rho2 = (((b2x + b1x) - (a2x + a1x)) ** 2 + ((b2y + b1y) - (a2y + a1y)) ** 2) / 4
    v = (4 / np.pi**2) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return (iou - (rho2 / c2 + v * alpha)).squeeze(-1).clamp_(0)


def bbox_ciou(box1, box2, eps=1e-7):
    """CIoU between (N,4) and (N,4) xyxy boxes; returns ``1 - CIoU``."""
    (x1, y1, x2, y2), (x1g, y1g, x2g, y2g) = box1.T, box2.T
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    cx2, cy2 = (x1g + x2g) / 2, (y1g + y2g) / 2
    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = x2g - x1g, y2g - y1g

    inter = (torch.min(x2, x2g) - torch.max(x1, x1g)).clamp(0) * (
        torch.min(y2, y2g) - torch.max(y1, y1g)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(x2, x2g) - torch.min(x1, x1g)
    ch = torch.max(y2, y2g) - torch.min(y1, y1g)
    c2 = cw**2 + ch**2 + eps

    rho2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    v = (4 / np.pi**2) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return 1 - (iou - rho2 / c2 - alpha * v)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Feature maps -> (anchor_points (A,2), stride_tensor (A,1)) in pixel units."""
    anchor_points, stride_tensor = [], []
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, dtype=feat.dtype, device=feat.device) + grid_cell_offset
        sy = torch.arange(h, dtype=feat.dtype, device=feat.device) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=feat.dtype, device=feat.device)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """ltrb distances -> xyxy (or xywh) boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox, reg_max=None):
    """xyxy boxes -> ltrb distances."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0)


def dfl_project(x, reg_max):
    """Distribution Focal Loss projection: ``(..., 4*reg_max)`` -> ``(..., 4)``.

    Each of the four sides is a softmax distribution over ``reg_max`` bins; the
    distance is its expectation (the standard DFL decode).
    """
    if reg_max <= 1:
        return x
    shape = x.shape[:-1]
    x = x.reshape(*shape, 4, reg_max).softmax(-1)
    bins = torch.arange(reg_max, dtype=x.dtype, device=x.device)
    return (x * bins).sum(-1)


def split_head(feats, num_classes, reg_max=1, mode="det", ne=1, project=True):
    """Split head outputs into ``(dist, scores, extra)``, applying DFL.

    Channel layouts per level:

    * ``det``      : ``[reg(4*reg_max), cls]``
    * ``obb``      : ``[reg(4*reg_max), cls, angle]``   (upstream layout)
    * ``obb_ours`` : ``[reg(4+ne), cls]``               (our hand-written layout)
    * ``seg``      : ``[reg(4*reg_max), cls, coeff]``
    * ``pose``     : ``[reg(4*reg_max), cls, kpt]``

    ``dist`` is ``(B, A, 4)`` (``5`` for obb), ``extra`` holds coeff/kpt or None.
    """
    rc = 4 * reg_max
    dists, scores, extras = [], [], []
    for feat in feats:
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w).permute(0, 2, 1)
        if mode == "obb_ours":
            dists.append(f[..., : 4 + ne])
            scores.append(f[..., 4 + ne : 4 + ne + num_classes])
        else:
            dists.append(f[..., :rc])
            scores.append(f[..., rc : rc + num_classes])
            if mode in ("obb", "seg", "pose"):
                extras.append(f[..., rc + num_classes :])
    dist = torch.cat(dists, 1)
    if mode != "obb_ours" and reg_max > 1 and project:
        dist = dfl_project(dist, reg_max)
    if mode == "obb":
        dist = torch.cat([dist, torch.cat(extras, 1)], dim=-1)
    scores = torch.cat(scores, 1)
    extra = None
    if mode in ("seg", "pose"):
        extra = torch.cat(extras, 1) if extras else None
    return dist, scores, extra


def nms(boxes, scores, iou_thres):
    """Greedy NMS. ``boxes`` (N,4) xyxy, ``scores`` (N,). Returns kept indices."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep = []
    order = scores.argsort(descending=True)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thres]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to ``new_shape``.

    Returns ``(img, ratio, (dw, dh))``.
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)
