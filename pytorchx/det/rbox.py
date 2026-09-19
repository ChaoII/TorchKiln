"""Oriented-box (OBB) helpers: xywhr <-> polygon, ProbIoU, rotated NMS/IoU.

Layout convention for rotated boxes: ``(cx, cy, w, h, theta)`` with ``theta`` in
radians in ``(-pi/2, pi/2]``. Corners are returned in order
top-left, top-right, bottom-right, bottom-left of the *unrotated* box, rotated
by ``theta`` (counter-clockwise).
"""
from __future__ import absolute_import

import math

import cv2
import numpy as np
import torch

__all__ = [
    "poly2rbox",
    "rbox2poly",
    "rbox2poly_np",
    "dist2rbox",
    "rbox2dist",
    "probiou",
    "points_in_rboxes",
    "poly_iou_np",
    "nms_rotated",
    "ensure_ccw",
]


# --------------------------------------------------------------------- convert
def _norm_angle(t):
    """Wrap to (-pi/2, pi/2]."""
    t = np.asarray(t, dtype=np.float64)
    t = np.mod(t + np.pi / 2, np.pi) - np.pi / 2
    return t


def poly2rbox(points):
    """4 corner points (4, 2) -> ``(cx, cy, w, h, theta)``."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    e1 = pts[1] - pts[0]
    e2 = pts[3] - pts[0]
    w = float(np.hypot(*e1))
    h = float(np.hypot(*e2))
    theta = float(np.arctan2(e1[1], e1[0]))
    if w < h:
        w, h = h, w
        theta += np.pi / 2
    theta = float(_norm_angle(theta))
    return np.array([cx, cy, w, h, theta], dtype=np.float32)


def rbox2poly_np(boxes):
    """``(N, 5)`` xywhr -> ``(N, 4, 2)`` corners."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    cx, cy, w, h, t = boxes.T
    cos, sin = np.cos(t), np.sin(t)
    hw, hh = w / 2, h / 2
    dx = np.stack([-hw, hw, hw, -hw], axis=1)
    dy = np.stack([-hh, -hh, hh, hh], axis=1)
    x = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    y = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return np.stack([x, y], axis=-1)


def rbox2poly(boxes):
    """Torch version of :func:`rbox2poly_np`; ``(..., 5) -> (..., 4, 2)``."""
    cx, cy, w, h, t = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3], boxes[..., 4]
    cos, sin = torch.cos(t), torch.sin(t)
    hw, hh = w / 2, h / 2
    dx = torch.stack([-hw, hw, hw, -hw], dim=-1)
    dy = torch.stack([-hh, -hh, hh, hh], dim=-1)
    x = cx.unsqueeze(-1) + dx * cos.unsqueeze(-1) - dy * sin.unsqueeze(-1)
    y = cy.unsqueeze(-1) + dx * sin.unsqueeze(-1) + dy * cos.unsqueeze(-1)
    return torch.stack([x, y], dim=-1)


# ----------------------------------------------------------------------- probiou
def dist2rbox(dist_angle, anchor_points, stride_tensor=None):
    """Decode ``(l, t, r, b, angle_logit)`` into ``xywhr`` OBBs.

    Aligned with ultralytics ``dist2rbox`` + OBB-head angle decoding: the box
    centre offset is rotated by the angle in *grid* units, ``w = l + r`` /
    ``h = t + b``, and ``theta = (sigmoid(angle_logit) - 0.25) * pi`` (the
    ultralytics ``OBB.forward`` angle decode). Returns grid-unit ``xywhr``
    unless ``stride_tensor`` scales it to pixels.
    """
    dist = dist_angle[..., :4]
    ang = dist_angle[..., 4:5]
    theta = (torch.sigmoid(ang) - 0.25) * math.pi
    lt, rb = dist.split(2, dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    xf, yf = ((rb - lt) / 2).split(1, dim=-1)
    x = xf * cos - yf * sin
    y = xf * sin + yf * cos
    xy = torch.cat([x, y], dim=-1) + anchor_points
    wh = lt + rb
    out = torch.cat([xy, wh], dim=-1)
    if stride_tensor is not None:
        out = out * stride_tensor
    return torch.cat([out, theta], dim=-1)


def rbox2dist(target_bboxes, anchor_points, target_angle, dim=-1, reg_max=None):
    """Inverse of :func:`dist2rbox`: ``xywhr`` -> ``(l, t, r, b)`` in grid units."""
    xy, wh = target_bboxes.split(2, dim)
    offset = xy - anchor_points
    ox, oy = offset.split(1, dim)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = ox * cos + oy * sin
    yf = -ox * sin + oy * cos
    w, h = wh.split(1, dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf
    dist = torch.cat([target_l, target_t, target_r, target_b], dim)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)
    return dist


def _probiou_pairwise(b1, b2, eps=1e-7, floor=0.0):
    """Element-wise ProbIoU; ``b1``/``b2`` broadcast to the same shape ``(..., 5)``.

    Aligned with ultralytics ``batch_probiou`` (ProbIoU from
    arXiv:2106.06072): the Gaussian covariance of an ``xywhr`` box uses the
    uniform-distribution variance ``w^2/12`` / ``h^2/12``.
    """

    def _cov(b):
        gbbs = torch.cat((b[..., 2:4].pow(2) / 12 + floor, b[..., 4:5]), dim=-1)
        aa, bb, c = gbbs.split(1, dim=-1)
        cos, sin = c.cos(), c.sin()
        cos2, sin2 = cos.pow(2), sin.pow(2)
        return aa * cos2 + bb * sin2, aa * sin2 + bb * cos2, (aa - bb) * cos * sin

    x1, y1 = b1[..., 0:1], b1[..., 1:2]
    x2, y2 = b2[..., 0:1], b2[..., 1:2]
    a1, b1_, c1 = _cov(b1)
    a2, b2_, c2 = _cov(b2)

    denom = (a1 + a2) * (b1_ + b2_) - (c1 + c2).pow(2) + eps
    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1_ + b2_) * (x1 - x2).pow(2)) / denom) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    t3 = (
        ((a1 + a2) * (b1_ + b2_) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1_ - c1.pow(2)).clamp_(0) * (a2 * b2_ - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5

    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return (1.0 - hd).squeeze(-1)


def probiou(box1, box2, eps=1e-7, floor=0.0):
    """Probabilistic IoU between rotated boxes.

    * ``(N, 5)`` against ``(N, 5)``  -> element-wise ``(N,)``
    * ``(B, N, 5)`` against ``(B, M, 5)`` -> cross ``(B, N, M)``
    """
    if box1.dim() == 2:
        return _probiou_pairwise(box1, box2, eps, floor)
    return _probiou_pairwise(box1.unsqueeze(2), box2.unsqueeze(1), eps, floor)


def points_in_rboxes(points, boxes, margin=0.0):
    """``points (A,2)`` inside ``boxes (B,M,5)`` xywhr -> bool ``(B,A,M)``."""
    dx = points[None, :, None, 0] - boxes[:, None, :, 0]
    dy = points[None, :, None, 1] - boxes[:, None, :, 1]
    cos = torch.cos(boxes[:, None, :, 4])
    sin = torch.sin(boxes[:, None, :, 4])
    local_x = dx * cos + dy * sin
    local_y = -dx * sin + dy * cos
    return (local_x.abs() <= boxes[:, None, :, 2] / 2 + margin) & (
        local_y.abs() <= boxes[:, None, :, 3] / 2 + margin
    )


# ----------------------------------------------------------------------- polygon
def ensure_ccw(poly):
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return poly[::-1] if area < 0 else poly


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _line_intersect(p1, p2, a, b):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py])


def _clip(subject, clip_poly):
    """Sutherland-Hodgman clip of ``subject`` by convex CCW ``clip_poly``."""
    output = [p for p in subject]
    n = len(clip_poly)
    for i in range(n):
        a, b = clip_poly[i], clip_poly[(i + 1) % n]
        inp = output
        output = []
        if not inp:
            break
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]

            def _inside(p):
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

            if _inside(cur):
                if not _inside(prev):
                    output.append(_line_intersect(prev, cur, a, b))
                output.append(cur)
            elif _inside(prev):
                output.append(_line_intersect(prev, cur, a, b))
    return np.array(output) if output else np.empty((0, 2))


def poly_iou_np(poly1, poly2):
    """IoU of two convex polygons (4,2)."""
    a1 = _poly_area(poly1)
    a2 = _poly_area(poly2)
    if a1 <= 0 or a2 <= 0:
        return 0.0
    inter = _poly_area(_clip(poly1, ensure_ccw(poly2)))
    return float(inter / (a1 + a2 - inter + 1e-9))


def _pairwise_probiou(boxes):
    """``(N,5)`` xywhr -> ``(N,N)`` ProbIoU matrix (GPU-vectorised)."""
    return probiou(boxes.unsqueeze(0), boxes.unsqueeze(0))[0].squeeze_(-1)


def nms_rotated(boxes, scores, iou_thres, max_candidates=3000):
    """Rotated NMS on ``(N,5)`` xywhr boxes -> kept indices.

    Aligned with ultralytics ``TorchNMS.fast_nms(boxes, scores, iou_thres,
    iou_func=batch_probiou)``: ProbIoU pairwise matrix + upper-triangular
    suppression, fully vectorised on the device (fast vs. the old O(N²)
    per-pair polygon clipping).

    ``max_candidates`` mirrors ultralytics ``max_nms``: if more than this many
    boxes pass the confidence filter, only the top-scoring ones are kept before
    building the (N, N) ProbIoU matrix, bounding GPU memory on dense scenes.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    sorted_idx = torch.argsort(scores, descending=True)
    b = boxes[sorted_idx]
    n = b.shape[0]
    if n > max_candidates:
        b = b[:max_candidates]
        sorted_idx = sorted_idx[:max_candidates]
    if b.shape[0] == 1:
        return sorted_idx
    ious = _pairwise_probiou(b).triu_(diagonal=1)
    pick = torch.nonzero((ious >= iou_thres).sum(0) <= 0).squeeze_(-1)
    return sorted_idx[pick]
