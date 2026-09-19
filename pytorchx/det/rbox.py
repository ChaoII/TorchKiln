"""Oriented-box (OBB) helpers: xywhr <-> polygon, ProbIoU, rotated NMS/IoU.

Layout convention for rotated boxes: ``(cx, cy, w, h, theta)`` with ``theta`` in
radians in ``(-pi/2, pi/2]``. Corners are returned in order
top-left, top-right, bottom-right, bottom-left of the *unrotated* box, rotated
by ``theta`` (counter-clockwise).
"""
from __future__ import absolute_import

import cv2
import numpy as np
import torch

from pytorchx.det.ops import dist2bbox

__all__ = [
    "poly2rbox",
    "rbox2poly",
    "rbox2poly_np",
    "dist2rbox",
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
def dist2rbox(dist_angle, anchor_points, stride_tensor):
    """Decode ``(l, t, r, b, angle_raw)`` into pixel ``xywhr`` boxes.

    The axis-aligned box is decoded in *grid* units and then scaled by the
    stride; ``theta = atan(angle_raw)`` keeps the angle in ``(-pi/2, pi/2)``.
    """
    dist = dist_angle[..., :4]
    ang = dist_angle[..., 4:5]
    xyxy = dist2bbox(dist, anchor_points, xywh=False)
    cxy = (xyxy[..., 0:2] + xyxy[..., 2:4]) / 2
    wh = xyxy[..., 2:4] - xyxy[..., 0:2]
    out = torch.cat([cxy, wh], dim=-1) * stride_tensor
    return torch.cat([out, torch.atan(ang)], dim=-1)


def _probiou_pairwise(b1, b2, eps=1e-7):
    """Element-wise ProbIoU; ``b1``/``b2`` have the same shape ``(..., 5)``."""

    def _gauss(b):
        cx, cy, w, h, t = (b[..., i] for i in range(5))
        a = (w / 2) ** 2
        bb = (h / 2) ** 2
        cos, sin = torch.cos(t), torch.sin(t)
        c2, s2 = cos * cos, sin * sin
        sxx = a * c2 + bb * s2
        syy = a * s2 + bb * c2
        sxy = (a - bb) * cos * sin
        return cx, cy, sxx, syy, sxy

    cx1, cy1, sxx1, syy1, sxy1 = _gauss(b1)
    cx2, cy2, sxx2, syy2, sxy2 = _gauss(b2)

    det1 = (sxx1 * syy1 - sxy1 * sxy1).clamp(min=eps)
    det2 = (sxx2 * syy2 - sxy2 * sxy2).clamp(min=eps)
    sxx = (sxx1 + sxx2) / 2
    syy = (syy1 + syy2) / 2
    sxy = (sxy1 + sxy2) / 2
    det = (sxx * syy - sxy * sxy).clamp(min=eps)

    dx = cx1 - cx2
    dy = cy1 - cy2
    quad = (syy * dx * dx - 2 * sxy * dx * dy + sxx * dy * dy) / det
    bc = 0.125 * quad + 0.5 * torch.log(det / (torch.sqrt(det1 * det2) + eps) + eps)
    hc = torch.sqrt((1 - torch.exp(-bc)).clamp(min=0, max=1 - 1e-7))
    # similarity in [0, 1]: 1.0 for identical boxes
    return 1.0 - hc


def probiou(box1, box2, eps=1e-7):
    """Probabilistic IoU between rotated boxes.

    * ``(N, 5)`` against ``(N, 5)``  -> element-wise ``(N,)``
    * ``(B, N, 5)`` against ``(B, M, 5)`` -> cross ``(B, N, M)``
    """
    if box1.dim() == 2:
        return _probiou_pairwise(box1, box2, eps)
    return _probiou_pairwise(box1.unsqueeze(2), box2.unsqueeze(1), eps)


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


def nms_rotated(boxes, scores, iou_thres):
    """Greedy rotated NMS on ``(N,5)`` xywhr boxes -> kept indices."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    polys = rbox2poly_np(boxes.detach().cpu().numpy())
    sc = scores.detach().cpu().numpy()
    order = np.argsort(-sc)
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(int(order[i]))
        for j in range(i + 1, len(order)):
            if suppressed[j]:
                continue
            if poly_iou_np(polys[order[i]], polys[order[j]]) > iou_thres:
                suppressed[j] = True
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
