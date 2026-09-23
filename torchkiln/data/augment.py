"""Training augmentations for detection-style tasks.

Independent implementations of the standard recipes:

* :class:`RandomHSV`      - colour jitter (HSV lookup tables)
* :class:`RandomAffine`   - random rotation / scale / translate / perspective
                            (sheet + boxes + keypoints + masks in one matrix)
* :func:`mosaic4`         - 2x2 mosaic of four samples
* :func:`mixup`           - alpha blend of two samples

Geometry contract: labels are ``(N, 5)`` = ``[cls, x1, y1, x2, y2]`` (``xyxy``)
or ``(N, 6)`` = ``[cls, cx, cy, w, h, theta]`` (``xywhr``), in *pixel* coords of
the current image. Keypoints are ``(N, K, 3)``. Masks are ``(N, h, w)``.
"""
from __future__ import absolute_import

import cv2
import numpy as np

__all__ = [
    "RandomHSV",
    "RandomAffine",
    "mosaic4",
    "mixup",
    "box_candidates",
    "TrainAugmenter",
]


def _rand(a=0.0, b=1.0):
    return float(np.random.rand() * (b - a) + a)


# --------------------------------------------------------------------- colour
class RandomHSV(object):
    def __init__(self, p=0.5, hgain=0.015, sgain=0.7, vgain=0.4):
        self.p = float(p)
        self.gains = (float(hgain), float(sgain), float(vgain))

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        r = np.random.uniform(-1, 1, 3) * self.gains + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=np.int16)
        lut_h = ((x * r[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v))
        )
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------- boxes
def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1):
    """Filter boxes that shrank too much / became degenerate after a warp."""
    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )


def _rbox_corners(labels):
    """``(N,6)`` xywhr -> ``(N,4,2)`` corners."""
    cx, cy, w, h, t = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4], labels[:, 5]
    cos, sin = np.cos(t), np.sin(t)
    dx = np.stack([-w / 2, w / 2, w / 2, -w / 2], axis=1)
    dy = np.stack([-h / 2, -h / 2, h / 2, h / 2], axis=1)
    x = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    y = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return np.stack([x, y], axis=-1).astype(np.float32)


def _corners_to_rbox(corners):
    """Corners ``(N,4,2)`` -> ``(N,5)`` (cx, cy, w, h, theta).

    Aligned with ultralytics ``xyxyxyxy2xywhr``: ``cv2.minAreaRect`` + canonical
    parameterisation (``w`` longer side, ``theta`` in ``[-pi/4, 3pi/4)``), so the
    augmented box convention matches the model / ground-truth labels.
    """
    out = np.zeros((corners.shape[0], 5), dtype=np.float32)
    for i, pts in enumerate(corners):
        try:
            (cx, cy), (w, h), angle = cv2.minAreaRect(pts.reshape(-1, 2))
            theta = angle / 180.0 * np.pi
        except cv2.error:
            # degenerate / collinear corners after augment: fall back to edge method
            c = pts.mean(0)
            e1 = pts[1] - pts[0]
            e2 = pts[3] - pts[0]
            w = float(np.hypot(*e1))
            h = float(np.hypot(*e2))
            theta = float(np.arctan2(e1[1], e1[0]))
            cx, cy = c[0], c[1]
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        out[i] = [cx, cy, w, h, theta]
    return out


def _warp_points(M, pts):
    """``M`` 3x3 homography, ``pts`` (...,2) -> same shape."""
    shape = pts.shape[:-1]
    p = pts.reshape(-1, 2).astype(np.float32)
    z = np.ones((p.shape[0], 1), dtype=np.float32)
    q = (M @ np.concatenate([p, z], axis=1).T).T
    q = q[:, :2] / np.maximum(q[:, 2:3], 1e-9)
    return q.reshape(*shape, 2)


# -------------------------------------------------------------------- affine
class RandomAffine(object):
    """Random rotation/scale/translate/perspective producing ``imgsz`` output."""

    def __init__(
        self,
        imgsz,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        border_value=114,
    ):
        self.imgsz = int(imgsz)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.shear = float(shear)
        self.perspective = float(perspective)
        self.border_value = int(border_value)

    def _matrix(self, w, h):
        # 对齐 ultralytics RandomPerspective._compute_affine_matrix:
        # M = T @ S @ R @ P @ C,输出 size=(imgsz, imgsz)。输入中心经 C 平移到原点,
        # 再由 T 映射到输出中心 -> 等价于以**输入画布中心**为基准裁剪/缩放。
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = _rand(-self.perspective, self.perspective)
        P[2, 1] = _rand(-self.perspective, self.perspective)
        R = np.eye(3, dtype=np.float32)
        a = _rand(-self.degrees, self.degrees)
        s = _rand(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D((0, 0), a, s)
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = np.tan(np.deg2rad(_rand(-self.shear, self.shear)))
        S[1, 0] = np.tan(np.deg2rad(_rand(-self.shear, self.shear)))
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = _rand(0.5 - self.translate, 0.5 + self.translate) * self.imgsz
        T[1, 2] = _rand(0.5 - self.translate, 0.5 + self.translate) * self.imgsz
        return (T @ S @ R @ P @ C), s

    def __call__(self, img, labels=None, kpts=None, masks=None, box_format="xyxy"):
        """Warp ``img`` to ``imgsz`` and transform labels/kpts/masks with it.

        The scale factor is baked into the homography, so labels must already be
        in the *current* image coordinates.
        """
        h, w = img.shape[:2]
        M, _ = self._matrix(w, h)
        out = cv2.warpPerspective(
            img,
            M,
            (self.imgsz, self.imgsz),
            flags=cv2.INTER_LINEAR,
            borderValue=(self.border_value,) * 3,
        )
        if labels is None or len(labels) == 0:
            return out, labels, kpts, masks

        labels = labels.copy()
        if box_format == "xyxy":
            corners = np.stack(
                [
                    labels[:, 1:3],
                    np.stack([labels[:, 3], labels[:, 2]], 1),
                    labels[:, 3:5],
                    np.stack([labels[:, 1], labels[:, 4]], 1),
                ],
                axis=1,
            )
        else:
            corners = _rbox_corners(labels)
        warped = _warp_points(M, corners.astype(np.float32))
        x1 = warped[..., 0].min(1)
        x2 = warped[..., 0].max(1)
        y1 = warped[..., 1].min(1)
        y2 = warped[..., 1].max(1)

        new = labels.copy()
        if box_format == "xyxy":
            new[:, 1], new[:, 2], new[:, 3], new[:, 4] = x1, y1, x2, y2
            # 裁剪到图像边界(对齐 ultralytics,避免越界框产生噪声目标)
            new[:, 1] = np.clip(new[:, 1], 0, self.imgsz)
            new[:, 3] = np.clip(new[:, 3], 0, self.imgsz)
            new[:, 2] = np.clip(new[:, 2], 0, self.imgsz)
            new[:, 4] = np.clip(new[:, 4], 0, self.imgsz)
            x1, y1, x2, y2 = new[:, 1], new[:, 2], new[:, 3], new[:, 4]
        else:
            new[:, 1:6] = _corners_to_rbox(warped)

        keep = ((x2 - x1) > 2) & ((y2 - y1) > 2)
        keep &= (x2 > 2) & (y2 > 2) & (x1 < self.imgsz - 2) & (y1 < self.imgsz - 2)
        labels = new[keep]

        if kpts is not None and len(kpts):
            kp = np.zeros_like(kpts, dtype=np.float32)
            kp[..., :2] = _warp_points(M, kpts[..., :2].astype(np.float32))
            if kp.shape[-1] > 2:  # optional visibility channel
                kp[..., 2:] = kpts[..., 2:]
            kpts = kp[keep]
        if masks is not None and len(masks):
            sel_m = masks[keep]
            if len(sel_m):
                masks = np.stack(
                    [
                        cv2.warpPerspective(
                            m.astype(np.uint8),
                            M,
                            (self.imgsz, self.imgsz),
                            flags=cv2.INTER_NEAREST,
                        )
                        for m in sel_m
                    ]
                ).astype(np.float32)
            else:
                masks = None
        return out, labels, kpts, masks


# ------------------------------------------------------------------ mosaics
def _random_crop_mosaic(canvas, labels, kpts, masks, S, box_format, center=None):
    """把 2S/3S 的 mosaic 画布随机裁出 ``S`` 窗口。

    对齐 ultralytics ``RandomPerspective``:其仿射矩阵 ``M=T@S@R@P@C`` 把**输入画布
    中心**映射到输出中心(scale 仅 jitter),等价于在**画布中心** ``(S,S)`` 处裁剪。
    因此 ``center`` 必须传画布中心 ``(S,S)``(而非 mosaic 的 tile 汇聚点),否则裁剪
    窗口偏离,物体分布与 ultra 不一致 → 训练 mAP 系统性偏低(v8n 0.595→0.650)。
    """
    h, w = canvas.shape[:2]
    cx0 = center[0] if center is not None else w / 2.0
    cy0 = center[1] if center is not None else h / 2.0
    lo_x = max(0.0, min(float(w - S), cx0 - S / 2.0))
    lo_y = max(0.0, min(float(h - S), cy0 - S / 2.0))
    x0 = int(np.clip(lo_x + np.random.uniform(-0.1, 0.1) * S, 0, w - S))
    y0 = int(np.clip(lo_y + np.random.uniform(-0.1, 0.1) * S, 0, h - S))
    canvas = np.ascontiguousarray(canvas[y0 : y0 + S, x0 : x0 + S])
    if labels is not None and len(labels):
        lb = labels.copy()
        if box_format == "xywhr":
            lb[:, 1] -= x0
            lb[:, 2] -= y0
        else:
            lb[:, 1] -= x0
            lb[:, 3] -= x0
            lb[:, 2] -= y0
            lb[:, 4] -= y0
            lb[:, 1] = np.clip(lb[:, 1], 0, S)
            lb[:, 3] = np.clip(lb[:, 3], 0, S)
            lb[:, 2] = np.clip(lb[:, 2], 0, S)
            lb[:, 4] = np.clip(lb[:, 4], 0, S)
        # 对齐原版:裁剪不丢弃实例,只 clip(保留上下文,交给后续框有效性处理)
        labels = lb
    if kpts is not None and len(kpts):
        kk = kpts.copy()
        kk[..., 0] -= x0
        kk[..., 1] -= y0
        kpts = kk
    if masks is not None and len(masks):
        masks = np.ascontiguousarray(masks[:, y0 : y0 + S, x0 : x0 + S])
    return canvas, labels, kpts, masks


def mosaic4(items, imgsz, box_format="xyxy", crop=True):
    """2x2 mosaic, ultralytics-style random-centre placement.

    Four patches are placed around a random centre ``(xc, yc)`` so their cores all
    converge near the canvas centre; a later random crop then keeps objects from
    all four patches (this is why ultralytics keeps ~6.9 instances/image vs our
    earlier fixed-corner layout which kept ~5.0).
    """
    s = int(imgsz)
    border = -s // 2
    canvas = np.full((2 * s, 2 * s, 3), 114, np.uint8)
    xc = int(np.random.uniform(-border, 2 * s + border))
    yc = int(np.random.uniform(-border, 2 * s + border))
    out_labels, out_kpts, out_masks = [], [], []
    for i, (img, labels, kpts, masks) in enumerate(items[:4]):
        h, w = img.shape[:2]
        r = min(s / max(w, 1), s / max(h, 1))
        nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
        pw, ph = (s - nw) // 2, (s - nh) // 2
        tile = np.full((s, s, 3), 114, np.uint8)
        tile[ph : ph + nh, pw : pw + nw] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_LINEAR
        )
        if i == 0:  # top-left patch: its bottom-right corner meets (xc, yc)
            x1a, y1a, x2a, y2a = max(xc - s, 0), max(yc - s, 0), xc, yc
            x1b, y1b = s - (x2a - x1a), s - (y2a - y1a)
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - s, 0), min(xc + s, 2 * s), yc
            x1b, y1b = 0, s - (y2a - y1a)
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - s, 0), yc, xc, min(yc + s, 2 * s)
            x1b, y1b = s - (x2a - x1a), 0
        else:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + s, 2 * s), min(yc + s, 2 * s)
            x1b, y1b = 0, 0
        x2b, y2b = x1b + (x2a - x1a), y1b + (y2a - y1a)
        canvas[y1a:y2a, x1a:x2a] = tile[y1b:y2b, x1b:x2b]
        padw, padh = x1a - x1b, y1a - y1b
        if labels is not None and len(labels):
            # scale instances into the patch (s-coords with the black border offset),
            # then translate to the canvas via (padw, padh)
            if box_format == "xywhr":
                lb = np.array(labels)
                lb[:, 1] = lb[:, 1] * r + pw + padw
                lb[:, 2] = lb[:, 2] * r + ph + padh
                lb[:, 3] = lb[:, 3] * r
                lb[:, 4] = lb[:, 4] * r
            else:
                lb = np.array(labels)
                lb[:, 1] = lb[:, 1] * r + pw + padw
                lb[:, 3] = lb[:, 3] * r + pw + padw
                lb[:, 2] = lb[:, 2] * r + ph + padh
                lb[:, 4] = lb[:, 4] * r + ph + padh
            out_labels.append(lb)
            if kpts is not None and len(kpts):
                kk = np.array(kpts)
                kk[..., 0] = kk[..., 0] * r + pw + padw
                kk[..., 1] = kk[..., 1] * r + ph + padh
                out_kpts.append(kk)
            if masks is not None and len(masks):
                new_m = []
                for m in masks:
                    mw = np.zeros((s, s), np.uint8)
                    mm = cv2.resize(
                        m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST
                    )
                    mw[ph : ph + nh, pw : pw + nw] = mm
                    canvas_m = np.zeros((2 * s, 2 * s), np.uint8)
                    canvas_m[y1a:y2a, x1a:x2a] = mw[y1b:y2b, x1b:x2b]
                    new_m.append(canvas_m)
                if new_m:
                    out_masks.append(np.stack(new_m).astype(np.float32))
    width = 6 if box_format == "xywhr" else 5
    labels = (
        np.concatenate(out_labels, 0) if out_labels else np.zeros((0, width), np.float32)
    )
    kpts = np.concatenate(out_kpts, 0) if out_kpts else None
    masks = np.concatenate(out_masks, 0) if out_masks else None
    if not crop:
        # 保留 2S 画布,交由 affine 缩放到 imgsz(对齐 ultralytics:实例不因裁剪丢失)
        return canvas, labels, kpts, masks
    canvas, labels, kpts, masks = _random_crop_mosaic(
        canvas, labels, kpts, masks, s, box_format, center=(s, s)
    )
    return canvas, labels, kpts, masks


def mixup(img1, labels1, kpts1, masks1, img2, labels2, kpts2, masks2, alpha=32.0):
    """Alpha-blend two samples and concatenate their labels."""
    r = np.random.beta(alpha, alpha)
    img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate([labels1, labels2], 0)
    kpts = None
    if kpts1 is not None and kpts2 is not None:
        kpts = np.concatenate([kpts1, kpts2], 0)
    masks = None
    if masks1 is not None and masks2 is not None:
        masks = np.concatenate([masks1, masks2], 0)
    return img, labels, kpts, masks


def mosaic9(items, imgsz, box_format="xyxy"):
    """3x3 mosaic (``mosaic9``): the target sample stays in the centre cell.

    Eight further samples fill the surrounding cells. Each image is zoomed a
    little beyond the cell size and randomly cropped, so objects are frequently
    cut by the cell borders - the stronger scale/context augmentation that
    ``mosaic9`` is known for. Returns a ``3*imgsz`` canvas; ``RandomAffine``
    then resizes it to the training size (like :func:`mosaic4`).
    """
    S = int(imgsz)
    canvas = np.full((3 * S, 3 * S, 3), 114, np.uint8)
    out_labels, out_kpts, out_masks = [], [], []
    order = [0, 1, 2, 3, 5, 6, 7, 8]
    np.random.shuffle(order)
    cells = {4: 0}
    for slot, cell in enumerate(order):
        cells[cell] = slot + 1
    for cell, item_idx in sorted(cells.items(), key=lambda kv: kv[1]):
        if item_idx >= len(items):
            continue
        img, labels, kpts, masks = items[item_idx]
        cy, cx = int(cell) // 3, int(cell) % 3
        oy, ox = cy * S, cx * S
        h, w = img.shape[:2]
        zoom = np.random.uniform(1.0, 1.2)
        r = max(S / max(h, 1), S / max(w, 1)) * zoom
        nw, nh = max(S, int(round(w * r))), max(S, int(round(h * r)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = int(np.random.randint(0, nw - S + 1))
        y0 = int(np.random.randint(0, nh - S + 1))
        canvas[oy : oy + S, ox : ox + S] = resized[y0 : y0 + S, x0 : x0 + S]

        if labels is None or not len(labels):
            continue
        lb = labels.copy()
        width = 6 if box_format == "xywhr" else 5
        if box_format == "xywhr":
            lb[:, 1] = lb[:, 1] * r - x0 + ox
            lb[:, 2] = lb[:, 2] * r - y0 + oy
            lb[:, 3] = np.maximum(lb[:, 3] * r, 2.0)
            lb[:, 4] = np.maximum(lb[:, 4] * r, 2.0)
        else:
            lb[:, 1] = lb[:, 1] * r - x0 + ox
            lb[:, 3] = lb[:, 3] * r - x0 + ox
            lb[:, 2] = lb[:, 2] * r - y0 + oy
            lb[:, 4] = lb[:, 4] * r - y0 + oy
            # clip to the cell: mosaic9 deliberately keeps cut-off objects
            for a, b in ((1, 3),):
                lb[:, a] = np.clip(lb[:, a], ox - 0.5 * S, ox + 1.5 * S)
                lb[:, b] = np.clip(lb[:, b], ox - 0.5 * S, ox + 1.5 * S)
            lb[:, 2] = np.clip(lb[:, 2], oy - 0.5 * S, oy + 1.5 * S)
            lb[:, 4] = np.clip(lb[:, 4], oy - 0.5 * S, oy + 1.5 * S)
        keep = (
            np.ones(len(lb), bool)
            if box_format == "xywhr"
            else ((lb[:, 3] - lb[:, 1]) > 2) & ((lb[:, 4] - lb[:, 2]) > 2)
        )
        out_labels.append(lb[keep])
        if kpts is not None and len(kpts):
            kk = kpts.copy()
            kk[..., 0] = kk[..., 0] * r - x0 + ox
            kk[..., 1] = kk[..., 1] * r - y0 + oy
            out_kpts.append(kk[keep])
        if masks is not None and len(masks):
            new_m = []
            for m in masks[keep]:
                mm = cv2.resize(
                    m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST
                )[y0 : y0 + S, x0 : x0 + S]
                cm = np.zeros((3 * S, 3 * S), np.uint8)
                tile = cm[oy : oy + S, ox : ox + S]
                tile[: mm.shape[0], : mm.shape[1]] = mm
                new_m.append(cm)
            if new_m:
                out_masks.append(np.stack(new_m).astype(np.float32))

    width = 6 if box_format == "xywhr" else 5
    labels = (
        np.concatenate(out_labels, 0) if out_labels else np.zeros((0, width), np.float32)
    )
    kpts = np.concatenate(out_kpts, 0) if out_kpts else None
    masks = np.concatenate(out_masks, 0) if out_masks else None
    canvas, labels, kpts, masks = _random_crop_mosaic(
        canvas, labels, kpts, masks, S, box_format
    )
    return canvas, labels, kpts, masks


def _mask_bbox(mask):
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def copy_paste(
    img,
    labels,
    masks,
    src_img,
    src_labels,
    src_masks,
    box_format="xyxy",
    p=0.5,
    max_paste=10,
    iou_thr=0.3,
):
    """Segmentation copy-paste: paste masked instances from ``src`` onto ``img``.

    Only instances whose class already occurs in ``img`` are pasted (keeping the
    image statistically plausible) and pastes that overlap an existing object
    with ``IoU > iou_thr`` are skipped. ``src_img``/``src_masks`` are resized to
    the target resolution so the paste is geometrically consistent.
    """
    if masks is None or src_masks is None or not len(masks) or not len(src_masks):
        return img, labels, masks
    H, W = img.shape[:2]
    if src_img.shape[:2] != (H, W):
        src_img = cv2.resize(src_img, (W, H), interpolation=cv2.INTER_LINEAR)
        src_masks = np.stack(
            [
                cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                for m in src_masks
            ]
        ).astype(np.float32)
    if src_labels is None or not len(src_labels):
        return img, labels, masks

    classes = set(int(c) for c in labels[:, 0]) if labels is not None and len(labels) else set()
    new_labels, new_masks = [], []
    existing = []
    for row in labels if labels is not None and len(labels) else []:
        if box_format == "xyxy":
            existing.append([row[1], row[2], row[3], row[4]])
        else:
            cx, cy, bw, bh = row[1], row[2], row[3], row[4]
            existing.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])

    for i in range(len(src_masks)):
        if len(new_labels) >= max_paste:
            break
        if np.random.rand() > p:
            continue
        cls = int(src_labels[i, 0])
        if classes and cls not in classes:
            continue
        m = src_masks[i] > 0.5
        bb = _mask_bbox(m.astype(np.float32))
        if bb is None:
            continue
        if any(_box_iou(bb, eb) > iou_thr for eb in existing):
            continue
        x1, y1, x2, y2 = (int(bb[0]), int(bb[1]), int(np.ceil(bb[2])), int(np.ceil(bb[3])))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        region = m[y1:y2, x1:x2]
        img[y1:y2, x1:x2][region] = src_img[y1:y2, x1:x2][region]
        pasted = np.zeros((H, W), np.float32)
        pasted[y1:y2, x1:x2] = region.astype(np.float32)
        exact = _mask_bbox(pasted)
        if exact is None:
            continue
        new_masks.append(pasted)
        row = src_labels[i].copy()
        if box_format == "xyxy":
            row[1], row[2], row[3], row[4] = exact[0], exact[1], exact[2], exact[3]
        else:
            row[1] = (exact[0] + exact[2]) / 2.0
            row[2] = (exact[1] + exact[3]) / 2.0
            row[3] = exact[2] - exact[0]
            row[4] = exact[3] - exact[1]
        new_labels.append(row)
        existing.append(list(exact))

    if not new_labels:
        return img, labels, masks
    labels = np.concatenate([labels, np.stack(new_labels).astype(np.float32)], 0)
    masks = np.concatenate([masks, np.stack(new_masks)], 0)
    return img, labels, masks


class RandomErasing(object):
    """Random-erasing augmentation (classification / image-only).

    Erases up to ``max_count`` rectangles covering ``sl``..``sh`` of the image
    with ``U(0, 1)`` noise, keeping the aspect ratio within ``min_aspect``.
    Accepts a ``(C, H, W)`` array/tensor (as returned by the classification
    transform) and mirrors the upstream ``RandomErasing`` semantics.
    """

    def __init__(self, p=0.25, sl=0.02, sh=0.4, min_aspect=0.3, max_count=1):
        self.p = float(p)
        self.sl = float(sl)
        self.sh = float(sh)
        self.min_aspect = float(min_aspect)
        self.max_count = int(max_count)

    def __call__(self, img):
        if self.p <= 0 or np.random.rand() > self.p:
            return img
        is_torch = hasattr(img, "clone") and not isinstance(img, np.ndarray)
        out = img.clone() if is_torch else np.array(img, copy=True)
        c, h, w = out.shape
        rand = np.random.rand if not is_torch else np.random.rand
        for _ in range(max(1, self.max_count)):
            for _ in range(10):
                area = h * w
                target = np.random.uniform(self.sl, self.sh) * area
                ar = np.random.uniform(self.min_aspect, 1.0 / self.min_aspect)
                eh, ew = int(round(np.sqrt(target * ar))), int(round(np.sqrt(target / ar)))
                if 0 < ew < w and 0 < eh < h:
                    y1 = int(np.random.randint(0, h - eh))
                    x1 = int(np.random.randint(0, w - ew))
                    noise = rand(c, eh, ew).astype(out.dtype)
                    out[:, y1 : y1 + eh, x1 : x1 + ew] = (
                        torch.from_numpy(noise) if is_torch else noise
                    )
                    break
        return out


class TrainAugmenter(object):
    """Orchestrates mosaic -> mixup -> random affine -> HSV (+ multi-scale).

    Config block (all optional)::

        augment:
          mosaic: 1.0            # probability; auto-disabled after close_mosaic
          mosaic9: 0.0           # 3x3 mosaic (nine images); takes precedence
          mixup: 0.0
          copy_paste: 0.0        # segmentation copy-paste (needs masks)
          close_mosaic: 10       # epochs
          multi_scale: 0.5       # per-epoch random size in [1-ms, 1+ms]
          hsv: {p: 0.5, hgain: 0.015, sgain: 0.7, vgain: 0.4}
          affine: {degrees: 0.0, translate: 0.1, scale: 0.5, shear: 0.0, perspective: 0.0}
    """

    def __init__(self, cfg, imgsz, box_format="xyxy"):
        cfg = dict(cfg or {})
        self.imgsz = int(imgsz)
        self.box_format = box_format
        # 未显式配置时,采用 ultralytics 默认配方(cfg/default.yaml)
        default = not cfg or all(k not in cfg for k in ("mosaic", "mosaic9", "mixup", "copy_paste"))
        self.mosaic_p = float(cfg.get("mosaic", 1.0 if default else 0.0))
        self.mosaic9_p = float(cfg.get("mosaic9", 0.0))
        self.mixup_p = float(cfg.get("mixup", 0.0))
        self.copy_paste_p = float(cfg.get("copy_paste", 0.0))
        self.mosaic_crop = bool(cfg.get("mosaic_crop", True))
        self.close_mosaic = int(cfg.get("close_mosaic", 10 if default else 0))
        self.multi_scale = float(cfg.get("multi_scale", 0.0))
        hsv = cfg.get("hsv")
        if hsv is None and default:
            hsv = {"p": 1.0, "hgain": 0.015, "sgain": 0.7, "vgain": 0.4}
        self.hsv = RandomHSV(**(hsv if isinstance(hsv, dict) else {})) if hsv else None
        affine = cfg.get("affine")
        if affine is None and default:
            affine = {"degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0}
        self.affine = RandomAffine(
            self.imgsz, **(affine if isinstance(affine, dict) else {})
        )
        self.fliplr = float(cfg.get("fliplr", 0.5 if default else 0.0))
        self.flipud = float(cfg.get("flipud", 0.0 if default else 0.0))
        self.epoch = 0
        self.total_epochs = int(cfg.get("total_epochs", 0) or 0)
        self._mosaic_on = self.mosaic_p > 0 or self.mosaic9_p > 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        # ultralytics 语义:在 epoch == total_epochs - close_mosaic 时关闭 mosaic;
        # 若 total_epochs < close_mosaic(短训练)则**从不关闭**。
        if self.close_mosaic and self.total_epochs:
            start = self.total_epochs - self.close_mosaic
            if start >= 0 and self.epoch >= start:
                self._mosaic_on = False
        elif self.close_mosaic and not self.total_epochs:
            if self.epoch > self.close_mosaic:
                self._mosaic_on = False
        if self.multi_scale:
            lo = max(64, int(self.imgsz * (1 - self.multi_scale)))
            hi = int(self.imgsz * (1 + self.multi_scale))
            choices = list(range(lo, hi + 1, 32)) or [self.imgsz]
            size = int(np.random.choice(choices))
            self.affine.imgsz = size

    def __call__(self, dataset, index):
        if self.multi_scale:   # ultralytics: 每个 batch 随机尺寸 ±multi_scale
            lo = max(64, int(self.imgsz * (1 - self.multi_scale)))
            hi = int(self.imgsz * (1 + self.multi_scale))
            choices = list(range(lo, hi + 1, 32)) or [self.imgsz]
            self.affine.imgsz = int(np.random.choice(choices))
        use_mosaic9 = self._mosaic_on and self.mosaic9_p > 0 and np.random.rand() < self.mosaic9_p
        use_mosaic = (
            not use_mosaic9
            and self._mosaic_on
            and np.random.rand() < self.mosaic_p
        )
        if use_mosaic9:
            idxs = [index] + [
                int(np.random.randint(0, len(dataset))) for _ in range(8)
            ]
            items = [dataset.load_raw(i) for i in idxs]
            img, labels, kpts, masks = mosaic9(items, self.affine.imgsz, self.box_format)
        elif use_mosaic:
            idxs = [index] + [
                int(np.random.randint(0, len(dataset))) for _ in range(3)
            ]
            items = [dataset.load_raw(i) for i in idxs]
            img, labels, kpts, masks = mosaic4(items, self.affine.imgsz, self.box_format,
                                               crop=self.mosaic_crop)
        else:
            img, labels, kpts, masks = dataset.load_raw(index)

        if self.copy_paste_p > 0 and np.random.rand() < self.copy_paste_p:
            if masks is not None and len(masks):
                j = int(np.random.randint(0, len(dataset)))
                simg, slabels, _, smasks = dataset.load_raw(j)
                img, labels, masks = copy_paste(
                    img,
                    labels,
                    masks,
                    simg,
                    slabels,
                    smasks,
                    box_format=self.box_format,
                    p=self.copy_paste_p,
                )
                if kpts is not None and len(kpts):
                    kpts = kpts[: len(labels)]

        if self.mixup_p > 0 and np.random.rand() < self.mixup_p:
            j = int(np.random.randint(0, len(dataset)))
            if use_mosaic:
                items2 = [dataset.load_raw(j)] + [
                    dataset.load_raw(int(np.random.randint(0, len(dataset))))
                    for _ in range(3)
                ]
                img2, l2, k2, m2 = mosaic4(items2, self.affine.imgsz, self.box_format)
            else:
                img2, l2, k2, m2 = mosaic4(
                    [dataset.load_raw(j)] * 4, self.affine.imgsz, self.box_format
                )
            img, labels, kpts, masks = mixup(img, labels, kpts, masks, img2, l2, k2, m2)

        img, labels, kpts, masks = self.affine(
            img, labels, kpts, masks, box_format=self.box_format
        )
        if self.flipud and np.random.rand() < self.flipud:
            img = np.ascontiguousarray(img[::-1])
            hgt = img.shape[0]
            if labels is not None and len(labels):
                lb = labels.copy()
                if self.box_format == "xywhr":
                    lb[:, 2] = hgt - lb[:, 2]
                    lb[:, 5] = -lb[:, 5]
                else:
                    y1 = lb[:, 2].copy()
                    lb[:, 2] = hgt - lb[:, 4]
                    lb[:, 4] = hgt - y1
                labels = lb
            if kpts is not None and len(kpts):
                kk = kpts.copy()
                kk[..., 1] = hgt - kk[..., 1]
                kpts = kk
            if masks is not None and len(masks):
                masks = np.ascontiguousarray(masks[:, ::-1, :])
        if self.fliplr and np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(img[:, ::-1])
            wid = img.shape[1]
            if labels is not None and len(labels):
                lb = labels.copy()
                if self.box_format == "xywhr":
                    lb[:, 1] = wid - lb[:, 1]
                    lb[:, 5] = np.pi - lb[:, 5]
                else:
                    x1 = lb[:, 1].copy()
                    lb[:, 1] = wid - lb[:, 3]
                    lb[:, 3] = wid - x1
                labels = lb
            if kpts is not None and len(kpts):
                kk = kpts.copy()
                kk[..., 0] = wid - kk[..., 0]
                kpts = kk
            if masks is not None and len(masks):
                masks = np.ascontiguousarray(masks[:, :, ::-1])
        if self.hsv is not None:
            img = self.hsv(img)
        return img, labels, kpts, masks


class DenseAugmenter(object):
    """Affine + HSV for dense targets (semantic mask / depth map), image-only HSV."""

    def __init__(self, cfg, imgsz, multi_scale=0.0):
        cfg = dict(cfg or {})
        self.imgsz = int(imgsz)
        self.multi_scale = float(cfg.get("multi_scale", multi_scale))
        hsv = cfg.get("hsv")
        self.hsv = RandomHSV(**(hsv if isinstance(hsv, dict) else {})) if hsv else None
        affine = cfg.get("affine")
        self.affine = RandomAffine(
            self.imgsz, **(affine if isinstance(affine, dict) else {})
        )

    def set_epoch(self, epoch):
        if self.multi_scale:
            lo = max(64, int(self.imgsz * (1 - self.multi_scale)))
            hi = int(self.imgsz * (1 + self.multi_scale))
            choices = list(range(lo, hi + 1, 32)) or [self.imgsz]
            self.affine.imgsz = int(np.random.choice(choices))

    def __call__(self, img, dense):
        h, w = img.shape[:2]
        M, _ = self.affine._matrix(w, h)
        S = self.affine.imgsz
        img = cv2.warpPerspective(
            img, M, (S, S), flags=cv2.INTER_LINEAR,
            borderValue=(self.affine.border_value,) * 3,
        )
        if dense is not None:
            dense = cv2.warpPerspective(dense, M, (S, S), flags=cv2.INTER_NEAREST)
        if self.hsv is not None:
            img = self.hsv(img)
        return img, dense
