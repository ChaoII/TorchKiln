"""LiDAR point-cloud data pipeline (pillarize → dense BEV feature map).

Layout::

    <data_dir>/train.txt                 # list of cloud paths (relative)
    <data_dir>/val.txt
    <data_dir>/clouds/train/*.bin|.npy   # float32, N×3 (x,y,z) or N×4 (x,y,z,i)
    <data_dir>/labels/train/*.txt|.npy   # optional: N int labels (one per line / int array)

``__getitem__`` returns::

    [bev (C, Ny, Nx) float32, pillar_labels (Ny, Nx) int64]

``pillar_labels`` is the majority point label per non-empty pillar
(``ignore_index`` for empty cells). Loss/metric run at pillar resolution
(v0; point-level gather can be added later without changing the model).
"""
from __future__ import absolute_import

import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "PointCloudDataset",
    "pillarize",
    "train_collate",
    "eval_collate",
]


def _as_range(pc_range):
    """Accept ``[xmin,ymin,zmin,xmax,ymax,zmax]`` or ``[[xmin,xmax],...]``.

    Flat 6-tuple follows the common LiDAR convention (mins first), matching
    the design doc ``pc_range: [-40, -40, -3, 40, 40, 1]``.
    """
    if pc_range is None:
        return (-40.0, 40.0, -40.0, 40.0, -3.0, 1.0)
    if isinstance(pc_range[0], (list, tuple)):
        (x0, x1), (y0, y1), (z0, z1) = pc_range
        return (float(x0), float(x1), float(y0), float(y1), float(z0), float(z1))
    v = [float(t) for t in pc_range]
    if len(v) != 6:
        raise ValueError("pc_range must have 6 numbers, got {}".format(pc_range))
    # flat: xmin ymin zmin xmax ymax zmax  ->  (xmin, xmax, ymin, ymax, zmin, zmax)
    return (v[0], v[3], v[1], v[4], v[2], v[5])


def _as_pillar(pillar_size):
    if pillar_size is None:
        return (0.5, 0.5, 4.0)
    if isinstance(pillar_size, (list, tuple)):
        ps = [float(t) for t in pillar_size]
        if len(ps) == 2:
            ps = [ps[0], ps[1], 4.0]
        if len(ps) != 3:
            raise ValueError("pillar_size must be 2 or 3 numbers")
        return tuple(ps)
    p = float(pillar_size)
    return (p, p, 4.0)


def load_cloud(path):
    """Load N×3/4 float32 cloud from ``.npy`` or KITTI-style ``.bin``."""
    if path.endswith(".npy"):
        pc = np.load(path).astype(np.float32)
    else:
        pc = np.fromfile(path, dtype=np.float32)
        if pc.size % 4 == 0:
            pc = pc.reshape(-1, 4)
        elif pc.size % 3 == 0:
            pc = pc.reshape(-1, 3)
        else:
            raise ValueError("cannot reshape .bin with {} floats: {}".format(pc.size, path))
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError("cloud must be N×3+, got {} from {}".format(pc.shape, path))
    return pc


def load_point_labels(path, n_points):
    """Load per-point int labels; missing file -> zeros (single class)."""
    if path is None or not os.path.isfile(path):
        return np.zeros(n_points, dtype=np.int64)
    if path.endswith(".npy"):
        lab = np.load(path).reshape(-1).astype(np.int64)
    else:
        with open(path, "r", encoding="utf-8") as f:
            rows = [ln.strip() for ln in f if ln.strip()]
        if rows and " " in rows[0]:
            # "idx label" or "label ..." — take 2nd field if present else 1st
            lab = []
            for ln in rows:
                parts = ln.split()
                lab.append(int(parts[1]) if len(parts) > 1 else int(parts[0]))
            lab = np.asarray(lab, dtype=np.int64)
        else:
            lab = np.asarray([int(float(x)) for x in rows], dtype=np.int64)
    if lab.size < n_points:
        lab = np.pad(lab, (0, n_points - lab.size), constant_values=0)
    elif lab.size > n_points:
        lab = lab[:n_points]
    return lab


def pillarize(
    pc,
    labels=None,
    pc_range=None,
    pillar_size=None,
    num_classes=None,
    ignore_index=255,
    max_points_per_pillar=None,
):
    """Scatter a point cloud into a dense BEV grid.

    Returns
    -------
    bev : (C, Ny, Nx) float32
        Per-pillar mean of ``[x, y, z, (i)]`` (C=3 or 4).
    pillar_labels : (Ny, Nx) int64
        Majority label of points in each pillar; empty -> ``ignore_index``.
    """
    xr = _as_range(pc_range)
    ps = _as_pillar(pillar_size)
    xmin, xmax, ymin, ymax, zmin, zmax = xr
    px, py, _ = ps
    nx = max(int(np.ceil((xmax - xmin) / px)), 1)
    ny = max(int(np.ceil((ymax - ymin) / py)), 1)

    pc = np.asarray(pc, dtype=np.float32)
    if not np.isfinite(pc).all():
        pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    intensity = pc[:, 3] if pc.shape[1] >= 4 else None

    m = (
        (x >= xmin)
        & (x < xmax)
        & (y >= ymin)
        & (y < ymax)
        & (z >= zmin)
        & (z < zmax)
    )
    x, y, z = x[m], y[m], z[m]
    if intensity is not None:
        intensity = intensity[m]
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)[m]

    has_i = intensity is not None
    c = 4 if has_i else 3
    bev = np.zeros((c, ny, nx), dtype=np.float32)
    if x.size == 0:
        plabels = np.full((ny, nx), ignore_index, dtype=np.int64)
        return bev, plabels

    gx = np.clip(((x - xmin) / px).astype(np.int64), 0, nx - 1)
    gy = np.clip(((y - ymin) / py).astype(np.int64), 0, ny - 1)
    flat = gy * nx + gx  # (N,)

    # mean-pool features per pillar via bincount
    feats = [x, y, z] + ([intensity] if has_i else [])
    counts = np.bincount(flat, minlength=ny * nx).astype(np.float32)
    counts_safe = np.maximum(counts, 1.0)
    for ci, ch in enumerate(feats):
        bev[ci] = (
            np.bincount(flat, weights=ch, minlength=ny * nx) / counts_safe
        ).reshape(ny, nx)

    # majority label per pillar
    plabels = np.full(ny * nx, ignore_index, dtype=np.int64)
    if labels is None:
        nonempty = counts > 0
        plabels[nonempty] = 0
    else:
        if num_classes is None:
            num_classes = int(labels.max()) + 1 if labels.size else 1
        # pack (class, cell) then sort & run-length majority
        # memory-safe path: loop unique classes with bincount-like accumulate
        nc = max(int(num_classes), 1)
        votes = np.zeros((nc, ny * nx), dtype=np.int32)
        li = np.clip(labels, 0, nc - 1)
        np.add.at(votes, (li, flat), 1)
        if max_points_per_pillar:
            pass  # reserved; majority already correct without cap
        best = votes.argmax(axis=0)
        best_cnt = votes.max(axis=0)
        plabels = np.where(best_cnt > 0, best, ignore_index).astype(np.int64)

    return bev, plabels.reshape(ny, nx)


def project_range_image(pc, labels, h, w, num_classes, ignore_index,
                        fov_up=3.0, fov_down=-25.0):
    """Spherical projection to a ``(5, H, W)`` range image + ``(H, W)`` labels.

    Channels: ``[range, x, y, z, remission]`` (Paddle3D SqueezeSegV3 order).
    Nearest point wins when several hit the same pixel.
    """
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    depth = np.sqrt(x * x + y * y + z * z)
    keep = depth > 1e-3
    if not keep.any():
        return (np.zeros((5, h, w), np.float32),
                np.full((h, w), ignore_index, np.int64))
    x, y, z, depth = x[keep], y[keep], z[keep], depth[keep]
    rem = pc[keep, 3] if pc.shape[1] >= 4 else np.zeros_like(depth)
    lab = labels[keep] if labels is not None else np.zeros_like(depth, np.int64)

    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / depth, -1.0, 1.0))
    fov = (abs(fov_up) + abs(fov_down)) * np.pi / 180.0
    px = np.clip((0.5 * (yaw / np.pi + 1.0) * w).astype(np.int64), 0, w - 1)
    py = np.clip(((1.0 - (pitch + abs(fov_down) * np.pi / 180.0) / fov) * h
                  ).astype(np.int64), 0, h - 1)
    flat = py * w + px

    order = np.argsort(-depth)  # far -> near; near written last (wins)
    rng_img = np.zeros(h * w, np.float32)
    chans = [np.zeros(h * w, np.float32) for _ in range(4)]
    lab_img = np.full(h * w, ignore_index, np.int64)
    fs = flat[order]
    rng_img[fs] = depth[order]
    chans[0][fs] = x[order]
    chans[1][fs] = y[order]
    chans[2][fs] = z[order]
    chans[3][fs] = rem[order]
    lab_img[fs] = lab[order] if lab is not None else 0
    img = np.stack([rng_img, chans[0], chans[1], chans[2], chans[3]], 0)
    return img.reshape(5, h, w), lab_img.reshape(h, w)


class PointCloudDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.pc_range = ds_cfg.get("pc_range")
        self.pillar_size = ds_cfg.get("pillar_size")
        self.num_classes = ds_cfg.get("num_classes")
        self.ignore_index = int(ds_cfg.get("ignore_index", 255))
        self.range_image = bool(ds_cfg.get("range_image", False))
        self.img_h = int(ds_cfg.get("range_image_h", 64))
        self.img_w = int(ds_cfg.get("range_image_w", 2048))
        self.fov = ds_cfg.get("fov", [3.0, -25.0])
        self.max_points = ds_cfg.get("max_points_per_pillar")
        self.labels_dir = ds_cfg.get("labels_dir")  # default: labels/<split>
        self.clouds_dir = ds_cfg.get("clouds_dir")  # default: clouds/<split>
        self.img_files = []
        for label_file in ds_cfg["label_file_list"]:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.img_files.append(line)
        if logger is not None:
            logger.info(
                "%s pc_seg dataset: %d clouds (data_dir=%s)",
                mode,
                len(self.img_files),
                self.data_dir,
            )

    def __len__(self):
        return len(self.img_files)

    def set_epoch(self, epoch):
        pass

    def _label_path(self, rel):
        parts = rel.replace("\\", "/").split("/")
        stem = os.path.splitext(parts[-1])[0]
        if "clouds" in parts:
            parts[parts.index("clouds")] = "labels"
            base = os.path.dirname("/".join(parts))
            if base:
                return os.path.join(base, stem + ".txt"), os.path.join(
                    base, stem + ".npy"
                )
            return stem + ".txt", stem + ".npy"
        # flat: labels/<stem>.txt next to cloud
        d = os.path.dirname(rel)
        return (
            os.path.join(d, stem + ".txt") if d else stem + ".txt",
            os.path.join(d, stem + ".npy") if d else stem + ".npy",
        )

    def __getitem__(self, index):
        rel = self.img_files[index]
        cloud_path = os.path.join(self.data_dir, rel)
        try:
            pc = load_cloud(cloud_path)
        except Exception:
            return []
        t_txt, t_npy = self._label_path(rel)
        lab_path = None
        for cand in (
            os.path.join(self.data_dir, t_txt),
            os.path.join(self.data_dir, t_npy),
            t_txt,
            t_npy,
        ):
            if os.path.isfile(cand):
                lab_path = cand
                break
        labels = load_point_labels(lab_path, pc.shape[0])
        if self.range_image:
            img, lab_img = project_range_image(
                pc, labels, self.img_h, self.img_w, self.num_classes,
                self.ignore_index, self.fov[0], self.fov[1])
            return [img, lab_img]
        bev, plabels = pillarize(
            pc,
            labels,
            pc_range=self.pc_range,
            pillar_size=self.pillar_size,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            max_points_per_pillar=self.max_points,
        )
        if bev.sum() == 0 and (plabels == self.ignore_index).all():
            # degenerate empty cloud — still return zeros so batch size holds
            pass
        return [bev, plabels]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    # pillar grids are fixed size for a given config → safe to stack
    bevs = [s[0] for s in batch]
    labs = [s[1] for s in batch]
    # guard rare grid mismatch (should not happen): resize skip → drop
    h0, w0 = bevs[0].shape[-2:]
    keep = [
        (b, l)
        for b, l in zip(bevs, labs)
        if b.shape[-2:] == (h0, w0) and l.shape == (h0, w0)
    ]
    if not keep:
        return []
    bevs = [b for b, _ in keep]
    labs = [l for _, l in keep]
    bev_t = torch.from_numpy(np.stack(bevs, axis=0)).float()
    lab_t = torch.from_numpy(np.stack(labs, axis=0)).long()
    return [bev_t, lab_t]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)
