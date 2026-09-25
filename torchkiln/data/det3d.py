"""3D detection data: cloud + boxes, pillarized to dense BEV (CenterPoint-style).

Layout::

    <data_dir>/train.txt
    <data_dir>/val.txt
    <data_dir>/clouds/<split>/*.bin|.npy   # float32 N×3/4
    <data_dir>/labels/<split>/*.txt        # one box per line:
        # cls  x y z  l w h  yaw   (LiDAR; cls = name or int; optional difficulty)

``__getitem__`` returns::

    [bev (C, Ny, Nx), labels (M, 8) float32 [cls,x,y,z,l,w,h,yaw], mask (M,) float32]
"""
from __future__ import absolute_import

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from torchkiln.data.pc import (
    _as_pillar,
    _as_range,
    load_cloud,
    pillarize,
)

__all__ = ["Det3DDataset", "load_box_labels", "augment_boxes",
           "train_collate", "eval_collate", "raw_collate"]


def augment_boxes(pc, labels, aug, rng):
    """In-place-safe global augmentation for ``(pc, labels)``.

    ``aug`` keys (all optional)::

        rot:        max |yaw| radians for a global rotation around z
        scale:      [lo, hi] uniform global xy-scale
        translate:  [sx, sy, sz] gaussian std (metres)
        flip_y:     probability of mirroring across the xz plane
    """
    pc = np.array(pc, dtype=np.float32, copy=True)
    if labels is not None:
        labels = np.array(labels, dtype=np.float32, copy=True)
    has_lab = labels is not None and labels.shape[0] > 0

    rot = aug.get("rot")
    if rot:
        th = float(rng.uniform(-rot, rot))
        c, s = np.cos(th), np.sin(th)
        x, y = pc[:, 0].copy(), pc[:, 1].copy()
        pc[:, 0] = x * c - y * s
        pc[:, 1] = x * s + y * c
        if has_lab:
            lx, ly = labels[:, 1].copy(), labels[:, 2].copy()
            labels[:, 1] = lx * c - ly * s
            labels[:, 2] = lx * s + ly * c
            labels[:, 7] = labels[:, 7] + th

    scale = aug.get("scale")
    if scale:
        sc = float(rng.uniform(scale[0], scale[1]))
        pc[:, :3] *= sc
        if has_lab:
            labels[:, 1:4] *= sc
            labels[:, 4:7] *= sc

    tr = aug.get("translate")
    if tr:
        t = rng.normal(0.0, 1.0, size=3) * np.asarray(tr, dtype=np.float32)
        pc[:, :3] += t
        if has_lab:
            labels[:, 1:4] += t

    if aug.get("flip_y") and rng.random() < float(aug["flip_y"]):
        pc[:, 1] *= -1.0
        if has_lab:
            labels[:, 2] *= -1.0
            labels[:, 7] = -labels[:, 7]

    return pc, labels



def load_box_labels(path, names=None):
    """Parse KITTI-simplified box lines -> ``(M, 8) float32`` [cls,x,y,z,l,w,h,yaw].

    Unknown class names map via ``names`` (list or dict) else via a built-in
    order of first-seen names at parse time is avoided — prefer ``names``.
    """
    if path is None or not os.path.isfile(path):
        return np.zeros((0, 8), dtype=np.float32)
    name_to_id = {}
    if names:
        if isinstance(names, dict):
            name_to_id = {str(k): int(v) for k, v in names.items()}
        else:
            name_to_id = {str(n): i for i, n in enumerate(names)}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 8:
                continue
            try:
                cls_tok = parts[0]
                if cls_tok in name_to_id:
                    cid = name_to_id[cls_tok]
                else:
                    cid = int(float(cls_tok))
                vals = [float(t) for t in parts[1:8]]
            except ValueError:
                continue
            rows.append([cid] + vals)
    if not rows:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _label_path_for(rel):
    parts = rel.replace("\\", "/").split("/")
    stem = os.path.splitext(parts[-1])[0]
    if "clouds" in parts:
        parts[parts.index("clouds")] = "labels"
        base = os.path.dirname("/".join(parts))
        if base:
            return os.path.join(base, stem + ".txt")
        return stem + ".txt"
    d = os.path.dirname(rel)
    return os.path.join(d, stem + ".txt") if d else stem + ".txt"


class Det3DDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.pc_range = ds_cfg.get("pc_range")
        self.pillar_size = ds_cfg.get("pillar_size")
        self.names = ds_cfg.get("names")
        self.num_classes = ds_cfg.get("num_classes")
        self.ignore_index = int(ds_cfg.get("ignore_index", 255))
        self.max_boxes = int(ds_cfg.get("max_boxes", 64))
        self.augment = ds_cfg.get("augment") if mode == "Train" else None
        # raw-points mode (CenterPoint): return the raw cloud instead of a BEV.
        self.raw_points = bool(ds_cfg.get("raw_points", False))
        self.point_cloud_range = ds_cfg.get("point_cloud_range") or ds_cfg.get("pc_range")
        self.img_files = []
        for label_file in ds_cfg["label_file_list"]:
            lf = label_file
            if not os.path.isabs(lf) and not os.path.isfile(lf):
                lf = os.path.join(self.data_dir, label_file)
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.img_files.append(line)
        if logger is not None:
            logger.info(
                "%s det3d dataset: %d clouds (data_dir=%s)",
                mode,
                len(self.img_files),
                self.data_dir,
            )

    def __len__(self):
        return len(self.img_files)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        rel = self.img_files[index]
        cloud_path = os.path.join(self.data_dir, rel)
        try:
            pc = load_cloud(cloud_path)
        except Exception:
            return []
        lab_rel = _label_path_for(rel)
        lab_path = os.path.join(self.data_dir, lab_rel)
        if not os.path.isfile(lab_path):
            lab_path = lab_rel
        labels = load_box_labels(lab_path, names=self.names)
        if self.augment:
            pc, labels = augment_boxes(pc, labels, self.augment, np.random)
        if self.raw_points:
            if labels.shape[0] > self.max_boxes:
                vol = labels[:, 4] * labels[:, 5] * labels[:, 6]
                labels = labels[np.argsort(-vol)[: self.max_boxes]]
            rng = self.point_cloud_range
            if rng is not None:
                v = list(rng)
                m = ((pc[:, 0] >= v[0]) & (pc[:, 0] < v[3])
                     & (pc[:, 1] >= v[1]) & (pc[:, 1] < v[4])
                     & (pc[:, 2] >= v[2]) & (pc[:, 2] < v[5]))
                pc = pc[m]
            mask = np.ones((labels.shape[0],), dtype=np.float32)
            return [pc.astype(np.float32), labels.astype(np.float32), mask]
        # pillarize empty label map (ignore everywhere) for BEV features only
        bev, _ = pillarize(
            pc,
            labels=None,
            pc_range=self.pc_range,
            pillar_size=self.pillar_size,
            num_classes=1,
            ignore_index=self.ignore_index,
        )
        if labels.shape[0] > self.max_boxes:
            # keep highest-volume boxes first (stable for tiny demos)
            vol = labels[:, 4] * labels[:, 5] * labels[:, 6]
            order = np.argsort(-vol)[: self.max_boxes]
            labels = labels[order]
        mask = np.ones((labels.shape[0],), dtype=np.float32)
        return [bev, labels.astype(np.float32), mask]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    bevs = [s[0] for s in batch]
    h0, w0 = bevs[0].shape[-2:]
    keep = [s for s in batch if s[0].shape[-2:] == (h0, w0)]
    if not keep:
        return []
    max_gt = max(int(s[1].shape[0]) for s in keep)
    max_gt = max(max_gt, 1)
    targets = np.zeros((len(keep), max_gt, 8), dtype=np.float32)
    masks = np.zeros((len(keep), max_gt), dtype=np.float32)
    for i, s in enumerate(keep):
        n = s[1].shape[0]
        if n:
            targets[i, :n] = s[1]
            masks[i, :n] = s[2]
    bev_t = torch.from_numpy(np.stack([s[0] for s in keep], axis=0)).float()
    return [
        bev_t,
        torch.from_numpy(targets).float(),
        torch.from_numpy(masks).float(),
    ]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)


def raw_collate(batch):
    """Collate raw clouds into padded tensors ``(B, N, C)`` (sentinel padding)."""
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    b = len(batch)
    c = batch[0][0].shape[1]
    max_n = max(s[0].shape[0] for s in batch)
    max_m = max(1, max(int(s[1].shape[0]) for s in batch))
    pts = np.full((b, max_n, c), -1e4, dtype=np.float32)
    tgt = np.zeros((b, max_m, 8), dtype=np.float32)
    msk = np.zeros((b, max_m), dtype=np.float32)
    for i, s in enumerate(batch):
        n = s[0].shape[0]
        pts[i, :n] = s[0]
        m = s[1].shape[0]
        if m:
            tgt[i, :m] = s[1]
            msk[i, :m] = s[2]
    return [torch.from_numpy(pts), torch.from_numpy(tgt), torch.from_numpy(msk)]
