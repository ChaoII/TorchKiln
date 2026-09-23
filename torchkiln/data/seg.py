"""Instance segmentation data pipeline (YOLO-seg polygon labels).

Layout::

    <data_dir>/images/<split>/xxx.jpg
    <data_dir>/labels/<split>/xxx.txt    # "cls x1 y1 ... xn yn" normalised
    <list.txt>

Each instance yields a box (min/max of its polygon) and a rasterised binary mask
at ``imgsz / mask_stride`` resolution (default stride 4).
"""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from torchkiln.det.ops import letterbox

__all__ = ["SegDataset", "train_collate", "eval_collate"]


def _label_path_for(img_rel):
    parts = img_rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return os.path.splitext("/".join(parts))[0] + ".txt"


class SegDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.imgsz = int((ds_cfg.get("transform") or {}).get("image_size", 640))
        self.mask_stride = int((ds_cfg.get("transform") or {}).get("mask_stride", 4))
        self.augmenter = None
        if mode == "Train" and ds_cfg.get("augment") is not None:
            from torchkiln.data.augment import TrainAugmenter

            aug_cfg = dict(ds_cfg["augment"] or {})
            aug_cfg.setdefault(
                "total_epochs", int((config.get("Global") or {}).get("epoch_num", 0) or 0)
            )
            self.augmenter = TrainAugmenter(aug_cfg, self.imgsz)
        from torchkiln.data.cache import LabelCache

        self._label_cache = LabelCache(
            self.data_dir, ds_cfg.get("label_file_list"),
            split=mode, logger=logger, enabled=bool(ds_cfg.get("labels_cache", True)),
        )
        self.img_files = []
        for label_file in ds_cfg["label_file_list"]:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.img_files.append(line)
        if logger is not None:
            logger.info(
                "%s dataset: %d images (data_dir=%s, imgsz=%d, mask_stride=%d)",
                mode,
                len(self.img_files),
                self.data_dir,
                self.imgsz,
                self.mask_stride,
            )

    def __len__(self):
        return len(self.img_files)

    def _load_polys(self, img_rel, w, h, ratio=1.0, pad=(0, 0)):
        path = os.path.join(self.data_dir, _label_path_for(img_rel))
        out = []
        if not os.path.isfile(path):
            return out
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                cls = float(parts[0])
                coords = np.array(parts[1:], dtype=np.float64)
                if coords.size % 2 != 0:
                    coords = coords[:-1]
                pts = coords.reshape(-1, 2)
                pts[:, 0] = pts[:, 0] * w * ratio + pad[0]
                pts[:, 1] = pts[:, 1] * h * ratio + pad[1]
                out.append((cls, pts.astype(np.float32)))
        return out

    def set_epoch(self, epoch):
        if self.augmenter is not None:
            self.augmenter.set_epoch(epoch)
        cache = getattr(self, "_label_cache", None)
        if cache is not None and getattr(cache, "_dirty", False):
            cache.save_now()

    def load_raw(self, index):
        """原图 + 原始像素坐标标签 + 原始比例掩码(供 ``TrainAugmenter`` 使用)。

        返回 ``(img, labels(N,5) [cls,x1,y1,x2,y2], kpts=None, masks(N,h,w))``,
        坐标系为原图像素;mosaic/affine 会按同一 r 缩放图像与掩码。
        """
        rel = self.img_files[index]
        img = cv2.imread(os.path.join(self.data_dir, rel))
        if img is None:
            return None
        h0, w0 = img.shape[:2]
        polys = self._load_polys(rel, w0, h0)
        labels = np.zeros((len(polys), 5), dtype=np.float32)
        # 掩码用原图分辨率:augmenter 的 affine/mosaic 都按图像像素网格变换,
        # 若这里用 imgsz/mask_stride 会与透视矩阵尺度不一致。
        mh, mw = max(1, h0), max(1, w0)
        masks = np.zeros((len(polys), mh, mw), dtype=np.float32)
        for i, (cls, pts) in enumerate(polys):
            labels[i] = [
                cls,
                pts[:, 0].min(),
                pts[:, 1].min(),
                pts[:, 0].max(),
                pts[:, 1].max(),
            ]
            canvas = np.zeros((mh, mw), dtype=np.uint8)
            cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
            masks[i] = canvas
        return img, labels, None, masks

    def __getitem__(self, index):
        if self.augmenter is not None:
            img, boxes, _, masks = self.augmenter(self, index)
            if boxes is None:
                boxes = np.zeros((0, 5), np.float32)
            mh = max(1, self.augmenter.affine.imgsz // self.mask_stride)
            if masks is not None and len(masks):
                small = np.stack(
                    [
                        cv2.resize(
                            m.astype(np.uint8), (mh, mh), interpolation=cv2.INTER_NEAREST
                        )
                        for m in masks
                    ]
                ).astype(np.uint8)
            else:
                small = np.zeros((0, mh, mh), np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            return [
                img,
                boxes.astype(np.float32),
                np.ones((boxes.shape[0],), np.float32),
                small,
            ]

        rel = self.img_files[index]
        img = cv2.imread(os.path.join(self.data_dir, rel))
        if img is None:
            return []
        h0, w0 = img.shape[:2]
        img, ratio, pad = letterbox(img, self.imgsz)
        H, W = img.shape[:2]
        polys = self._load_polys(rel, w0, h0, ratio, pad)
        targets = np.zeros((len(polys), 5), dtype=np.float32)
        valid = np.zeros((len(polys),), dtype=np.float32)

        if self.mode == "Eval":
            # 评估掩码用原图分辨率(对齐原版 Validator:pred 由 process_mask 上采样到 640);
            # 实例索引图保存,省内存。
            mh, mw = max(1, H), max(1, W)
            inst = np.zeros((mh, mw), dtype=np.int32)
            for i, (cls, pts) in enumerate(polys):
                targets[i] = [
                    cls,
                    pts[:, 0].min(),
                    pts[:, 1].min(),
                    pts[:, 0].max(),
                    pts[:, 1].max(),
                ]
                cv2.fillPoly(inst, [np.round(pts).astype(np.int32)], i + 1)
                valid[i] = 1.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            return [img, targets, valid, inst]

        mh, mw = max(1, H // self.mask_stride), max(1, W // self.mask_stride)
        masks = np.zeros((len(polys), mh, mw), dtype=np.uint8)
        for i, (cls, pts) in enumerate(polys):
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            targets[i] = [cls, x1, y1, x2, y2]
            canvas = np.zeros((mh, mw), dtype=np.uint8)
            p = np.round(pts / self.mask_stride).astype(np.int32)
            cv2.fillPoly(canvas, [p], 1)
            masks[i] = canvas
            valid[i] = 1.0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return [img, targets, valid, masks]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(np.stack([s[0] for s in batch], axis=0)).float()
    mh, mw = batch[0][3].shape[-2:]
    max_gt = max(1, max(s[1].shape[0] for s in batch))
    targets = np.zeros((len(batch), max_gt, 5), dtype=np.float32)
    valid = np.zeros((len(batch), max_gt), dtype=np.float32)
    masks = np.zeros((len(batch), max_gt, mh, mw), dtype=np.float32)
    for i, s in enumerate(batch):
        n = s[1].shape[0]
        if n:
            targets[i, :n] = s[1]
            valid[i, :n] = s[2]
            masks[i, :n] = s[3]
    return [
        images,
        torch.from_numpy(targets).float(),
        torch.from_numpy(valid).float(),
        torch.from_numpy(masks).float(),
    ]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    """评估:掩码以 ``(H, W)`` 实例索引图保存(原图分辨率),避免 (N,H,W) 开销。"""
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    images = torch.from_numpy(np.stack([s[0] for s in batch], axis=0)).float()
    max_gt = max(1, max(s[1].shape[0] for s in batch))
    targets = np.zeros((len(batch), max_gt, 5), dtype=np.float32)
    valid = np.zeros((len(batch), max_gt), dtype=np.float32)
    maps = np.stack([s[3] for s in batch], axis=0).astype(np.int64)
    for i, s in enumerate(batch):
        n = s[1].shape[0]
        if n:
            targets[i, :n] = s[1]
            valid[i, :n] = s[2]
    return [
        images,
        torch.from_numpy(targets).float(),
        torch.from_numpy(valid).float(),
        torch.from_numpy(maps),
    ]
