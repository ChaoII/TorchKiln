"""Attribute recognition task (multi-label classification).

Implements the PaddleX/PaddleClas "pedestrian / vehicle attribute recognition"
modules::

    pytorchx/attr.py  AttributeDataset | MultiLabelLoss | AttrMetric |
                         MultiLabelThresPostProcess | AttributeTask

Differences between the two shipped models are only ``num_classes`` and the
image size, so one code path serves both:

    pedestrian : 26 attributes, 256x192  (configs/attr/pedestrian_attribute.yml)
    vehicle    : 19 attributes, 192x256  (configs/attr/vehicle_attribute.yml)
"""
from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ptcore.task import TaskAdapter
from pytorchx.data.augment import RandomErasing

__all__ = [
    "AttributeDataset",
    "MultiLabelLoss",
    "AttrMetric",
    "MultiLabelThresPostProcess",
    "AttributeTask",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ratio2weight",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ratio2weight(targets, ratio):
    """PaddleClas ``ppcls/loss/multilabelloss.py::ratio2weight``.

    ``exp((1 - t) * r + t * (1 - r))`` with ``t`` the 0/1 target mask and ``r``
    the dataset-level positive ratio of every attribute.
    """
    pos_weights = targets * (1.0 - ratio)
    neg_weights = (1.0 - targets) * ratio
    return torch.exp(neg_weights + pos_weights)


def _load_label_list(path):
    names = []
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.append(line)
    return names


def _label_path_for(img_rel):
    parts = img_rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return "/".join(parts)


class AttributeDataset(Dataset):
    """Multi-label dataset.

    Label line (PaddleClas ``MultiLabelDataset``)::

        relative/path.jpg  v1 v2 ... vC     # multi-hot or soft ratios

    With ``label_ratio: true`` (default, as in the PaddleX configs) every sample
    carries the *dataset level* positive ratio so the loss can re-weight classes.
    """

    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg.get("data_dir", "")
        self.label_ratio = bool(ds_cfg.get("label_ratio", True))
        tf = ds_cfg.get("transform") or {}
        size = tf.get("image_size", [256, 192])
        self.w, self.h = (int(size[0]), int(size[1])) if isinstance(size, (list, tuple)) else (192, 256)
        self.resize = tf.get("resize")
        self.crop_pad = tf.get("crop_pad")
        self.train = mode == "Train"
        aug = ds_cfg.get("augment") or {}
        self.flip = bool(aug.get("flip", self.train))
        erasing = aug.get("erasing")
        self.erasing = None
        if self.train and erasing:
            spec = erasing if isinstance(erasing, dict) else {}
            self.erasing = RandomErasing(p=float(spec.get("p", 0.4)))
        self.randaug = None
        if self.train and aug.get("randaugment"):
            from pytorchx.attr_randaug import RandAugment

            self.randaug = RandAugment(**(aug["randaugment"] if isinstance(aug["randaugment"], dict) else {}))

        self.samples = []
        for label_file in ds_cfg["label_file_list"]:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    self.samples.append((parts[0], np.array([float(v) for v in parts[1:]], np.float32)))
        self.num_classes = max((len(lb) for _, lb in self.samples), default=0)
        ratio = np.zeros(self.num_classes, np.float32)
        if self.samples:
            for _, lb in self.samples:
                ratio += (lb > 0.5).astype(np.float32)
            ratio = np.maximum(ratio / len(self.samples), 1e-6)
        self.ratio = ratio
        if logger is not None:
            logger.info(
                "%s dataset: %d samples, %d attributes (data_dir=%s, label_ratio=%s)",
                mode, len(self.samples), self.num_classes, self.data_dir, self.label_ratio,
            )

    def __len__(self):
        return len(self.samples)

    def _path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.data_dir, rel)

    def _image(self, img):
        sw, sh = (self.resize or (self.w, self.h))
        img = cv2.resize(img, (int(sw), int(sh)), interpolation=cv2.INTER_LINEAR)
        if self.train and self.crop_pad:
            pw, ph = self.crop_pad
            pad = np.full((int(ph), int(pw), 3), 0, np.uint8)
            pad[: img.shape[0], : img.shape[1]] = img
            y0 = np.random.randint(0, max(1, int(ph) - self.h + 1))
            x0 = np.random.randint(0, max(1, int(pw) - self.w + 1))
            img = pad[y0 : y0 + self.h, x0 : x0 + self.w]
        else:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if self.train and self.flip and np.random.rand() < 0.5:
            img = img[:, ::-1]
        return img

    def __getitem__(self, index):
        rel, label = self.samples[index]
        img = cv2.imread(self._path(rel))
        if img is None:
            return []
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.randaug is not None:
            img = self.randaug(img)
        img = self._image(img).astype(np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        if self.erasing is not None:
            img = self.erasing(img)
        return [img, label.astype(np.float32), self.ratio.astype(np.float32)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    c = batch[0][1].shape[0]
    images = torch.from_numpy(np.stack([s[0] for s in batch], axis=0)).float()
    labels = torch.zeros(len(batch), c)
    ratios = torch.zeros(len(batch), c)
    for i, s in enumerate(batch):
        labels[i, : len(s[1])] = torch.from_numpy(s[1])
        ratios[i, : len(s[2])] = torch.from_numpy(s[2])
    return [images, labels, ratios]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)


class MultiLabelLoss(nn.Module):
    """PaddleClas ``MultiLabelLoss`` (only the options used by the PaddleX configs).

    ``weight_ratio`` weights the BCE term per attribute with
    :func:`ratio2weight`; ``size_sum`` sums over attributes and averages over
    the batch (otherwise a plain mean); ``epsilon`` enables label smoothing.
    """

    def __init__(self, epsilon=None, size_sum=True, weight_ratio=True, **kwargs):
        super().__init__()
        self.epsilon = None if (epsilon is not None and not 0 < float(epsilon) < 1) else epsilon
        self.weight_ratio = bool(weight_ratio)
        self.size_sum = bool(size_sum)

    def forward(self, preds, batch):
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        labels, ratio = batch[1], batch[2]
        target = labels
        if self.epsilon is not None:
            eps = float(self.epsilon)
            target = target * (1.0 - eps) + (1.0 - target) * eps
        cost = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.weight_ratio:
            mask = (labels > 0.5).float()
            lr = ratio.mean(0).unsqueeze(0).expand_as(mask)
            cost = cost * ratio2weight(mask, lr)
        loss = cost.sum(1).mean() if self.size_sum else cost.mean()
        return {"loss": loss, "loss_bce": loss.detach()}


class AttrMetric(object):
    """``ATTRMetric``: per-attribute accuracy at threshold 0.5 -> mA (+ mAP)."""

    def __init__(self, main_indicator="mA", threshold=0.5, **kwargs):
        self.main_indicator = main_indicator
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self.correct = None
        self.total = None
        self.num = 0
        self._scores = []
        self._labels = []

    def __call__(self, post_result, batch):
        labels = batch[1].detach().cpu().numpy()
        scores = np.stack([r["scores"] for r in post_result], 0) if post_result else np.zeros_like(labels)
        c = labels.shape[1]
        if self.correct is None:
            self.correct = np.zeros(c)
            self.total = np.zeros(c)
        pred = (scores >= self.threshold).astype(np.float32)
        gt = (labels > 0.5).astype(np.float32)
        self.correct += (pred == gt).sum(0)
        self.total += len(labels)
        self.num += len(labels)
        self._scores.append(scores)
        self._labels.append(gt)

    def _map(self):
        if not self._scores:
            return 0.0
        s = np.concatenate(self._scores, 0)
        g = np.concatenate(self._labels, 0)
        aps = []
        for c in range(s.shape[1]):
            sc, gt = s[:, c], g[:, c]
            if gt.sum() == 0:
                continue
            order = np.argsort(-sc)
            gt = gt[order]
            tp = np.cumsum(gt)
            prec = tp / (np.arange(len(gt)) + 1.0)
            rec = tp / gt.sum()
            ap = float(np.sum(prec * gt) / gt.sum()) if gt.sum() else 0.0
            aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0

    def get_metric(self):
        if self.correct is None:
            return {"mA": 0.0, "mAP": 0.0, "num": 0}
        per_attr = self.correct / np.maximum(self.total, 1)
        return {
            "mA": float(per_attr.mean()),
            "mAP": self._map(),
            "num": self.num,
        }


class MultiLabelThresPostProcess(object):
    """``MultiLabelThresOutput``: sigmoid scores + names above the threshold."""

    def __init__(self, threshold=0.5, label_list=None, names=None, **kwargs):
        self.threshold = float(threshold)
        self.names = names or _load_label_list(label_list)

    def __call__(self, preds):
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        scores = torch.sigmoid(logits.detach())
        out = []
        for b in range(scores.shape[0]):
            sc = scores[b].cpu().numpy()
            hit = [int(i) for i in np.where(sc >= self.threshold)[0]]
            out.append(
                {
                    "scores": sc,
                    "labels": hit,
                    "attributes": [self.names[i] if i < len(self.names) else str(i) for i in hit],
                }
            )
        return out


def build_attribute_model(arch):
    from pytorchx.nn.attribute import AttributeNet

    head = (arch or {}).get("Head") or {}
    backbone = (arch or {}).get("Backbone") or {}
    return AttributeNet(
        num_classes=int(head.get("num_classes", 26)),
        scale=float(backbone.get("scale", 1.0)),
        class_expand=int(head.get("class_expand", 1280)),
        dropout_prob=float(head.get("dropout_prob", backbone.get("dropout_prob", 0.2))),
    )


class AttributeTask(TaskAdapter):
    """Multi-label attribute recognition (``Architecture.task: attribute``)."""

    name = "attribute"

    def build_post_process(self, config):
        cfg = dict(config.get("PostProcess") or {})
        cfg.pop("name", None)
        arch = config.get("Architecture") or {}
        head = arch.get("Head") or {}
        cfg.setdefault("label_list", head.get("label_list"))
        return MultiLabelThresPostProcess(**cfg)

    def build_model(self, config, post_process):
        return build_attribute_model(config["Architecture"])

    def build_loss(self, config, model=None):
        cfg = dict(config.get("Loss") or {})
        cfg.pop("name", None)
        cfg.pop("weight", None)
        return MultiLabelLoss(**cfg)

    def build_metric(self, config):
        cfg = dict(config.get("Metric") or {})
        cfg.pop("name", None)
        return AttrMetric(**cfg)

    def build_datasets(self, config, logger):
        train_ds = AttributeDataset(config, "Train", logger)
        eval_ds = AttributeDataset(config, "Eval", logger) if config.get("Eval") is not None else None
        return train_ds, eval_ds

    def train_collate(self, batch):
        return train_collate(batch)

    def eval_collate(self, batch):
        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        metric(post_process(model(images)), batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture") or {}
        head = arch.get("Head") or {}
        ds = (config.get("Train") or {}).get("dataset") or {}
        return [
            "task=attribute backbone={} num_classes={} image_size={} dropout={}".format(
                (arch.get("Backbone") or {}).get("name", "PPLCNet_x1_0"),
                head.get("num_classes"),
                (ds.get("transform") or {}).get("image_size"),
                head.get("dropout_prob"),
            )
        ]
