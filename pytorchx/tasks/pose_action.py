"""Skeleton-based action recognition (安防行为识别): ``task: pose_action``.

数据 = ``YOLO-pose`` 出的关键点序列(复用现有 pose 能力),模型 = ST-GCN
(空间图卷积 + 时序卷积,无 3D 卷积,算力友好),适合跌倒/打架/攀爬/挥手等行为。

标签文件(yolo 风格:``train.txt`` 每行一个样本)::

    path/to/seq_0001.npy  <action_id>

``.npy`` 形状 ``(T, V, C)`` 或 ``(C, T, V)``;``V`` 为关键点数(默认 17 = COCO/YOLO-pose,
支持 4 车牌角点等自定义);``C`` = 坐标维数(默认 2 = x,y)。

增广:随机时间裁剪/采样、水平翻转(左右关键点对称交换)、坐标抖动。
指标:top-1 / top-5(安防行为通常类别少,top-1 足够)。
"""

from __future__ import absolute_import, division

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ptcore.task import TaskAdapter

__all__ = ["STGCN", "PoseActionDataset", "PoseActionLoss", "PoseActionMetric",
           "PoseActionTask", "build_pose_action_model", "coco17_edges"]


# --------------------------------------------------------------------------- #
# 骨架拓扑(COCO 17 点)+ 三划分邻接矩阵(自环 + 向心 + 离心)
# --------------------------------------------------------------------------- #
def coco17_edges():
    return [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 6), (5, 11), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16)]


def _normalize_adjacency(V, edges):
    """3 个子集(自环 / 向心 / 离心)的归一化邻接矩阵 (3, V, V)。"""
    A = np.zeros((3, V, V), np.float32)
    for i, j in edges:
        A[0, i, j] = A[0, j, i] = 1.0
        A[1, j, i] = 1.0
        A[2, i, j] = 1.0
    for k in range(3):
        A[k] += np.eye(V, dtype=np.float32)
        deg = A[k].sum(1, keepdims=True)
        A[k] /= np.maximum(deg, 1e-6)
    return A


class SpatialGraphConv(nn.Module):
    """在顶点维做 GCN:对每个子集一个 1x1 卷积后按邻接矩阵聚合。"""

    def __init__(self, in_c, out_c, adj, num_subset=3):
        super().__init__()
        self.num_subset = num_subset
        self.register_buffer("A", torch.from_numpy(adj))
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_c, out_c, 1) for _ in range(num_subset)]
        )
        # 让邻接权重可学习(残差式)
        self.alpha = nn.Parameter(torch.zeros(num_subset, 1, 1))

    def forward(self, x):  # (N, C, T, V)
        N, C, T, V = x.shape
        y = None
        for k in range(self.num_subset):
            h = self.conv[k](x).view(N, -1, T, V)          # (N, C', T, V)
            h = torch.einsum("nctv,vw->nctw", h, self.A[k])
            y = h * (1.0 + self.alpha[k]) if y is None else y + h * (1.0 + self.alpha[k])
        return y


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, adj, stride=1, residual=True, dropout=0.0):
        super().__init__()
        self.gcn = SpatialGraphConv(in_c, out_c, adj, adj.shape[0])
        self.bn1 = nn.BatchNorm2d(out_c)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout),
        )
        if not residual:
            self.res = None
        elif in_c == out_c and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, (stride, 1)), nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = 0 if self.res is None else self.res(x)
        x = self.tcn(self.bn1(self.gcn(x)))
        return self.relu(x + res)


class STGCN(nn.Module):
    """ST-GCN:10 个 block,输出 top-1/top-5 分类。"""

    def __init__(self, num_classes, in_channels=2, num_joints=17, edges=None,
                 dropout=0.0):
        super().__init__()
        adj = _normalize_adjacency(num_joints, edges or coco17_edges())
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        chans = [64, 64, 64, 128, 128, 128, 256, 256, 256]
        blocks = []
        in_c = in_channels
        for i, out_c in enumerate(chans):
            blocks.append(
                STGCNBlock(in_c, out_c, adj, stride=1 if i in (0, 3, 6) else 1,
                           residual=(i != 0), dropout=dropout)
            )
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, num_classes)
        self.num_classes = int(num_classes)

    def forward(self, x):  # (N, C, T, V)
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)  # (N, V*C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()   # (N, C, T, V)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).view(N, -1)
        return self.fc(x)


def build_pose_action_model(arch):
    head = (arch or {}).get("Head") or {}
    bb = (arch or {}).get("Backbone") or {}
    return STGCN(
        num_classes=int(head.get("num_classes", 3)),
        in_channels=int(bb.get("in_channels", 2)),
        num_joints=int(bb.get("num_joints", 17)),
        dropout=float(bb.get("dropout", 0.0)),
    )


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class PoseActionDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg.get("data_dir", "")
        tf = ds_cfg.get("transform") or {}
        self.clip_len = int(tf.get("clip_len", 60))
        self.num_joints = int(tf.get("num_joints", 17))
        aug = ds_cfg.get("augment") or {}
        self.train = mode == "Train"
        self.flip_p = float(aug.get("flip", 0.5)) if self.train else 0.0
        self.jitter = float(aug.get("jitter", 0.01)) if self.train else 0.0
        self.samples = []
        for lf in ds_cfg["label_file_list"]:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2:
                        self.samples.append((p[0], int(float(p[1]))))
        # 左右对称点对(COCO 17),翻转时交换
        self.flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
                           (13, 14), (15, 16)]
        if logger is not None:
            logger.info(
                "%s dataset: %d skeleton sequences (clip_len=%d, joints=%d, classes=%d)",
                mode, len(self.samples), self.clip_len, self.num_joints,
                max((s[1] for s in self.samples), default=-1) + 1,
            )

    def __len__(self):
        return len(self.samples)

    def _path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.data_dir, rel)

    def _load(self, path):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:                       # (T, C) 单点 -> (T, 1, C)
            arr = arr[:, None, :]
        if arr.shape[0] in (2, 3):              # (C, T, V) -> (T, V, C)
            arr = arr.transpose(1, 2, 0)
        return arr                              # (T, V, C)

    def _sample_time(self, arr):
        T = arr.shape[0]
        if T == self.clip_len:
            return arr
        idx = np.linspace(0, T - 1, self.clip_len).astype(np.int64)
        if self.train and T > self.clip_len:    # 训练时随机裁剪窗口
            start = np.random.randint(0, T - self.clip_len + 1)
            idx = start + np.arange(self.clip_len)
        return arr[np.clip(idx, 0, T - 1)]

    def __getitem__(self, index):
        rel, label = self.samples[index]
        try:
            arr = self._load(self._path(rel))
        except Exception:
            return []
        arr = self._sample_time(arr)
        if arr.shape[1] != self.num_joints:
            return []
        if self.train and self.flip_p > 0 and np.random.rand() < self.flip_p:
            arr = arr[:, :, ::-1].copy()
            for a, b in self.flip_pairs:
                if b < arr.shape[1]:
                    arr[:, [a, b]] = arr[:, [b, a]]
        if self.jitter > 0:
            arr = arr + np.random.uniform(-self.jitter, self.jitter, arr.shape).astype(np.float32)
        data = np.ascontiguousarray(arr.transpose(2, 0, 1))   # (C, T, V)
        return [data, np.int64(label)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    data = torch.from_numpy(np.stack([s[0] for s in batch], 0)).float()
    labels = torch.from_numpy(np.asarray([s[1] for s in batch])).long()
    return [data, labels]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)


# --------------------------------------------------------------------------- #
# 损失 / 指标
# --------------------------------------------------------------------------- #
class PoseActionLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing or 0.0))

    def forward(self, preds, batch):
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        labels = batch[1].to(logits.device)
        loss = self.ce(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        return {"loss": loss, "loss_ce": loss.detach(), "acc": acc.detach()}


class PoseActionMetric(object):
    def __init__(self, main_indicator="acc", topk=5, **kwargs):
        self.main_indicator = main_indicator
        self.topk = int(topk)
        self.reset()

    def reset(self):
        self.n = 0
        self.top1 = 0
        self.topk_hit = 0

    def __call__(self, post_result, batch):
        preds = torch.cat([r["logits"] for r in post_result], 0) if post_result else None
        if preds is None:
            return
        labels = batch[1].to(preds.device)
        self.n += labels.numel()
        order = preds.argsort(dim=1, descending=True)[:, : self.topk]
        self.top1 += (order[:, 0] == labels).sum().item()
        self.topk_hit += (order == labels.view(-1, 1)).any(1).sum().item()

    def get_metric(self):
        return {
            "acc": self.top1 / max(self.n, 1),
            "top%d" % self.topk: self.topk_hit / max(self.n, 1),
            "num": self.n,
        }


class PoseActionPostProcess(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, preds):
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        logits = logits.detach()
        prob = torch.softmax(logits, dim=1)
        return [{"logits": logits[i:i + 1], "label": int(prob[i].argmax()),
                 "score": float(prob[i].max())} for i in range(logits.shape[0])]


class PoseActionTask(TaskAdapter):
    """骨架行为识别(``Architecture.task: pose_action``)。"""

    name = "pose_action"

    def build_post_process(self, config):
        return PoseActionPostProcess()

    def build_model(self, config, post_process):
        return build_pose_action_model(config["Architecture"])

    def build_loss(self, config, model=None):
        cfg = dict(config.get("Loss") or {})
        cfg.pop("name", None)
        return PoseActionLoss(**cfg)

    def build_metric(self, config):
        cfg = dict(config.get("Metric") or {})
        cfg.pop("name", None)
        return PoseActionMetric(**cfg)

    def build_datasets(self, config, logger):
        train_ds = PoseActionDataset(config, "Train", logger)
        eval_ds = PoseActionDataset(config, "Eval", logger) if config.get("Eval") is not None else None
        return train_ds, eval_ds

    def train_collate(self, batch):
        return train_collate(batch)

    def eval_collate(self, batch):
        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        data = batch[0].to(device, non_blocking=True)
        metric(post_process(model(data)), batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture") or {}
        head = arch.get("Head") or {}
        bb = arch.get("Backbone") or {}
        ds = (config.get("Train") or {}).get("dataset") or {}
        tf = ds.get("transform") or {}
        return [
            "task=pose_action model=STGCN joints={} channels={} classes={} clip_len={}".format(
                bb.get("num_joints", 17), bb.get("in_channels", 2),
                head.get("num_classes"), tf.get("clip_len"),
            )
        ]
