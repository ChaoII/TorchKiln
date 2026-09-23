"""Video action classification (安防视频行为): ``task: video_cls``.

数据 = **视频片段**(一段视频 / 一个帧目录),模型 = **TSM 系**
(2D CNN + Temporal Shift + 分段共识 —— PP-TSM / PP-TSMv2 的核心思想),
2D 主干直接复用 :class:`torchkiln.nn.attribute.PPLCNetX1_0`(PP-TSMv2 用的就是 LCNetV2),
所以算力友好、无需 3D 卷积。

标签文件(yolo 风格,一行一样本)::

    path/to/video.mp4        <action_id>
    path/to/frames_dir       <action_id>

采样:`num_segments` 段 × 每段 `frames_per_seg` 帧 = `clip_len` 帧(均匀采样,可选 dense);
预处理:resize 短边 + center/random crop → ImageNet 归一化;
增广:随机裁剪 + 水平翻转(训练);
指标:top-1 / top-5。想接自己的行为类别时只改 `Head.num_classes`。
"""

from __future__ import absolute_import, division

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ptcore.task import TaskAdapter
from torchkiln.nn.attribute import PPLCNetX1_0

__all__ = ["TemporalShift", "TSMVideoNet", "VideoClsDataset", "VideoClsLoss",
           "VideoClsMetric", "VideoClsTask", "build_video_cls_model",
           "IMAGENET_MEAN", "IMAGENET_STD"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# 模型:TSM(时序位移)+ 2D 主干 + 分段共识
# --------------------------------------------------------------------------- #
class TemporalShift(nn.Module):
    """把 1/shift_div 的通道沿时间轴前后平移一帧(无参、零 FLOPs)。"""

    def __init__(self, n_segment=8, shift_div=8):
        super().__init__()
        self.n_segment = int(n_segment)
        self.fold = int(shift_div)

    def forward(self, x):  # (N*T, C, H, W) -> 同形状
        if self.fold <= 1:
            return x
        nt, c, h, w = x.shape
        t = self.n_segment
        if nt % t:
            return x
        n = nt // t
        fold = c // self.fold
        if fold <= 0:
            return x
        out = torch.zeros_like(x)
        out = out.view(n, t, c, h, w)
        xv = x.view(n, t, c, h, w)
        out[:, 1:, :fold] = xv[:, :-1, :fold]                       # 后移
        out[:, : t - 1, fold : 2 * fold] = xv[:, 1:, fold : 2 * fold]  # 前移
        out[:, :, 2 * fold :] = xv[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class TSMVideoNet(nn.Module):
    """(N, T, C, H, W) -> logits(N, num_classes)。"""

    def __init__(self, num_classes, num_segments=8, shift_div=8, dropout_prob=0.0,
                 scale=1.0):
        super().__init__()
        self.num_segments = int(num_segments)
        self.shift = TemporalShift(self.num_segments, shift_div)
        self.backbone = PPLCNetX1_0(scale=scale)
        self.dropout = nn.Dropout(p=float(dropout_prob)) if dropout_prob else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)  # scale 决定末通道
        self.num_classes = int(num_classes)

    def forward(self, x):  # (N, T, C, H, W)
        n, t, c, h, w = x.shape
        x = x.reshape(n * t, c, h, w)
        x = self.shift(x)
        f = self.backbone(x)
        f = self.pool(f).flatten(1).view(n, t, -1).mean(1)   # 分段共识
        return self.fc(self.dropout(f))


def build_video_cls_model(arch):
    head = (arch or {}).get("Head") or {}
    bb = (arch or {}).get("Backbone") or {}
    return TSMVideoNet(
        num_classes=int(head.get("num_classes", 3)),
        num_segments=int(bb.get("num_segments", 8)),
        shift_div=int(bb.get("shift_div", 8)),
        dropout_prob=float(head.get("dropout_prob", 0.0)),
        scale=float(bb.get("scale", 1.0)),
    )


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class VideoClsDataset(Dataset):
    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg.get("data_dir", "")
        tf = ds_cfg.get("transform") or {}
        self.num_segments = int(tf.get("num_segments", 8))
        self.frames_per_seg = int(tf.get("frames_per_seg", 1))
        self.clip_len = self.num_segments * self.frames_per_seg
        self.size = int(tf.get("image_size", 224))
        self.crop = int(tf.get("crop_size", 224))
        aug = ds_cfg.get("augment") or {}
        self.train = mode == "Train"
        self.flip_p = float(aug.get("flip", 0.5)) if self.train else 0.0
        self.samples = []
        for lf in ds_cfg["label_file_list"]:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2:
                        self.samples.append((p[0], int(float(p[1]))))
        if logger is not None:
            logger.info(
                "%s dataset: %d video clips (segments=%d x %d frames, size=%d->%d, classes=%d)",
                mode, len(self.samples), self.num_segments, self.frames_per_seg,
                self.size, self.crop, max((s[1] for s in self.samples), default=-1) + 1,
            )

    def __len__(self):
        return len(self.samples)

    def _path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.data_dir, rel)

    def _frames(self, path):
        """返回帧列表(np.uint8 BGR);支持视频文件与帧目录。"""
        if os.path.isdir(path):
            names = sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS))
            return [cv2.imread(os.path.join(path, n)) for n in names]
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
        finally:
            cap.release()
        return frames

    def _sample_idx(self, total):
        if total <= 0:
            return [0] * self.clip_len
        idx = []
        seg = max(total // self.num_segments, 1)
        for i in range(self.num_segments):
            lo = i * seg
            hi = min(total, (i + 1) * seg)
            for j in range(self.frames_per_seg):
                if self.train:
                    k = np.random.randint(lo, max(hi, lo + 1))
                else:
                    k = lo + int((j + 0.5) * max(hi - lo, 1) / self.frames_per_seg)
                idx.append(min(max(k, 0), total - 1))
        return idx

    def _prep(self, frames):
        out = []
        for fr in frames:
            if fr is None:
                fr = np.zeros((self.size, self.size, 3), np.uint8)
            h, w = fr.shape[:2]
            r = self.size / min(h, w)
            fr = cv2.resize(fr, (max(1, int(round(w * r))), max(1, int(round(h * r)))))
            if self.train:
                hh, ww = fr.shape[:2]
                y0 = np.random.randint(0, max(1, hh - self.crop + 1))
                x0 = np.random.randint(0, max(1, ww - self.crop + 1))
            else:
                hh, ww = fr.shape[:2]
                y0, x0 = max(0, (hh - self.crop) // 2), max(0, (ww - self.crop) // 2)
            fr = cv2.resize(fr[y0:y0 + self.crop, x0:x0 + self.crop], (self.crop, self.crop))
            if self.train and self.flip_p > 0 and np.random.rand() < self.flip_p:
                fr = fr[:, ::-1]
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            fr = (fr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
            out.append(fr.transpose(2, 0, 1))
        return np.stack(out, 0).astype(np.float32)   # (T, C, H, W)

    def __getitem__(self, index):
        rel, label = self.samples[index]
        try:
            frames = self._frames(self._path(rel))
        except Exception:
            return []
        if not frames:
            return []
        idx = self._sample_idx(len(frames))
        clip = self._prep([frames[i] for i in idx])
        return [clip, np.int64(label)]


def _collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    clips = torch.from_numpy(np.stack([s[0] for s in batch], 0)).float()   # (N,T,C,H,W)
    labels = torch.from_numpy(np.asarray([s[1] for s in batch])).long()
    return [clips, labels]


def train_collate(batch):
    return _collate(batch)


def eval_collate(batch):
    return _collate(batch)


# --------------------------------------------------------------------------- #
# 损失 / 指标 / 后处理
# --------------------------------------------------------------------------- #
class VideoClsLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing or 0.0))

    def forward(self, preds, batch):
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        labels = batch[1].to(logits.device)
        loss = self.ce(logits, labels)
        return {"loss": loss, "loss_ce": loss.detach(),
                "acc": (logits.argmax(1) == labels).float().mean().detach()}


class VideoClsMetric(object):
    def __init__(self, main_indicator="acc", topk=5, **kwargs):
        self.main_indicator = main_indicator
        self.topk = int(topk)
        self.reset()

    def reset(self):
        self.n = self.top1 = self.topk_hit = 0

    def __call__(self, post_result, batch):
        if not post_result:
            return
        preds = torch.cat([r["logits"] for r in post_result], 0)
        labels = batch[1].to(preds.device)
        self.n += labels.numel()
        order = preds.argsort(dim=1, descending=True)[:, : self.topk]
        self.top1 += (order[:, 0] == labels).sum().item()
        self.topk_hit += (order == labels.view(-1, 1)).any(1).sum().item()

    def get_metric(self):
        return {"acc": self.top1 / max(self.n, 1),
                "top%d" % self.topk: self.topk_hit / max(self.n, 1),
                "num": self.n}


class VideoClsPostProcess(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, preds):
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        logits = logits.detach()
        prob = torch.softmax(logits, dim=1)
        return [{"logits": logits[i:i + 1], "label": int(prob[i].argmax()),
                 "score": float(prob[i].max())} for i in range(logits.shape[0])]


class VideoClsTask(TaskAdapter):
    """视频行为分类(``Architecture.task: video_cls``)。"""

    name = "video_cls"

    def build_post_process(self, config):
        return VideoClsPostProcess()

    def build_model(self, config, post_process):
        return build_video_cls_model(config["Architecture"])

    def build_loss(self, config, model=None):
        cfg = dict(config.get("Loss") or {})
        cfg.pop("name", None)
        return VideoClsLoss(**cfg)

    def build_metric(self, config):
        cfg = dict(config.get("Metric") or {})
        cfg.pop("name", None)
        return VideoClsMetric(**cfg)

    def build_datasets(self, config, logger):
        train_ds = VideoClsDataset(config, "Train", logger)
        eval_ds = VideoClsDataset(config, "Eval", logger) if config.get("Eval") is not None else None
        return train_ds, eval_ds

    def train_collate(self, batch):
        return train_collate(batch)

    def eval_collate(self, batch):
        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        clips = batch[0].to(device, non_blocking=True)
        metric(post_process(model(clips)), batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture") or {}
        head = arch.get("Head") or {}
        bb = arch.get("Backbone") or {}
        tf = ((config.get("Train") or {}).get("dataset") or {}).get("transform") or {}
        return [
            "task=video_cls model=TSM(LCNet) segments={}x{} crop={} classes={}".format(
                bb.get("num_segments", 8), tf.get("frames_per_seg", 1),
                tf.get("crop_size", 224), head.get("num_classes"),
            )
        ]
