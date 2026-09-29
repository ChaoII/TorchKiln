"""单目（图像）3D 检测数据集。

与 ``torchkiln/data/det3d.py`` 的 LiDAR 版**共用同一套标签格式与同一个
``PillarDetNet``**：模型输入契约只有 ``(B, C, H, W)``，RGB 图天然满足，改
``Architecture.in_channels: 3`` 即可，模型 / loss / metric / 后处理全部零改动。

标签行与 LiDAR 版完全一致：``cls x y z l w h yaw``（≥8 列，第 9 列 difficulty 忽略），
坐标系也是 **LiDAR/ego 系**（x 前 / y 左 / z 上），``z`` 为**框中心**，``l`` 沿 ``yaw``。
本模块只管「把图读进来」，**不做任何坐标系转换**——转换在平台侧的导出器里做
（相机系 → LiDAR 系是一次固定的轴置换，见 AIStation 的 cuboid 导出）。

磁盘布局沿用全仓 ``images/`` → ``labels/`` 的路径替换约定（与 ``DetDataset`` /
``SegDataset`` 一致）::

    <data_dir>/
    ├── train.txt                     # 每行: images/train/<id>.jpg
    ├── val.txt
    ├── images/<split>/<id>.jpg
    └── labels/<split>/<id>.txt       # 每行: cls x y z l w h yaw
"""
from __future__ import annotations

import os

import numpy as np
from torch.utils.data import Dataset

from .det3d import load_box_labels

#: det3d 系列统一的「images → labels」路径替换（与 ``det._label_path_for`` 相同）
def _label_path_for(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return os.path.splitext("/".join(parts))[0] + ".txt"
    d = os.path.dirname(rel)
    return os.path.join(d, os.path.splitext(os.path.basename(rel))[0] + ".txt")


def _load_rgb(path: str, size) -> "np.ndarray":
    """读 RGB 图并转成 ``(3, H, W) float32``，像素归一化到 ``[0, 1]``。

    归一化口径（除 255、不减 mean/std）与 ``torchkiln/data/det.py`` 保持一致。
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot read image: {}".format(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if size is not None and tuple(img.shape[:2]) != tuple(size):
        # 注意 cv2.resize 的入参是 (宽, 高)，而 size 是 (H, W)
        img = cv2.resize(img, (int(size[1]), int(size[0])), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


class Mono3DDataset(Dataset):
    """图像 + 3D 框（LiDAR 系 7-dof）的检测数据集。"""

    def __init__(self, config, mode="Train", logger=None):
        ds_cfg = config[mode]["dataset"]
        self.mode = mode
        self.data_dir = ds_cfg["data_dir"]
        self.names = ds_cfg.get("names")
        self.num_classes = ds_cfg.get("num_classes")
        self.max_boxes = int(ds_cfg.get("max_boxes", 64))
        tf = ds_cfg.get("transform") or {}
        self.image_size = tuple(tf["image_size"]) if tf.get("image_size") else None
        aug = (ds_cfg.get("augment") or {}) if mode == "Train" else {}
        self.hflip = bool(aug.get("hflip", False))
        self.jitter = float(aug.get("color_jitter", 0.0)) or 0.0

        self.img_files = []
        for label_file in ds_cfg["label_file_list"]:
            # 与其他任务族一致：非绝对路径且不是现存文件时拼到 data_dir 下
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
                "%s mono3d dataset: %d images (data_dir=%s, hflip=%s)",
                mode, len(self.img_files), self.data_dir, self.hflip,
            )

    def __len__(self):
        return len(self.img_files)

    def set_epoch(self, epoch):
        pass

    def _augment(self, img, labels):
        """图像增强。**只做对标签有确定物理含义的变换。**

        ⚠️ LiDAR 版 ``det3d.augment_boxes`` 对点云数组做数值变换，喂给图像数组
        在语法上「能跑」但语义是垃圾（``pc[:, 0]`` 会把通道数当 batch），绝不能复用。

        当前只支持水平翻转：前视相机左右镜像 = LiDAR 系 y 取反，
        所以 ``y -> -y`` 且 ``yaw -> -yaw``（yaw 是车长方向航向角）。
        图像缩放**不改标签**——它只改像素网格，不改米制坐标。
        """
        if self.hflip and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, :, ::-1])
            if labels.shape[0]:
                labels = labels.copy()
                labels[:, 2] = -labels[:, 2]   # y
                labels[:, 7] = -labels[:, 7]   # yaw
        if self.jitter > 0:
            g = 1.0 + (np.random.rand() * 2.0 - 1.0) * self.jitter
            img = np.clip(img * g, 0.0, 1.0).astype(np.float32)
        return img, labels

    def __getitem__(self, index):
        rel = self.img_files[index]
        img_path = os.path.join(self.data_dir, rel)
        try:
            img = _load_rgb(img_path, self.image_size)
        except Exception:
            return []                       # 读图失败：collate 会过滤掉，不中断训练

        lab_path = os.path.join(self.data_dir, _label_path_for(rel))
        if not os.path.isfile(lab_path):
            lab_path = _label_path_for(rel)
        labels = load_box_labels(lab_path, names=self.names)

        img, labels = self._augment(img, labels)

        if labels.shape[0] > self.max_boxes:
            # 保留体积最大的若干个（与 LiDAR 版一致的稳定裁剪策略）
            vol = labels[:, 4] * labels[:, 5] * labels[:, 6]
            labels = labels[np.argsort(-vol)[: self.max_boxes]]
        mask = np.ones((labels.shape[0],), dtype=np.float32)
        return [img, labels.astype(np.float32), mask]
