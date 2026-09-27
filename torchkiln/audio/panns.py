"""PANNs（Pretrained Audio Neural Networks）音频分类 —— torch 移植。

对齐 PaddleSpeech ``paddlespeech/cls/models/panns/panns.py``（逐层同名，便于权重转换）：

    ConvBlock / ConvBlock5x5 / CNN14 / CNN10 / CNN6

参考: PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
      https://arxiv.org/pdf/1912.10211.pdf

输入口径（与 ``paddlespeech/cls/exps/panns/predict.py::extract_features`` 一致）：
    waveform(sr=32000) -> LogMelSpectrogram(n_fft=1024, hop=320, n_mels=64,
                                            f_min=50, f_max=14000, hann)
    -> transpose(0,2,1) -> (N, T, 64) -> unsqueeze(1) -> (N, 1, T, 64)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConvBlock", "ConvBlock5x5", "CNN14", "CNN10", "CNN6",
           "cnn14", "cnn10", "cnn6", "PANNsPretrainedURLs", "panns_log_mel"]

# PaddleSpeech 官方预训练权重（AudioSet 527 类）
PANNsPretrainedURLs = {
    "cnn14": "https://bj.bcebos.com/paddleaudio/models/panns_cnn14.pdparams",
    "cnn10": "https://bj.bcebos.com/paddleaudio/models/panns_cnn10.pdparams",
    "cnn6": "https://bj.bcebos.com/paddleaudio/models/panns_cnn6.pdparams",
}

# paddlespeech/cls/**/panns.yaml 的特征配置（必须一致，否则 ② 对不齐）
PANNs_FEATURE_CONF = dict(
    sample_rate=32000, n_fft=1024, hop_length=320, win_length=1024,
    window="hann", f_min=50.0, f_max=14000.0, n_mels=64,
)


def _pool(x, pool_size, pool_type):
    if pool_type == "max":
        return F.max_pool2d(x, kernel_size=pool_size)
    if pool_type == "avg":
        return F.avg_pool2d(x, kernel_size=pool_size)
    if pool_type == "avg+max":
        return (F.avg_pool2d(x, kernel_size=pool_size)
                + F.max_pool2d(x, kernel_size=pool_size))
    raise ValueError(
        f"Pooling type of {pool_type} is not supported. "
        "It must be one of 'max', 'avg' and 'avg+max'.")


class ConvBlock(nn.Module):
    """2 x [conv3x3 - BN - ReLU] + 池化。 ``conv bias_attr=False``。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return _pool(x, pool_size, pool_type)


class ConvBlock5x5(nn.Module):
    """1 x [conv5x5 - BN - ReLU] + 池化（CNN6 用）。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu(self.bn1(self.conv1(x)))
        return _pool(x, pool_size, pool_type)


def _bn0_nhwc(x, bn0):
    """Paddle 用 NHWC 转置把 mel 维(64)当通道做 BatchNorm2D(64)。

    x 形状 (N, 1, T, 64) -> transpose -> (N, 64, T, 1) -> bn0 -> transpose 回。
    """
    x = x.transpose(1, 3)          # == paddle.transpose([0,3,2,1])
    x = bn0(x)
    x = x.transpose(1, 3)
    return x


class CNN14(nn.Module):
    """14 层 CNN = 6 个 ConvBlock（每块 2 层 3x3），emb_size=2048。

    forward 输出：
      extract_embedding=True  -> 2048 维 embedding（后接 SoundClassifier）
      extract_embedding=False -> sigmoid(fc_audioset(x))，AudioSet 527 类概率
    """

    emb_size = 2048

    def __init__(self, extract_embedding: bool = True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

    def forward(self, x):
        x = _bn0_nhwc(x, self.bn0)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        # 注意: 这一块 pool_size=(1,1)（前 5 块才是 2x2）
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        # mean 掉 mel 维(keepdim=False, 与 Paddle 一致) -> 再对时间维 max+mean
        x = x.mean(dim=3)
        x = torch.amax(x, dim=2) + x.mean(dim=2)   # amax == paddle.Tensor.max(axis)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            return F.dropout(x, p=0.5, training=self.training)
        return torch.sigmoid(self.fc_audioset(x))


class CNN10(nn.Module):
    """10 层 CNN = 4 个 ConvBlock，emb_size=512。"""

    emb_size = 512

    def __init__(self, extract_embedding: bool = True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)

        self.fc1 = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

    def forward(self, x):
        x = _bn0_nhwc(x, self.bn0)
        for blk in (self.conv_block1, self.conv_block2,
                    self.conv_block3, self.conv_block4):
            x = blk(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)
        x = x.mean(dim=3)
        x = torch.amax(x, dim=2) + x.mean(dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        if self.extract_embedding:
            return F.dropout(x, p=0.5, training=self.training)
        return torch.sigmoid(self.fc_audioset(x))


class CNN6(nn.Module):
    """6 层 CNN = 4 个 ConvBlock5x5，emb_size=512。"""

    emb_size = 512

    def __init__(self, extract_embedding: bool = True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock5x5(1, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)

        self.fc1 = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

    def forward(self, x):
        x = _bn0_nhwc(x, self.bn0)
        for blk in (self.conv_block1, self.conv_block2,
                    self.conv_block3, self.conv_block4):
            x = blk(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)
        x = x.mean(dim=3)
        x = torch.amax(x, dim=2) + x.mean(dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        if self.extract_embedding:
            return F.dropout(x, p=0.5, training=self.training)
        return torch.sigmoid(self.fc_audioset(x))


def cnn14(extract_embedding: bool = True) -> CNN14:
    return CNN14(extract_embedding=extract_embedding)


def cnn10(extract_embedding: bool = True) -> CNN10:
    return CNN10(extract_embedding=extract_embedding)


def cnn6(extract_embedding: bool = True) -> CNN6:
    return CNN6(extract_embedding=extract_embedding)
