"""BEV-LaneDet — PyTorch port of Paddle3D's lane detector (ResNet-34 + BEV).

Module/parameter names mirror ``PaddlePaddle/Paddle3D`` so the official
``.pdparams`` loads 1:1.  Inference forward returns ``(seg, emb, offset_y, z)``.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BEVLaneDet", "load_paddle_bev_lanedet"]

_BN = dict(eps=1e-5, momentum=0.1)


def _bn(c):
    return nn.BatchNorm2d(c, **_BN)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = _bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = _bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False), _bn(planes))
    layers = [_BasicBlock(inplanes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(_BasicBlock(planes, planes))
    return nn.Sequential(*layers)


def _resnet34_children():
    return [
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        _bn(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        _make_layer(64, 64, 3),
        _make_layer(64, 128, 4, stride=2),
        _make_layer(128, 256, 6, stride=2),
        _make_layer(256, 512, 3, stride=2),
    ]


class _Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super().__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.module(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _FCTransform(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super().__init__()
        ic, ih, iw = image_featmap_size
        sc, sh, sw = space_featmap_size
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw), nn.ReLU(inplace=True),
            nn.Linear(sh * sw, sh * sw), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ic, sc, 1, bias=False), _bn(sc), nn.ReLU(inplace=True))
        self.residual = _Residual(nn.Sequential(
            nn.Conv2d(sc, sc, 3, padding=1, bias=False), _bn(sc)))

    def forward(self, x):
        b, c = x.shape[:2]
        x = x.reshape(b, c, self.image_featmap_size[1] * self.image_featmap_size[2])
        bev = self.fc_transform(x)
        bev = bev.reshape(b, bev.shape[1], self.space_featmap_size[1],
                          self.space_featmap_size[2])
        return self.residual(self.conv1(bev))


def _conv_bn_relu(ci, co):
    return [nn.Conv2d(ci, co, 3, padding=1, bias=False), _bn(co), nn.ReLU(inplace=True)]


class _InstanceEmbeddingOffsetYZ(nn.Module):
    def __init__(self, ci, co=1):
        super().__init__()
        self.neck_new = nn.Sequential(
            *_conv_bn_relu(ci, 128), *_conv_bn_relu(128, ci))
        self.ms_new = nn.Sequential(
            *_conv_bn_relu(ci, 128), *_conv_bn_relu(128, 64),
            nn.Conv2d(64, 1, 3, padding=1, bias=True))
        self.m_offset_new = nn.Sequential(
            *_conv_bn_relu(ci, 128), *_conv_bn_relu(128, 64),
            nn.Conv2d(64, 1, 3, padding=1, bias=True))
        self.m_z = nn.Sequential(
            *_conv_bn_relu(ci, 128), *_conv_bn_relu(128, 64),
            nn.Conv2d(64, 1, 3, padding=1, bias=True))
        self.me_new = nn.Sequential(
            *_conv_bn_relu(ci, 128), *_conv_bn_relu(128, 64),
            nn.Conv2d(64, co, 3, padding=1, bias=True))

    def forward(self, x):
        feat = self.neck_new(x)
        return (self.ms_new(feat), self.me_new(feat),
                self.m_offset_new(feat), self.m_z(feat))


class _InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super().__init__()
        self.neck = nn.Sequential(*_conv_bn_relu(ci, 128), *_conv_bn_relu(128, ci))
        self.ms = nn.Sequential(
            *_conv_bn_relu(ci, 128), *_conv_bn_relu(128, 64),
            nn.Conv2d(64, 1, 3, padding=1, bias=True))
        self.me = nn.Sequential(
            *_conv_bn_relu(ci, 128), *_conv_bn_relu(128, 64),
            nn.Conv2d(64, co, 3, padding=1, bias=True))

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class _LaneHeadWithOffsetZ(nn.Module):
    def __init__(self, output_size, input_channel=512):
        super().__init__()
        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),
            _Residual(
                nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False), _bn(64),
                    nn.ReLU(inplace=True), nn.Dropout2d(0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False), _bn(128)),
                downsample=nn.Conv2d(input_channel, 128, 1)),
            nn.Upsample(size=output_size),
            _Residual(
                nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False), _bn(64),
                    nn.ReLU(inplace=True), nn.Dropout2d(0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False), _bn(64)),
                downsample=nn.Conv2d(128, 64, 1)),
        )
        self.head = _InstanceEmbeddingOffsetYZ(64, 2)

    def forward(self, x):
        return self.head(self.bev_up_new(x))


class _LaneHead2D(nn.Module):
    def __init__(self, output_size, input_channel=512):
        super().__init__()
        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            _Residual(
                nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False), _bn(64),
                    nn.ReLU(inplace=True), nn.Dropout2d(0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False), _bn(128)),
                downsample=nn.Conv2d(input_channel, 128, 1)),
            nn.Upsample(scale_factor=2),
            _Residual(
                nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False), _bn(64),
                    nn.ReLU(inplace=True), nn.Dropout2d(0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False), _bn(32)),
                downsample=nn.Conv2d(128, 32, 1)),
            nn.Upsample(size=output_size),
            _Residual(
                nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False), _bn(16),
                    nn.ReLU(inplace=True), nn.Dropout2d(0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False), _bn(32))),
        )
        self.head = _InstanceEmbedding(32, 2)

    def forward(self, x):
        return self.head(self.bev_up(x))


class BEVLaneDet(nn.Module):
    def __init__(self, bev_shape, output_2d_shape, train=True):
        super().__init__()
        self.bb = nn.Sequential(*_resnet34_children())
        self.down = _Residual(
            nn.Sequential(
                nn.Conv2d(512, 1024, 3, stride=2, padding=1), _bn(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1), _bn(1024)),
            downsample=nn.Conv2d(512, 1024, 3, stride=2, padding=1))
        self.s32transformer = _FCTransform((512, 18, 32), (256, 25, 5))
        self.s64transformer = _FCTransform((1024, 9, 16), (256, 25, 5))
        self.lane_head = _LaneHeadWithOffsetZ(bev_shape, input_channel=512)
        self.is_train = train
        if train:
            self.lane_head_2d = _LaneHead2D(output_2d_shape, input_channel=512)

    def forward(self, img):
        img_s32 = self.bb(img)
        bev_32 = self.s32transformer(img_s32)
        if self.is_train:
            img_s64 = self.down(img_s32)
            bev_64 = self.s64transformer(img_s64)
        else:
            bev_64 = self.s64transformer(self.down(img_s32))
        bev = torch.cat([bev_64, bev_32], dim=1)
        seg, emb, offset_y, z = self.lane_head(bev)
        out = [seg, emb, offset_y, z]
        if self.is_train:
            out += list(self.lane_head_2d(img_s32))
        return out


def load_paddle_bev_lanedet(model, pkl_path, verbose=True):
    import pickle

    sd = pickle.load(open(pkl_path, "rb"))
    mapped = {}
    for k, v in sd.items():
        t = torch.as_tensor(np.asarray(v))
        if k.endswith("fc_transform.0.weight") or k.endswith("fc_transform.2.weight"):
            t = t.t().contiguous()  # Paddle Linear [in,out] -> torch [out,in]
        if k.endswith("._mean"):
            k = k[: -len("._mean")] + ".running_mean"
        elif k.endswith("._variance"):
            k = k[: -len("._variance")] + ".running_var"
        mapped[k] = t
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if verbose:
        print("loaded:", len(mapped), "missing:", len(missing), missing[:8],
              "unexpected:", len(unexpected), unexpected[:8])
    return missing, unexpected
