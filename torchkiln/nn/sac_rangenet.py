"""SqueezeSegV3 (SAC-RangeNet53) — PyTorch port of Paddle3D's range-view segmenter.

Module/parameter names mirror ``PaddlePaddle/Paddle3D`` so the official
``.pdparams`` loads 1:1.  Input is a range image ``(N, 5, H, W)`` =
``[range, x, y, z, remission]``; output is a list of per-scale logits.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SACRangeNet", "SqueezeSegV3", "load_paddle_squeezesegv3",
           "SqueezeSegV3Loss", "SqueezeSegV3PostProcess",
           "build_squeezesegv3_loss", "build_squeezesegv3_postprocess"]


class _ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=None, bn_momentum=0.9):
        super().__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding,
                               bias=(bias is not False))
        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=1 - bn_momentum)

    def forward(self, x):
        return self._batch_norm(self._conv(x))


class _DeconvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=None, bn_momentum=0.9):
        super().__init__()
        self._deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding,
                                          bias=(bias is not False))
        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=1 - bn_momentum)

    def forward(self, x):
        return self._batch_norm(self._deconv(x))


class _SACISKBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.attention_layer = _ConvBNLayer(3, 9 * num_channels, 7, padding=3,
                                            bn_momentum=0.9)
        self.position_mlp = nn.Sequential(
            _ConvBNLayer(9 * num_channels, num_channels, 1, bn_momentum=0.9),
            nn.ReLU(),
            _ConvBNLayer(num_channels, num_channels, 3, padding=1, bn_momentum=0.9),
            nn.ReLU(),
        )

    def forward(self, xyz, feature):
        n, c, h, w = feature.shape
        new_feature = F.unfold(feature, 3, padding=1).reshape(n, 9 * c, h, w)
        attention_map = torch.sigmoid(self.attention_layer(xyz))
        new_feature = new_feature * attention_map
        new_feature = self.position_mlp(new_feature)
        return xyz, new_feature + feature


class _DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.9):
        super().__init__()
        self.ds_layer = nn.Sequential(
            _ConvBNLayer(in_channels, out_channels, 3, stride=(1, 2), padding=1,
                         bias=False, bn_momentum=bn_momentum),
            nn.LeakyReLU(0.1),
        )

    def forward(self, xyz, feature):
        feature = self.ds_layer(feature)
        xyz = F.interpolate(xyz, size=[xyz.shape[2], xyz.shape[3] // 2],
                            mode="bilinear", align_corners=True)
        return xyz, feature


class _EncoderStage(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, dropout_prob,
                 downsample=True, bn_momentum=0.9):
        super().__init__()
        self.downsample = downsample
        self.layers = nn.ModuleList(
            [_SACISKBlock(in_channels) for _ in range(num_blocks)])
        if downsample:
            self.layers.append(_DownsampleBlock(in_channels, out_channels,
                                                bn_momentum=bn_momentum))
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, xyz, feature):
        for layer in self.layers:
            xyz, feature = layer(xyz, feature)
        return xyz, self.dropout(feature)


class _Encoder(nn.Module):
    def __init__(self, in_channels, num_stage_blocks=(1, 2, 8, 8, 4),
                 dropout_prob=0.01, bn_momentum=0.9):
        super().__init__()
        down_channels = ((32, 64), (64, 128), (128, 256), (256, 256), (256, 256))
        self.conv_1 = nn.Sequential(
            _ConvBNLayer(in_channels, 32, 3, stride=1, padding=1, bias=False,
                         bn_momentum=bn_momentum),
            nn.LeakyReLU(0.1),
        )
        self.encoder_stages = nn.ModuleList([
            _EncoderStage(nb, ic, oc, dropout_prob=dropout_prob,
                          downsample=i < 3, bn_momentum=bn_momentum)
            for i, (nb, (ic, oc)) in enumerate(zip(num_stage_blocks, down_channels))
        ])

    def forward(self, inputs):
        xyz = inputs[:, 1:4, :, :]
        feature = self.conv_1(inputs)
        short_cuts = []
        for stage in self.encoder_stages:
            if stage.downsample:
                short_cuts.append(feature.detach())
            xyz, feature = stage(xyz, feature)
        return feature, short_cuts


class _InvertedResidual(nn.Module):
    def __init__(self, channels, bn_momentum=0.9):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNLayer(channels[1], channels[0], 1, stride=1, padding=0,
                         bias=False, bn_momentum=bn_momentum),
            nn.LeakyReLU(0.1),
            _ConvBNLayer(channels[0], channels[1], 3, stride=1, padding=1,
                         bias=False, bn_momentum=bn_momentum),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.conv(x) + x


class _DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, bn_momentum=0.9):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.Sequential()
        if upsample:
            self.layers.add_module("0", _DeconvBNLayer(
                in_channels, out_channels, [1, 4], stride=[1, 2], padding=[0, 1],
                bn_momentum=bn_momentum))
        else:
            self.layers.add_module("0", _ConvBNLayer(
                in_channels, out_channels, 3, padding=1, bn_momentum=bn_momentum))
        self.layers.add_module("1", nn.LeakyReLU(0.1))
        self.layers.add_module("2", _InvertedResidual([in_channels, out_channels],
                                                      bn_momentum=bn_momentum))

    def forward(self, feature):
        return self.layers(feature)


class _Decoder(nn.Module):
    def __init__(self, dropout_prob=0.01, bn_momentum=0.9):
        super().__init__()
        up_channels = ((256, 256), (256, 256), (256, 128), (128, 64), (64, 32))
        self.decoder_stages = nn.ModuleList([
            _DecoderStage(ic, oc, upsample=i > 1, bn_momentum=bn_momentum)
            for i, (ic, oc) in enumerate(up_channels)
        ])
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, feature, short_cuts):
        feature_list = []
        for stage in self.decoder_stages:
            feature = stage(feature)
            if stage.upsample:
                feature = feature + short_cuts.pop()
            feature_list.append(self.dropout(feature))
        return feature_list


class SACRangeNet(nn.Module):
    def __init__(self, in_channels, num_layers=53, encoder_dropout_prob=0.01,
                 decoder_dropout_prob=0.01, bn_momentum=0.99, **kwargs):
        super().__init__()
        assert num_layers in (21, 53)
        num_stage_blocks = (1, 1, 2, 2, 1) if num_layers == 21 else (1, 2, 8, 8, 4)
        self.encoder = _Encoder(in_channels, num_stage_blocks, encoder_dropout_prob,
                                bn_momentum=bn_momentum)
        self.decoder = _Decoder(decoder_dropout_prob, bn_momentum=bn_momentum)

    def forward(self, inputs):
        feature, short_cuts = self.encoder(inputs)
        return self.decoder(feature, short_cuts)


class SqueezeSegV3(nn.Module):
    """SqueezeSegV3 = SACRangeNet backbone + 5 multi-scale 1x1 heads."""

    def __init__(self, num_classes=20, in_channels=5, num_layers=53,
                 encoder_dropout_prob=0.01, decoder_dropout_prob=0.01):
        super().__init__()
        self.backbone = SACRangeNet(in_channels, num_layers=num_layers,
                                    encoder_dropout_prob=encoder_dropout_prob,
                                    decoder_dropout_prob=decoder_dropout_prob)
        self.heads = nn.ModuleList([
            nn.Conv2d(256, num_classes, 1),
            nn.Conv2d(256, num_classes, 1),
            nn.Conv2d(128, num_classes, 1),
            nn.Conv2d(64, num_classes, 1),
            nn.Conv2d(32, num_classes, 3, padding=1),
        ])

    def forward(self, x, return_features=False):
        feature_list = self.backbone(x)
        if return_features:
            return feature_list
        logits = [head(f) for head, f in zip(self.heads, feature_list)]
        return logits


class SqueezeSegV3Loss(nn.Module):
    """Weighted NLL over the 5 multi-scale logits (Paddle3D ``SSGLossComputation``)."""

    def __init__(self, num_classes=20, ignore_index=0, epsilon_w=0.001,
                 class_weight=None, **kwargs):
        super().__init__()
        w = torch.as_tensor(class_weight, dtype=torch.float32) if class_weight is not None else None
        self.nll = nn.NLLLoss(weight=w, ignore_index=int(ignore_index))
        self.ignore_index = int(ignore_index)

    def forward(self, preds, batch):
        logits_list = preds["logits"] if isinstance(preds, dict) else preds
        target = batch[1]
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)
        total = 0.0
        for logits in logits_list:
            t = F.interpolate(target.unsqueeze(1).float(), size=logits.shape[-2:],
                              mode="nearest").squeeze(1).long()
            total = total + self.nll(F.log_softmax(logits, dim=1), t)
        return {"loss": total}


class SqueezeSegV3PostProcess(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, preds, size=None):
        logits_list = preds["logits"] if isinstance(preds, dict) else preds
        logits = logits_list[-1]  # full-resolution scale
        if size is not None and logits.shape[-2:] != tuple(size):
            logits = F.interpolate(logits, size=tuple(size), mode="bilinear",
                                   align_corners=False)
        return logits.argmax(1)


def build_squeezesegv3_loss(loss_cfg, num_classes=None):
    cfg = dict(loss_cfg or {})
    cfg.pop("name", None)
    if num_classes is not None:
        cfg.setdefault("num_classes", num_classes)
    return SqueezeSegV3Loss(**cfg)


def build_squeezesegv3_postprocess(pp_cfg):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    return SqueezeSegV3PostProcess(**cfg)


def load_paddle_squeezesegv3(model, pkl_path, verbose=True):
    import pickle

    sd = pickle.load(open(pkl_path, "rb"))
    mapped = {}
    for k, v in sd.items():
        nk = k
        if k.endswith("._mean"):
            nk = k[: -len("._mean")] + ".running_mean"
        elif k.endswith("._variance"):
            nk = k[: -len("._variance")] + ".running_var"
        mapped[nk] = torch.as_tensor(np.asarray(v))
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if verbose:
        print("loaded:", len(mapped), "missing:", len(missing), missing[:8],
              "unexpected:", len(unexpected), unexpected[:8])
    return missing, unexpected
