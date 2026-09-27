"""轻量端到端 PGNet 组件（torch 版，对齐 PaddleOCR 权重命名）。

设计目标：单模型端到端 <10MB，用于高框密度（几百框/图）场景。
三个组件与 Paddle 侧 ``ppocr/modeling/e2e_pgnet_lite.py`` **逐属性同名**，
因此 Paddle 训练好的 ``.pdparams`` 可直接转换后加载。

    PPLCNetV4E2E  骨干: PPLCNetV4(det, tiny) -> [RGB, f1@32s4, f2@48s8, f3@64s16, f4@160s32]
                  （返回 RGB 是关键：原 PGNet 骨干 e2e_resnet_vd_pg 同样返回 [image, f1..f6]）
    PGFPNLCNet    颈  : PGFPN 轻量改写，适配上面 4 个特征图
                  （原版硬编码 7 输入 [3,64,256,512,1024,2048,2048]，含 s2/s64 两级）
    PGHeadLite    头  : 宽度可配的 PGHead
                  （原版 f_char 分支宽 128/256，占总参数 84%；此处默认压到 64/128）
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PPLCNetV4E2E", "PGFPNLCNet", "PGHeadLite"]


class PPLCNetV4E2E(nn.Module):
    """PPLCNetV4(det,tiny) 骨干，返回 ``[RGB, f1, f2, f3, f4]``。

    与原 PGNet 骨干 ``e2e_resnet_vd_pg.ResNet`` 的返回形式同构
    （``[image, f1..f6]``），使框架的 ``backbone->neck->head`` 流水线可直接复用。
    """

    def __init__(self, in_channels=3, **kwargs):
        super(PPLCNetV4E2E, self).__init__()
        from torchkiln.ocr.modeling.backbones.rec_lcnetv4 import PPLCNetV4

        self.lcnet = PPLCNetV4(det=True, model_size="tiny",
                               in_channels=in_channels)
        # 框架会读 backbone.out_channels 作为 neck 的 in_channels
        # (LCNet tiny 末级输出 160 通道; neck 用 **kwargs 吞掉)
        self.out_channels = 160

    def forward(self, x):
        feats = self.lcnet(x)
        return [x] + list(feats)


class PGFPNLCNet(nn.Module):
    """PGFPN 轻量改写：输入 ``[RGB, f1, f2, f3, f4]``，输出 ``(B, w, H/4, W/4)``。

    保持原 PGFPN 的双路结构：
      * down-fusion : RGB(池化到 s4) + f1(s4)  ->  f_down
      * up-fusion   : f4(s32) -> f3(s16) -> f2(s8) -> f1(s4) 逐级上采样  ->  f_up
    输出与原 PGFPN 同分辨率(s4)，可直接接 PGHead。
    """

    def __init__(self, w=64, feat_ch=(32, 48, 64, 160), **kwargs):
        super(PGFPNLCNet, self).__init__()
        from torchkiln.ocr.modeling.necks.pg_fpn import ConvBNLayer, DeConvBNLayer

        self.w = w
        self.out_channels = w          # 供 head 取 in_channels
        self.feat_ch = tuple(feat_ch)

        def cb(cin, cout, k, s, act, nm):
            # 注意 torch 版 ConvBNLayer 的第5个位置参数是 groups, 必须用关键字
            return ConvBNLayer(in_channels=cin, out_channels=cout,
                               kernel_size=k, stride=s, act=act, name=nm)

        def dcb(cin, cout, nm):
            return DeConvBNLayer(in_channels=cin, out_channels=cout, name=nm)

        # ---- down fusion: img(s4) + f1(s4) ----
        self.d0 = cb(3, w, 3, 1, "relu", "lf_d0")
        self.d1 = cb(feat_ch[0], w, 3, 1, "relu", "lf_d1")
        self.d2 = cb(w, w, 3, 1, "relu", "lf_d2")
        self.d3 = cb(w, w, 3, 1, None, "lf_d3")
        # ---- up fusion: f4(s32) -> f3(s16) -> f2(s8) -> f1(s4) ----
        self.h0 = cb(feat_ch[3], w, 1, 1, None, "uf_h0")
        self.h1 = cb(feat_ch[2], w, 1, 1, None, "uf_h1")
        self.h2 = cb(feat_ch[1], w, 1, 1, None, "uf_h2")
        self.h3 = cb(feat_ch[0], w, 1, 1, None, "uf_h3")
        self.up0 = dcb(w, w, "uf_up0")
        self.g1 = cb(w, w, 3, 1, "relu", "uf_g1")
        self.up1 = dcb(w, w, "uf_up1")
        self.g2 = cb(w, w, 3, 1, "relu", "uf_g2")
        self.up2 = dcb(w, w, "uf_up2")
        self.g3 = cb(w, w, 3, 1, "relu", "uf_g3")

    def forward(self, x):
        img, f1, f2, f3, f4 = x
        # down fusion（RGB 先池化到 s4，保证与 f1 同分辨率且对任意尺寸稳健）
        h0 = self.d0(F.avg_pool2d(img, 4, 4, ceil_mode=True))
        h1 = self.d1(f1)
        g = F.relu(h0 + h1)
        g = self.d2(g)
        f_down = self.d3(g)

        # up fusion
        a0 = self.h0(f4)
        a1 = self.h1(f3)
        a2 = self.h2(f2)
        a3 = self.h3(f1)
        g = self.up0(a0)               # s32 -> s16
        g = F.relu(g + a1)
        g = self.g1(g)
        g = self.up1(g)                # s16 -> s8
        g = F.relu(g + a2)
        g = self.g2(g)
        g = self.up2(g)                # s8  -> s4
        g = F.relu(g + a3)
        g = self.g3(g)
        return F.relu(f_down + g)


class PGHeadLite(nn.Module):
    """宽度可配的 PGHead（4 头: f_score / f_border / f_char / f_direction）。

    官方 ``PGHead`` 的 f_char 分支宽 128/128/256/256/256，而其它分支只 64/64/128，
    导致该分支独占 84% 参数（``conv_f_char4`` 单层 589K = 54%）。
    此处 ``w_char`` 默认压到 ``(64,64,128,128,128)``，与其它分支同级。
    """

    def __init__(self, in_channels, w_char=(64, 64, 128, 128, 128),
                 w_score=(64, 64, 128), character_length=37, **kwargs):
        super(PGHeadLite, self).__init__()
        from torchkiln.ocr.modeling.heads.e2e_pg_head import ConvBNLayer

        self.out_channels = in_channels
        self.w_char = tuple(w_char)

        def cb(cin, cout, k, nm):
            return ConvBNLayer(in_channels=cin, out_channels=cout,
                               kernel_size=k, stride=1, padding=k // 2,
                               act="relu", name=nm)

        def cb1(cin, cout, nm):
            return ConvBNLayer(in_channels=cin, out_channels=cout,
                               kernel_size=1, stride=1, padding=0,
                               act="relu", name=nm)

        s0, s1, s2 = w_score
        # ---- f_score 分支 ----
        self.cs1 = cb1(in_channels, s0, "conv_f_score1")
        self.cs2 = cb(s0, s1, 3, "conv_f_score2")
        self.cs3 = cb1(s1, s2, "conv_f_score3")
        self.conv1 = cb(s2, 1, 3, "conv1")
        # ---- f_border 分支 ----
        self.cb1 = cb1(in_channels, s0, "conv_f_boder1")
        self.cb2 = cb(s0, s1, 3, "conv_f_boder2")
        self.cb3 = cb1(s1, s2, "conv_f_boder3")
        self.conv2 = cb(s2, 4, 3, "conv2")
        # ---- f_char 分支（宽度可配）----
        c0, c1, c2, c3, c4 = w_char
        self.cc1 = cb1(in_channels, c0, "conv_f_char1")
        self.cc2 = cb(c0, c1, 3, "conv_f_char2")
        self.cc3 = cb1(c1, c2, "conv_f_char3")
        self.cc4 = cb(c2, c3, 3, "conv_f_char4")
        self.cc5 = cb1(c3, c4, "conv_f_char5")
        self.conv3 = cb(c4, character_length, 3, "conv3")
        # ---- f_direction 分支 ----
        self.cd1 = cb1(in_channels, s0, "conv_f_direc1")
        self.cd2 = cb(s0, s1, 3, "conv_f_direc2")
        self.cd3 = cb1(s1, s2, "conv_f_direc3")
        self.conv4 = cb(s2, 2, 3, "conv4")

    def forward(self, x, targets=None):
        return {
            "f_score": self.conv1(self.cs3(self.cs2(self.cs1(x)))),
            "f_border": self.conv2(self.cb3(self.cb2(self.cb1(x)))),
            "f_char": self.conv3(
                self.cc5(self.cc4(self.cc3(self.cc2(self.cc1(x)))))),
            "f_direction": self.conv4(self.cd3(self.cd2(self.cd1(x)))),
        }
