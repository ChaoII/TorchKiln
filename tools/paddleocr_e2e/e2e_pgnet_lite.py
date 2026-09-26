# Lightweight end-to-end PGNet components (Step-2 candidate).
#
#   PPLCNetV4E2E : PPLCNetV4(det, tiny) 骨干,返回 [RGB, f1, f2, f3, f4]
#                  (与原 e2e_resnet_vd_pg 返回 [image, f1..f6] 同构,
#                   使 BaseModel 的 backbone->neck->head 流水线可直接复用)
#   PGFPNLCNet   : PGFPN 的轻量改写,适配上面 4 个特征图(原版硬编码 7 输入/2048 通道)
#   PGHeadLite   : 宽度可配的 PGHead(原版 f_char 分支占 84% 参数,此处可压缩)
#
# 目标: <10MB 单模型端到端,用于高框密度(几百框/图)场景。
# 注意: 参数名带 uid 前缀(>=5 字符),因 ConvBNLayer 的 bn_name = "bn"+name[3:]
#       会剥掉前 3 字符,uid 太短会让多实例 bn 名冲突。
import itertools

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["PPLCNetV4E2E", "PGFPNLCNet", "PGHeadLite"]

_UID = itertools.count()
_UID_FMT = "p%03d_"


class PPLCNetV4E2E(nn.Layer):
    """PPLCNetV4(det,tiny) -> [RGB, f1@32s4, f2@48s8, f3@64s16, f4@160s32]。"""

    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        from .backbones.rec_lcnetv4 import PPLCNetV4

        kwargs.pop("model_size", None)
        kwargs.pop("det", None)
        self.lcnet = PPLCNetV4(det=True, model_size="tiny",
                               in_channels=in_channels)
        # BaseModel 会读 backbone.out_channels 传给 neck 当 in_channels
        # (LCNet tiny 最后一级输出 160 通道;neck 用 **kwargs 吞掉即可)
        self.out_channels = 160

    def forward(self, x):
        feats = self.lcnet(x)
        return [x] + list(feats)


class PGFPNLCNet(nn.Layer):
    """PGFPN 轻量改写:输入 [RGB, f1, f2, f3, f4],输出 (B, w, H/4, W/4)。

    结构保持原 PGFPN 双路: down-fusion(RGB+浅层) + up-fusion(深层逐级上采样)。
    """

    def __init__(self, w=64, feat_ch=(32, 48, 64, 160), **kwargs):
        super().__init__()
        self.w = w
        # BaseModel 读 neck.out_channels 传给 head 当 in_channels
        self.out_channels = w
        uid = _UID_FMT % next(_UID)
        from .necks.pg_fpn import ConvBNLayer, DeConvBNLayer

        def cb(cin, cout, k, s, act, nm):
            return ConvBNLayer(in_channels=cin, out_channels=cout,
                               kernel_size=k, stride=s, act=act, name=uid + nm)

        def dcb(cin, cout, nm):
            return DeConvBNLayer(in_channels=cin, out_channels=cout, name=uid + nm)

        # down fusion: img(s4) + f1(s4)
        self.d0 = cb(3, w, 3, 1, "relu", "lf_d0")
        self.d1 = cb(feat_ch[0], w, 3, 1, "relu", "lf_d1")
        self.d2 = cb(w, w, 3, 1, "relu", "lf_d2")
        self.d3 = cb(w, w, 3, 1, None, "lf_d3")
        # up fusion: f4(s32) -> f3(s16) -> f2(s8) -> f1(s4)
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
        # down fusion: RGB 先池化到 s4,保证与 f1 同分辨率且对任意尺寸稳健
        h0 = self.d0(F.avg_pool2d(img, 4, 4, ceil_mode=True))
        h1 = self.d1(f1)
        g = F.relu(h0 + h1)
        g = self.d2(g)
        f_down = self.d3(g)
        # up fusion
        a0, a1, a2, a3 = self.h0(f4), self.h1(f3), self.h2(f2), self.h3(f1)
        g = self.up0(a0)                      # s32 -> s16
        g = F.relu(g + a1)
        g = self.g1(g)
        g = self.up1(g)                       # s16 -> s8
        g = F.relu(g + a2)
        g = self.g2(g)
        g = self.up2(g)                       # s8  -> s4
        g = F.relu(g + a3)
        g = self.g3(g)
        return F.relu(f_down + g)             # (B, w, H/4, W/4)


class PGHeadLite(nn.Layer):
    """宽度可配的 PGHead。

    官方 f_char 分支宽 128/256,占 PGHead 84% 参数(conv_f_char4 单层 54%);
    改为 w_char 后可压到 0.4M 级。w_score 分支与官方一致(64,64,128)。
    """

    def __init__(self, in_channels, w_char=(64, 64, 128, 128, 128),
                 w_score=(64, 64, 128), character_dict_path=None, **kwargs):
        super().__init__()
        uid = _UID_FMT % next(_UID)
        from .heads.e2e_pg_head import ConvBNLayer

        # 字典长度 + 1 (与官方 PGHead 一致: character_length = len(lines) + 1)
        if character_dict_path is None:
            character_dict_path = "ppocr/utils/ic15_dict.txt"
        with open(character_dict_path, "rb") as f:
            character_length = len(f.readlines()) + 1
        self.character_length = character_length

        def cb(cin, cout, k, nm):
            return ConvBNLayer(in_channels=cin, out_channels=cout,
                               kernel_size=k, stride=1, padding=k // 2,
                               act="relu", name=uid + nm)

        def cb1(cin, cout, nm):
            return ConvBNLayer(in_channels=cin, out_channels=cout,
                               kernel_size=1, stride=1, padding=0,
                               act="relu", name=uid + nm)

        s0, s1, s2 = w_score
        # score
        self.cs1 = cb1(in_channels, s0, "conv_f_score1")
        self.cs2 = cb(s0, s1, 3, "conv_f_score2")
        self.cs3 = cb1(s1, s2, "conv_f_score3")
        self.conv1 = cb(s2, 1, 3, "conv1")
        # border
        self.cb1 = cb1(in_channels, s0, "conv_f_boder1")
        self.cb2 = cb(s0, s1, 3, "conv_f_boder2")
        self.cb3 = cb1(s1, s2, "conv_f_boder3")
        self.conv2 = cb(s2, 4, 3, "conv2")
        # direction
        self.cd1 = cb1(in_channels, s0, "conv_f_direc1")
        self.cd2 = cb(s0, s1, 3, "conv_f_direc2")
        self.cd3 = cb1(s1, s2, "conv_f_direc3")
        self.conv4 = cb(s2, 2, 3, "conv4")
        # char
        c0, c1, c2, c3, c4 = w_char
        self.cc1 = cb1(in_channels, c0, "conv_f_char1")
        self.cc2 = cb(c0, c1, 3, "conv_f_char2")
        self.cc3 = cb1(c1, c2, "conv_f_char3")
        self.cc4 = cb(c2, c3, 3, "conv_f_char4")
        self.cc5 = cb1(c3, c4, "conv_f_char5")
        self.conv3 = cb(c4, character_length, 3, "conv3")

    def forward(self, x, targets=None):
        return {
            "f_score": self.conv1(self.cs3(self.cs2(self.cs1(x)))),
            "f_border": self.conv2(self.cb3(self.cb2(self.cb1(x)))),
            "f_char": self.conv3(self.cc5(self.cc4(self.cc3(self.cc2(self.cc1(x)))))),
            "f_direction": self.conv4(self.cd3(self.cd2(self.cd1(x)))),
        }
