"""kokoro-82M（TTS）模型 —— 移植自官方 ``hexgrad/kokoro``（MIT 许可，保留原版权头）。

官方仓库/包已是 PyTorch，本目录做两件事：
  1. 把绝对包引用 ``kokoro.*`` 归位为包内相对引用（``istftnet.py`` 里有 1 处）
  2. 把 ``loguru`` 换成标准 ``logging``；去掉 ``hf_hub_download`` 回退
     （本机 HuggingFace 被墙；必须显式传本地 ``config.json`` 与 ``.pth``）

权重（ModelScope ``hexgrad/Kokoro-82M`` / ``hexgrad/Kokoro-82M-v1.1-zh``）：
  ``kokoro-v1_0.pth``（英文/多语言）、``kokoro-v1_1-zh.pth``（中文）。

⚠️ 加载要点（① 已验证，两版权重通用）：
  * 权重键带 ``module.`` 前缀 -> ``k[7:]`` 去掉
  * ``AdaIN1d`` 内的 ``InstanceNorm1d`` 必须用 **``affine=False``**
    （产权重的旧版无 weight/bias；用默认 affine=True 会 missing 140 键）
    桩包法构造/修法见 ``_downloads/kokoro_load.py``、``kokoro_fix_test.py``。
"""
from .model import KModel  # noqa: F401
from .istftnet import Decoder  # noqa: F401
from .modules import (  # noqa: F401
    CustomAlbert,
    ProsodyPredictor,
    TextEncoder,
    DurationEncoder,
    AdaLayerNorm,
    LinearNorm,
    LayerNorm,
)

__all__ = [
    "KModel", "Decoder", "CustomAlbert", "ProsodyPredictor",
    "TextEncoder", "DurationEncoder", "AdaLayerNorm", "LinearNorm",
    "LayerNorm",
]
