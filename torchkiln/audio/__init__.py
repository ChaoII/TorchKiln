"""torchkiln.audio: 音频分类 / 说话人 / 关键词 任务的 SOTA 模型（PaddleSpeech 移植）。"""

from torchkiln.audio.panns import (  # noqa: F401
    CNN14,
    CNN10,
    CNN6,
    ConvBlock,
    ConvBlock5x5,
    cnn14,
    cnn10,
    cnn6,
    PANNsPretrainedURLs,
    PANNs_FEATURE_CONF,
)

__all__ = [
    "CNN14", "CNN10", "CNN6", "ConvBlock", "ConvBlock5x5",
    "cnn14", "cnn10", "cnn6", "PANNsPretrainedURLs", "PANNs_FEATURE_CONF",
]
