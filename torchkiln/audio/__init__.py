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
from torchkiln.audio.ecapa_tdnn import (  # noqa: F401
    EcapaTdnn,
    ECAPA_TDNN_CONF,
    AttentiveStatisticsPooling,
    SEBlock,
    SERes2NetBlock,
    Res2NetBlock,
    TDNNBlock,
    length_to_mask,
)
from torchkiln.audio.mdtc import (  # noqa: F401
    DSDilatedConv1d,
    TCNBlock,
    TCNStack,
    MDTC,
    KWSModel,
    build_mdtc,
    MDTC_CONF,
    KALDI_FBANK_CONF,
)

__all__ = [
    "CNN14", "CNN10", "CNN6", "ConvBlock", "ConvBlock5x5",
    "cnn14", "cnn10", "cnn6", "PANNsPretrainedURLs", "PANNs_FEATURE_CONF",
    "EcapaTdnn", "ECAPA_TDNN_CONF", "AttentiveStatisticsPooling",
    "SEBlock", "SERes2NetBlock", "Res2NetBlock", "TDNNBlock",
    "length_to_mask",
    "DSDilatedConv1d", "TCNBlock", "TCNStack", "MDTC", "KWSModel",
    "build_mdtc", "MDTC_CONF", "KALDI_FBANK_CONF",
]
