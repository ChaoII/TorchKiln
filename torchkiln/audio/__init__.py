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
    "KokoroLoss", "KokoroDataset", "kokoro_collate",
    "DSDilatedConv1d", "TCNBlock", "TCNStack", "MDTC", "KWSModel",
    "build_mdtc", "MDTC_CONF", "KALDI_FBANK_CONF",
]

# ---- kokoro-82M (TTS) ----
# ⚠️ kokoro 子包是**官方代码的搬运**（MIT，保留原头），其 ``__init__`` 会 import KModel；
#    训练用的 loss/dataset 在包外（kokoro_loss.py / kokoro_dataset.py）。
from torchkiln.audio.kokoro_loss import KokoroLoss  # noqa: F401
from torchkiln.audio.kokoro_dataset import (  # noqa: F401
    KokoroDataset,
    kokoro_collate,
)
