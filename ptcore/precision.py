from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch

__all__ = ["enable_paddle_like_precision"]


def enable_paddle_like_precision():
    """Match PaddleOCR's default GPU numerical behaviour.

    Paddle enables TF32 for cuDNN convolutions and cuBLAS matmuls on
    Ampere+ GPUs by default. For cross-framework parity (training and
    inference) we enable the same in PyTorch. Set the env var
    ``PYTORCHOCR_TF32=0`` (or ``NVIDIA_TF32_OVERRIDE=0``) to force full fp32.
    """
    if not torch.cuda.is_available():
        return
    disabled = os.environ.get("PYTORCHOCR_TF32", "1") == "0" or os.environ.get(
        "NVIDIA_TF32_OVERRIDE", ""
    ) == "0"
    torch.backends.cudnn.allow_tf32 = not disabled
    torch.backends.cuda.matmul.allow_tf32 = not disabled
    # Optional: autotune cuDNN conv algorithms for the (fixed) training shape.
    # Can speed up training on some GPUs; may pick different algorithms so it is
    # opt-in to keep cross-framework numerics stable by default.
    if os.environ.get("PYTORCHOCR_CUDNN_BENCHMARK", "0") == "1":
        torch.backends.cudnn.benchmark = True
