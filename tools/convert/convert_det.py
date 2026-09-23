"""Convert a dumped Paddle state dict (numpy pickle) into PyTorch weights.

Run this with the *PyTorch* interpreter (the `ptocr` conda env).

Usage:
    python tools/convert/convert_det.py \
        --config configs/ocr/det/PP-OCRv4_mobile_det.yml \
        --paddle-pkl _downloads/det_v4/v4_det.pkl \
        --output _downloads/det_v4/ch_PP-OCRv4_det_mobile.pth
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys

import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

from torchkiln.ocr.modeling.architectures.base_model import BaseModel
from torchkiln.ocr.utils.config import load_config


def rename(k):
    k = k.replace("._mean", ".running_mean")
    k = k.replace("._variance", ".running_var")
    return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--paddle-pkl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model = BaseModel(config["Architecture"])
    tgt = model.state_dict()

    with open(args.paddle_pkl, "rb") as f:
        paddle_sd = pickle.load(f)

    mapped = []
    shape_mismatch = []
    unused_paddle = []
    tgt_keys = set(tgt.keys())
    used = set()
    for k, v in paddle_sd.items():
        kk = rename(k)
        if kk not in tgt:
            unused_paddle.append(k)
            continue
        if tuple(tgt[kk].shape) != tuple(v.shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(tgt[kk].shape)))
            continue
        tgt[kk].copy_(torch.from_numpy(v))
        used.add(kk)
        mapped.append(kk)

    missing = [k for k in tgt_keys if k not in used]
    print("mapped tensors      :", len(mapped))
    print("unused paddle keys  :", len(unused_paddle))
    print("shape mismatches    :", len(shape_mismatch))
    print("torch keys unfilled :", len(missing))
    if unused_paddle[:10]:
        print("  e.g. unused :", unused_paddle[:10])
    if shape_mismatch[:10]:
        print("  e.g. mismatch:", shape_mismatch[:10])
    if missing[:10]:
        print("  e.g. missing :", missing[:10])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
