"""Run our PyTorch model on a preprocessed input tensor and optionally compare
it against a reference numpy array produced by Paddle.

Run with the PyTorch interpreter (the `ptocr` conda env).
"""
import argparse
import os
import sys

import numpy as np
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

from pytorchx.ocr.modeling.architectures.base_model import BaseModel
from pytorchx.ocr.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument(
        "--make-input",
        action="store_true",
        help="create the input npy with a fixed seed instead of loading it",
    )
    args = ap.parse_args()

    if args.make_input:
        rng = np.random.RandomState(0)
        x = rng.randn(1, 3, 640, 640).astype(np.float32)
        np.save(args.input, x)
        print("created input", args.input, x.shape)

    x = np.load(args.input)

    config = load_config(args.config)
    model = BaseModel(config["Architecture"])
    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        out = model(torch.from_numpy(x))
    maps = out["maps"].numpy()
    if args.output:
        np.save(args.output, maps)
    print(
        "torch  maps {} sum={:.4f} mean={:.6f} max={:.4f} min={:.4f}".format(
            maps.shape, float(maps.sum()), float(maps.mean()), float(maps.max()), float(maps.min())
        )
    )

    if args.compare:
        ref = np.load(args.compare)
        if ref.shape != maps.shape:
            print("SHAPE MISMATCH", ref.shape, maps.shape)
        else:
            diff = np.abs(ref - maps)
            print(
                "compare: max_abs_diff={:.3e} mean_abs_diff={:.3e} allclose(1e-4)={}".format(
                    float(diff.max()), float(diff.mean()), bool(np.allclose(ref, maps, atol=1e-4))
                )
            )


if __name__ == "__main__":
    main()
