"""Run the official PaddleOCR model on a preprocessed input tensor.

Run with the Paddle interpreter (e.g. the `paddlex` conda env).

Usage:
    python tools/convert/paddle_forward.py \
        --paddle-repo <PaddleOCR repo> --config <cfg.yml> \
        --weights <x.pdparams> --input <in.npy> --output <out.npy>
"""
import argparse
import sys

import numpy as np
import paddle
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paddle-repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.paddle_repo)
    from ppocr.modeling.architectures.base_model import BaseModel

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["Architecture"]

    model = BaseModel(cfg)
    sd = paddle.load(args.weights)
    model.set_state_dict(sd)
    model.eval()

    x = np.load(args.input)
    out = model(paddle.to_tensor(x))
    maps = out["maps"].numpy()
    np.save(args.output, maps)
    print(
        "paddle maps {} sum={:.4f} mean={:.6f} max={:.4f} min={:.4f}".format(
            maps.shape, float(maps.sum()), float(maps.mean()), float(maps.max()), float(maps.min())
        )
    )


if __name__ == "__main__":
    main()
