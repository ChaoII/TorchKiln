"""Run every official Paddle model on the same fixed input and save the output.

Run with the Paddle interpreter (`paddlex`).
"""
import glob
import os
import sys

import numpy as np
import paddle
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PADDLE_REPO = r"E:\TorchKiln\_downloads\PaddleOCR"
sys.path.insert(0, PADDLE_REPO)

from ppocr.modeling.architectures.base_model import BaseModel  # noqa: E402

OFFICIAL = os.path.join(ROOT, "_downloads", "official")
UTILS = os.path.join(ROOT, "torchkiln/ocr", "utils")


def char_dict_for(name):
    if "PP-OCRv5" in name:
        return os.path.join(UTILS, "dict", "ppocrv5_dict.txt")
    if "PP-OCRv6_medium" in name or "PP-OCRv6_small" in name:
        return os.path.join(UTILS, "dict", "ppocrv6_dict.txt")
    if "PP-OCRv6_tiny" in name:
        return os.path.join(UTILS, "dict", "ppocrv6_tiny_dict.txt")
    return os.path.join(UTILS, "ppocr_keys_v1.txt")


def count_chars(path, use_space_char=True):
    n = sum(1 for _ in open(path, encoding="utf-8"))
    return n + 1 + (1 if use_space_char else 0)  # blank (+ space)


def main():
    for cfg in sorted(glob.glob(os.path.join(ROOT, "configs", "*", "*.yml"))):
        name = os.path.splitext(os.path.basename(cfg))[0]
        kind = os.path.basename(os.path.dirname(cfg))
        w = os.path.join(OFFICIAL, name + "_pretrained.pdparams")
        if not os.path.isfile(w):
            continue
        with open(cfg, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        arch = config["Architecture"]
        if kind == "rec" and arch.get("Head", {}).get("name") == "MultiHead":
            char_num = count_chars(char_dict_for(name))
            arch["Head"]["out_channels_list"] = {
                "CTCLabelDecode": char_num,
                "SARLabelDecode": char_num + 2,
                "NRTRLabelDecode": char_num + 4,
            }
        model = BaseModel(arch)
        sd = paddle.load(w)
        tgt = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            if k in tgt and tuple(tgt[k].shape) == tuple(np.asarray(v).shape):
                tgt[k].set_value(paddle.to_tensor(np.asarray(v)))
                loaded += 1
        model.eval()
        if kind == "det":
            x = np.random.RandomState(0).randn(1, 3, 640, 640).astype(np.float32)
        else:
            x = np.random.RandomState(0).randn(1, 3, 48, 320).astype(np.float32)
        out = model(paddle.to_tensor(x))
        y = out["maps"] if isinstance(out, dict) else out
        y = y.numpy()
        np.save(os.path.join(OFFICIAL, name + "_paddle_out.npy"), y)
        print("{:32s} paddle out {} loaded={}".format(name, tuple(y.shape), loaded))


if __name__ == "__main__":
    main()
