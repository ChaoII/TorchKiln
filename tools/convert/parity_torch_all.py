"""Run every converted torch model on a fixed input and save the output.

Run with the PyTorch interpreter (`ptocr`).
"""
import glob
import os
import sys

import numpy as np
import torch

# Match Paddle's default GPU numerical precision (TF32 enabled on Ampere+).
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from pytorchx.ocr.modeling.architectures.base_model import BaseModel
from pytorchx.ocr.postprocess import build_post_process
from pytorchx.ocr.utils.config import load_config

OFFICIAL = os.path.join(ROOT, "_downloads", "official")
UTILS = os.path.join(ROOT, "pytorchx/ocr", "utils")


def char_dict_for(name):
    if "PP-OCRv5" in name:
        return os.path.join(UTILS, "dict", "ppocrv5_dict.txt")
    if "PP-OCRv6_medium" in name or "PP-OCRv6_small" in name:
        return os.path.join(UTILS, "dict", "ppocrv6_dict.txt")
    if "PP-OCRv6_tiny" in name:
        return os.path.join(UTILS, "dict", "ppocrv6_tiny_dict.txt")
    return os.path.join(UTILS, "ppocr_keys_v1.txt")


def build(config, name):
    arch = config["Architecture"]
    post = build_post_process(config.get("PostProcess"), global_config=config.get("Global"))
    if hasattr(post, "character"):
        char_num = len(post.character)
        if arch.get("Head", {}).get("name") == "MultiHead":
            arch["Head"]["out_channels_list"] = {
                "CTCLabelDecode": char_num,
                "SARLabelDecode": char_num + 2,
                "NRTRLabelDecode": char_num + 4,
            }
        else:
            arch["Head"]["out_channels"] = char_num
    return BaseModel(arch)


def main():
    for cfg in sorted(glob.glob(os.path.join(ROOT, "configs", "*", "*.yml"))):
        name = os.path.splitext(os.path.basename(cfg))[0]
        kind = os.path.basename(os.path.dirname(cfg))
        w = os.path.join(OFFICIAL, name + "_ptocr.pth")
        if not os.path.isfile(w):
            continue
        config = load_config(cfg)
        if kind == "rec":
            config["Global"]["character_dict_path"] = char_dict_for(name)
        model = build(config, name)
        state = torch.load(w, map_location="cpu")
        own = model.state_dict()
        state = {
            k: v
            for k, v in state.items()
            if k in own and tuple(own[k].shape) == tuple(v.shape)
        }
        model.load_state_dict(state, strict=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        if kind == "det":
            x = np.random.RandomState(0).randn(1, 3, 640, 640).astype(np.float32)
        else:
            x = np.random.RandomState(0).randn(1, 3, 48, 320).astype(np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(x).to(device))
        y = y["maps"] if isinstance(y, dict) else y
        np.save(os.path.join(OFFICIAL, name + "_torch_out.npy"), y.detach().cpu().numpy())
        print("{:32s} torch out {}".format(name, tuple(y.shape)))


if __name__ == "__main__":
    main()
