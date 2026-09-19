"""Build every configured model and run a forward pass to catch architecture issues.

Run with the PyTorch interpreter.
"""
import glob
import os
import sys
import traceback

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pytorchx.ocr.modeling.architectures.base_model import BaseModel
from pytorchx.ocr.postprocess import build_post_process
from pytorchx.ocr.utils.config import load_config


def out_channels_list(config, post):
    char_num = len(post.character)
    if config.get("PostProcess", {}).get("name") == "SARLabelDecode":
        char_num -= 2
    if config.get("PostProcess", {}).get("name") == "NRTRLabelDecode":
        char_num -= 3
    return {
        "CTCLabelDecode": char_num,
        "SARLabelDecode": char_num + 2,
        "NRTRLabelDecode": char_num + 4,
    }


def main():
    ok = 0
    fail = 0
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "configs", "*", "*.yml"))):
        name = os.path.relpath(cfg_path, ROOT)
        try:
            config = load_config(cfg_path)
            arch = config["Architecture"]
            mtype = arch["model_type"]
            post = build_post_process(
                config.get("PostProcess"), global_config=config.get("Global")
            )
            if hasattr(post, "character") and arch.get("Head", {}).get("name") == "MultiHead":
                arch["Head"]["out_channels_list"] = out_channels_list(config, post)
            model = BaseModel(arch)
            model.eval()
            if mtype == "det":
                x = torch.randn(1, 3, 640, 640)
                y = model(x)
                shape = y["maps"].shape if isinstance(y, dict) else y.shape
            else:
                x = torch.randn(1, 3, 48, 320)
                y = model(x)
                shape = y.shape if torch.is_tensor(y) else {k: v.shape for k, v in y.items()}
            n = sum(p.numel() for p in model.parameters()) / 1e6
            print("OK  {:42s} params={:6.2f}M out={}".format(name, n, shape))
            ok += 1
        except Exception as e:
            print("FAIL {:42s} {}".format(name, e))
            traceback.print_exc()
            fail += 1
    print("\n{} OK, {} FAIL".format(ok, fail))


if __name__ == "__main__":
    main()
