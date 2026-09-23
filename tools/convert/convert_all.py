"""Convert all dumped official Paddle state dicts into PyTorch weights.

Run with the PyTorch interpreter (the `ptocr` conda env).
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from torchkiln.ocr.modeling.architectures.base_model import BaseModel
from torchkiln.ocr.postprocess import build_post_process
from torchkiln.ocr.utils.config import load_config

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


def rename(k):
    k = k.replace("._mean", ".running_mean").replace("._variance", ".running_var")
    # Paddle MobileNetV3 det uses `stage{N}.` while the torch port uses `stages.{N}.`
    k = re.sub(r"^backbone\.stage(\d+)\.",
               lambda m: "backbone.stages.{0}.".format(m.group(1)), k)
    # Paddle ResNet_vd uses `bb_{s}_{i}.` while the torch port nests it under `stages.{s}.`
    k = re.sub(r"^backbone\.bb_(\d+)_(\d+)\.",
               lambda m: "backbone.stages.{0}.bb_{0}_{1}.".format(m.group(1), m.group(2)), k)
    return k


def build_model(config):
    arch = config["Architecture"]
    post = build_post_process(config.get("PostProcess"), global_config=config.get("Global"))
    head = arch.get("Head", {})
    if hasattr(post, "character"):
        char_num = len(post.character)
        if head.get("name") == "MultiHead":
            head["out_channels_list"] = {
                "CTCLabelDecode": char_num,
                "SARLabelDecode": char_num + 2,
                "NRTRLabelDecode": char_num + 4,
            }
        else:
            head["out_channels"] = char_num
    return BaseModel(arch)


def convert(config_path, pkl_path, out_path, name):
    config = load_config(config_path)
    if config["Architecture"].get("model_type") == "rec":
        config.setdefault("Global", {})["character_dict_path"] = char_dict_for(name)
    model = build_model(config)
    tgt = model.state_dict()

    # Map each weight param to its owner module kind so that Linear weights are
    # always transposed (Paddle stores [in, out], torch stores [out, in]) even
    # when the matrix is square, while Embedding weights are copied verbatim.
    linear_weights = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight is not None:
            linear_weights.add(mod_name + ".weight")

    with open(pkl_path, "rb") as f:
        psd = pickle.load(f)

    mapped = 0
    transposed = 0
    unused = []
    mismatch = []
    used = set()
    for k, v in psd.items():
        if k.endswith("num_batches_tracked"):
            continue
        kk = rename(k)
        if kk not in tgt:
            unused.append(k)
            continue
        t = tgt[kk]
        if kk in linear_weights and v.ndim == 2:
            src = v.T
            is_t = True
        else:
            src = v
            is_t = False
        if tuple(t.shape) != tuple(src.shape):
            mismatch.append((k, tuple(v.shape), tuple(t.shape)))
            continue
        t.copy_(torch.from_numpy(np.ascontiguousarray(src)))
        if is_t:
            transposed += 1
        used.add(kk)
        mapped += 1

    missing = [k for k in tgt.keys() if k not in used and not k.endswith("num_batches_tracked")]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return {
        "mapped": mapped,
        "transposed": transposed,
        "unused": len(unused),
        "mismatch": mismatch,
        "missing": missing,
    }


def main():
    ok = 0
    for pkl in sorted(glob.glob(os.path.join(OFFICIAL, "*.pkl"))):
        name = os.path.basename(pkl)[: -len(".pkl")]
        if name.endswith("_pretrained"):
            name = name[: -len("_pretrained")]
        kind = "det" if name.endswith("_det") else "rec"
        cfg = os.path.join(ROOT, "configs", kind, name + ".yml")
        if not os.path.isfile(cfg):
            print("no config for", name)
            continue
        out = os.path.join(OFFICIAL, name + ".pth")
        try:
            r = convert(cfg, pkl, out, name)
            flag = "OK " if (not r["mismatch"] and not r["missing"]) else "WARN"
            print(
                "{} {:34s} mapped={} transposed={} unused={} mismatch={} missing={}".format(
                    flag, name, r["mapped"], r["transposed"], r["unused"],
                    len(r["mismatch"]), len(r["missing"]),
                )
            )
            if r["missing"][:5]:
                print("     missing e.g.", r["missing"][:5])
            if r["mismatch"][:5]:
                print("     mismatch e.g.", r["mismatch"][:5])
            ok += 1
        except Exception as e:
            import traceback

            print("FAIL", name, e)
            traceback.print_exc()
    print("\nconverted", ok, "models")


if __name__ == "__main__":
    main()
