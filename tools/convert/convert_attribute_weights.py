"""Convert the dumped PaddleX attribute weights into a plain torch ``state_dict``.

Run this in the **ptocr** environment (needs torch only)::

    python tools/convert/convert_attribute_weights.py \
        --pkl _downloads/official/ped_attr_paddle.pkl  --num-classes 26 \
        --out _downloads/official/PP-LCNet_x1_0_pedestrian_attribute_ptocr.pth

Key mapping (PaddleClas -> torchkiln):

    conv1.* / blocks*.*      -> backbone.<same>
    last_conv.weight / fc.*  -> head.<same>

Paddle stores BatchNorm stats as ``_mean``/``_variance`` (torch:
``running_mean``/``running_var``) and Linear weights as ``(out, in)`` (torch
needs ``(in, out)``), both handled here. Loads with missing=0/unexpected=0.
"""

import argparse
import os
import pickle
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def convert(pkl_path, num_classes):
    with open(pkl_path, "rb") as f:
        off = pickle.load(f)
    out = {}
    for k, v in off.items():
        v = torch.from_numpy(v)
        if k.startswith(("conv1", "blocks")):
            name = "backbone." + k
        elif k.startswith(("last_conv", "fc")):
            name = "head." + k
        else:
            name = k
        name = name.replace("._mean", ".running_mean").replace(
            "._variance", ".running_var"
        )
        if k.endswith("fc.weight"):
            v = v.t()  # Paddle Linear (out, in) -> torch (in, out)
        out[name] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    from torchkiln.nn.attribute import AttributeNet

    model = AttributeNet(num_classes=args.num_classes)
    sd = convert(args.pkl, args.num_classes)
    own = model.state_dict()
    matched = sum(
        1 for k, v in sd.items() if k in own and tuple(own[k].shape) == tuple(v.shape)
    )
    missing = [k for k in own if k not in sd and "num_batches_tracked" not in k]
    unused = [k for k in sd if k not in own]
    res = model.load_state_dict(sd, strict=False)
    print(
        "matched {}/{} | missing {} | unused {} | load missing {} unexpected {}".format(
            matched, len(own), len(missing), len(unused),
            len(res.missing_keys), len(res.unexpected_keys),
        )
    )
    if missing[:3]:
        print("  missing:", missing[:3])
    if unused[:3]:
        print("  unused :", unused[:3])
    if args.verify_only:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": sd}, args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
