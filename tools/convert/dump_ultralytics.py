"""Dump an Ultralytics checkpoint into plain PyTorch tensors (run in the *ultralytics* env).

Upstream ``.pt`` files pickle the whole model object, so they cannot be read by a
PyTorch-only environment.  This script extracts a plain ``state_dict`` plus the
model YAML, which our loader can then consume:

    # 1) in an env that has ultralytics installed
    python tools/convert/dump_ultralytics.py yolo11n.pt --out _downloads/upstream

    # 2) in the ptocr env: fine-tune with the dumped weights
    python tools/train.py -c configs/yolo/yolo11-det.yml \
        -o Global.pretrained_model=_downloads/upstream/yolo11n.pth

Notes
-----
* Only weights whose names/shapes match our graph model are loaded; the rest are
  reported by the trainer as ``missing`` / skipped.
* The reg head of newer checkpoints is DFL-free (``reg_max = 1``); set
  ``Architecture.Head.reg_max: 1`` accordingly (the v11/v26 YAMLs already use the
  ``Detect26`` head for this).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import yaml


def main():
    ap = argparse.ArgumentParser(description="dump an Ultralytics .pt to plain tensors")
    ap.add_argument("weights", help="path to .pt, or a model name (e.g. yolo11n.pt)")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--name", default=None, help="output stem (default: weights basename)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    stem = args.name or os.path.splitext(os.path.basename(args.weights))[0]
    os.makedirs(args.out, exist_ok=True)

    model = YOLO(args.weights).model.float()
    state = model.state_dict()
    pth = os.path.join(args.out, stem + ".pth")
    torch.save(state, pth)

    yml = os.path.join(args.out, stem + ".yaml")
    spec = model.yaml if isinstance(model.yaml, dict) else {}
    with open(yml, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)

    print("saved:", pth, "({} tensors, {:.3f}M params)".format(
        len(state), sum(v.numel() for v in state.values()) / 1e6))
    print("saved:", yml)
    print("task:", getattr(model, "task", "?"), "| nc:", spec.get("nc"),
          "| scale:", spec.get("scale"))


if __name__ == "__main__":
    sys.exit(main())
