"""Point-cloud pc_seg inference: load .npy/.bin → pillar class map PNG.

Example::

    python tools/infer/predict_pc.py -c configs/pc/pointpillars-seg.yml \\
        --weights output/pointpillars-seg/best_accuracy.pth \\
        --input datasets/pc_demo/clouds/val/val_000.npy --output out.png
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

from ptcore.pretrained import resolve_pretrained  # noqa: E402
from torchkiln.data.pc import load_cloud, pillarize  # noqa: E402
from torchkiln.trainer import build_task  # noqa: E402
from torchkiln.ocr.utils.config import flatten_opts, parse_args_to_config  # noqa: E402


def _palette(nc):
    rng = np.random.RandomState(3)
    pal = np.zeros((max(nc, 1), 3), dtype=np.uint8)
    pal[0] = (30, 30, 30)
    if nc > 1:
        pal[1:] = rng.randint(60, 255, size=(nc - 1, 3))
    return pal


def main():
    ap = argparse.ArgumentParser(description="pc_seg inference")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--opt", nargs="*", action="append", default=None)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True, help=".npy / .bin point cloud")
    ap.add_argument("--output", default="output/pc_seg_result.png")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    config = parse_args_to_config(args.config, flatten_opts(args.opt))
    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    task = build_task(config)
    post = task.build_post_process(config)
    model = task.build_model(config, post)
    path = resolve_pretrained(args.weights)
    if not path:
        raise FileNotFoundError("Could not resolve weights: {}".format(args.weights))
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    own = model.state_dict()
    state = {
        k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)
    }
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    ds = (config.get("Train", {}).get("dataset") or {})
    pc = load_cloud(args.input)
    bev, plabels = pillarize(
        pc,
        labels=None,
        pc_range=ds.get("pc_range"),
        pillar_size=ds.get("pillar_size"),
        num_classes=(config.get("Architecture", {}).get("Head") or {}).get("num_classes"),
        ignore_index=int(ds.get("ignore_index", 255)),
    )
    x = torch.from_numpy(bev)[None].to(device)
    with torch.no_grad():
        logits = model(x)
        pred = post(logits)  # (1, Ny, Nx)
    pred_np = pred[0].cpu().numpy().astype(np.int64)
    pred_np[plabels == int(ds.get("ignore_index", 255))] = -1

    nc = int((config.get("Architecture", {}).get("Head") or {}).get("num_classes", 5))
    pal = _palette(nc)
    h, w = pred_np.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(nc):
        vis[pred_np == c] = pal[c]
    # nearest-upscale for visibility
    scale = max(1, int(512 // max(h, 1)))
    if scale > 1:
        vis = np.kron(vis, np.ones((scale, scale, 1), dtype=np.uint8))

    try:
        import cv2

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        cv2.imwrite(args.output, vis[:, :, ::-1])
        print("saved", args.output, "grid", (h, w))
    except Exception as e:
        print("grid", (h, w), "unique", np.unique(pred_np), "save failed:", e)


if __name__ == "__main__":
    main()
