"""Unified inference for the YOLO tasks (cls / det / obb / sem / depth / segment).

Examples::

    python tools/infer/predict_yolo.py -c configs/yolo/YOLO11n_det.yml --weights output/YOLO11n_det/best_accuracy.pth --input img.jpg
    python tools/infer/predict_yolo.py -c configs/yolo/YOLO11n_sem.yml --weights ... --input img.jpg --output out.png

The image is letterboxed to the training size (``Train.dataset.transform.image_size``),
results are mapped back to the original resolution.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import cv2
import numpy as np
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

from ptcore.precision import enable_paddle_like_precision  # noqa: E402
from ptcore.pretrained import resolve_pretrained  # noqa: E402
from torchkiln.trainer import build_task  # noqa: E402
from torchkiln.det.ops import letterbox  # noqa: E402
from torchkiln.det.rbox import rbox2poly_np  # noqa: E402
from torchkiln.ocr.utils.config import flatten_opts, parse_args_to_config  # noqa: E402

enable_paddle_like_precision()


def _names(config):
    ds = (config.get("Train", {}).get("dataset") or {})
    return ds.get("names")


def _load(config, weights, device):
    task = build_task(config)
    post = task.build_post_process(config)
    model = task.build_model(config, post)
    path = resolve_pretrained(weights)
    if not path:
        raise FileNotFoundError("Could not resolve weights: {}".format(weights))
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    own = model.state_dict()
    state = {
        k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)
    }
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return task, post, model


def _color(i):
    rng = np.random.RandomState(i * 9973 + 17)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def main():
    ap = argparse.ArgumentParser(description="YOLO task inference")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--opt", nargs="*", action="append", default=None)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="output/yolo_result.jpg")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    config = parse_args_to_config(args.config, flatten_opts(args.opt))
    task_name = (config.get("Architecture") or {}).get("task", "classify")
    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    task, post, model = _load(config, args.weights, device)

    img0 = cv2.imread(args.input)
    if img0 is None:
        raise FileNotFoundError("Cannot read image: {}".format(args.input))
    ds = (config.get("Train", {}).get("dataset") or {})
    size = int((ds.get("transform") or {}).get("image_size", 640))
    img, ratio, pad = letterbox(img0, size)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)
    names = _names(config)

    with torch.no_grad():
        raw = model(x)
        result = post(raw) if task_name not in ("semantic", "depth") else post(
            raw, size=(img.shape[0], img.shape[1])
        )

    vis = img0.copy()
    h0, w0 = img0.shape[:2]

    if task_name == "classify":
        probs = torch.softmax(raw, dim=1)[0].cpu().numpy()
        top = probs.argsort()[::-1][:5]
        print("top-5:", [(int(i), names[i] if names else int(i), round(float(probs[i]), 4)) for i in top])
        cv2.putText(vis, "{} {:.3f}".format(names[int(top[0])] if names else int(top[0]), probs[top[0]]),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    elif task_name in ("detect", "segment"):
        r = result[0]
        for box, score, lab in zip(r["bboxes"].cpu().numpy(), r["scores"].cpu().numpy(), r["labels"].cpu().numpy()):
            p1 = ((box[0] - pad[0]) / ratio, (box[1] - pad[1]) / ratio)
            p2 = ((box[2] - pad[0]) / ratio, (box[3] - pad[1]) / ratio)
            c = _color(int(lab))
            cv2.rectangle(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), c, 2)
            label = "{} {:.2f}".format(names[int(lab)] if names else int(lab), score)
            cv2.putText(vis, label, (int(p1[0]), max(12, int(p1[1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            if task_name == "segment" and r["masks"] is not None and r["masks"].shape[0]:
                idx = list(r["labels"].cpu().numpy()).index(lab)
                m = r["masks"][idx].cpu().numpy().astype(np.uint8)
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                m = m[int(pad[1]):int(pad[1] + h0 * ratio), int(pad[0]):int(pad[0] + w0 * ratio)]
                m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)
                vis[m > 0] = (0.5 * vis[m > 0] + 0.5 * np.array(c)).astype(np.uint8)

    elif task_name == "obb":
        r = result[0]
        for box, score, lab in zip(r["bboxes"].cpu().numpy(), r["scores"].cpu().numpy(), r["labels"].cpu().numpy()):
            bx = box.copy()
            bx[0] = (bx[0] - pad[0]) / ratio
            bx[1] = (bx[1] - pad[1]) / ratio
            bx[2] /= ratio
            bx[3] /= ratio
            poly = rbox2poly_np(bx[None])[0].astype(np.int32)
            c = _color(int(lab))
            cv2.polylines(vis, [poly], True, c, 2)
            cv2.putText(vis, "{} {:.2f}".format(names[int(lab)] if names else int(lab), score),
                        (int(poly[0][0]), max(12, int(poly[0][1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    elif task_name == "semantic":
        mask = result[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask[int(pad[1]):int(pad[1] + h0 * ratio), int(pad[0]):int(pad[0] + w0 * ratio)]
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        colored = np.zeros_like(vis)
        for c in np.unique(mask):
            if c == 0:
                continue
            colored[mask == c] = _color(int(c))
        vis = cv2.addWeighted(vis, 0.5, colored, 0.5, 0)

    elif task_name == "depth":
        d = result[0].cpu().numpy()
        d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        d = d[int(pad[1]):int(pad[1] + h0 * ratio), int(pad[0]):int(pad[0] + w0 * ratio)]
        d = cv2.resize(d, (w0, h0), interpolation=cv2.INTER_LINEAR)
        dn = ((d - d.min()) / max(1e-6, d.max() - d.min()) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
        print("depth range: {:.3f} .. {:.3f} m".format(float(d.min()), float(d.max())))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, vis)
    print("saved to", args.output)


if __name__ == "__main__":
    main()
