"""Parity check: our re-implementations vs the upstream released weights / ONNX.

Usage::

    python tools/check_plate_models.py [--vendor <dir>] [--onnx <dir>]

``--vendor`` defaults to ``_downloads/Chinese_license_plate_detection_recognition``
and must contain ``models/yolov5n-0.5.yaml`` and ``weights/plate_detect.pt`` /
``weights/plate_rec_color.pth``.  ``--onnx`` points at the exported ONNX files
(optional but recommended - it validates the deployed decode numerically).
"""

import argparse
import io
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torchkiln.models import build_arch_model  # noqa: E402
from torchkiln.nn import plate as P  # noqa: E402

VENDOR = os.path.join(ROOT, "_downloads", "Chinese_license_plate_detection_recognition")


# --------------------------------------------------------------------------- #
# Unpickling upstream checkpoints without importing their tree
# --------------------------------------------------------------------------- #
def load_upstream_state(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "convert_plate_weights",
        __import__("os").path.join(__import__("os").path.dirname(__file__), "convert", "convert_plate_weights.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_upstream_state(path)


def _count(sd):
    return sum(int(np.prod(v.shape)) for v in sd.values() if hasattr(v, "shape"))


# --------------------------------------------------------------------------- #
# 1) detector
# --------------------------------------------------------------------------- #
def check_det(vendor, onnx_dir):
    print("=" * 72)
    yaml_path = os.path.join(vendor, "models", "yolov5n-0.5.yaml")
    spec = yaml.safe_load(io.open(yaml_path, encoding="utf-8").read())
    ours_yaml = os.path.join(ROOT, "torchkiln", "cfg", "models", "plate", "yolov5n-0.5.yaml")
    arch = {
        "yaml_file": os.path.relpath(ours_yaml, ROOT).replace("\\", "/"),
        "in_channels": 3,
        "Head": {"num_classes": 2},
    }
    model = build_arch_model(arch, "plate_det")
    print("yaml        : upstream nc=%s anchors=%d rows | ours built OK" % (spec.get("nc"), len(spec["anchors"])))

    sd, _ = load_upstream_state(os.path.join(vendor, "weights", "plate_detect.pt"))
    own = model.state_dict()
    matched = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    conflict = [(k, tuple(sd[k].shape), tuple(own[k].shape)) for k in sd if k in own and own[k].shape != sd[k].shape]
    missing = [k for k in own if k not in sd]
    extra = [k for k in sd if k not in own]
    print("checkpoint  : %d tensors, %s params" % (len(sd), "{:,}".format(_count(sd))))
    print("ours        : %s params" % "{:,}".format(sum(p.numel() for p in model.parameters())))
    print("matched     : %d | shape conflicts: %d | missing: %d | unused: %d" % (len(matched), len(conflict), len(missing), len(extra)))
    for c in conflict[:5]:
        print("   conflict", c)
    for k in missing[:5]:
        print("   missing ", k)
    for k in extra[:5]:
        print("   unused  ", k)
    res = model.load_state_dict(sd, strict=False)
    print("load_state_dict -> missing %d, unexpected %d" % (len(res.missing_keys), len(res.unexpected_keys)))

    model.eval()
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    cat = out[0] if isinstance(out, tuple) else out
    print("forward     :", tuple(cat.shape))
    return model, cat, x


# --------------------------------------------------------------------------- #
# 2) recogniser
# --------------------------------------------------------------------------- #
def check_rec(vendor, onnx_dir):
    print("=" * 72)
    sd, ck = load_upstream_state(os.path.join(vendor, "weights", "plate_rec_color.pth"))
    cfg = ck.get("cfg") or P.PLATE_REC_CFG_SMALL
    print("ckpt cfg    :", cfg)
    model = P.PlateRecNet(cfg=cfg, num_classes=len(P.PLATE_CHARSET), color_num=len(P.PLATE_COLORS), export=False)
    own = model.state_dict()
    matched = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    conflict = [k for k in sd if k in own and own[k].shape != sd[k].shape]
    missing = [k for k in own if k not in sd]
    extra = [k for k in sd if k not in own]
    print("checkpoint  : %d tensors, %s params" % (len(sd), "{:,}".format(_count(sd))))
    print("ours        : %s params" % "{:,}".format(sum(p.numel() for p in model.parameters())))
    print("matched     : %d | shape conflicts: %d | missing: %d | unused: %d" % (len(matched), len(conflict), len(missing), len(extra)))
    for k in conflict[:5]:
        print("   conflict", k)
    for k in missing[:5]:
        print("   missing ", k)
    res = model.load_state_dict(sd, strict=False)
    print("load_state_dict -> missing %d, unexpected %d" % (len(res.missing_keys), len(res.unexpected_keys)))

    model.eval()
    x = torch.rand(1, 3, 48, 168)
    with torch.no_grad():
        logp, color = model(x)
    print("forward     : logp %s color %s" % (tuple(logp.shape), tuple(color.shape)))
    return model, (logp, color), x


# --------------------------------------------------------------------------- #
# 3) optional numeric comparison against the shipped ONNX
# --------------------------------------------------------------------------- #
def compare_onnx(onnx_path, torch_out, x, atol=1e-3):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available - skipping ONNX comparison")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = sess.run(None, {name: x.numpy()})
    print("onnx        : %s -> %s" % (os.path.basename(onnx_path), [tuple(o.shape) for o in outs]))

    if isinstance(torch_out, (list, tuple)):
        torch_list = list(torch_out)
    else:
        torch_list = [torch_out]
    onnx_list = list(outs)
    if len(torch_list) == len(onnx_list):
        for i, (a, b) in enumerate(zip(torch_list, onnx_list)):
            a = a.detach().numpy() if torch.is_tensor(a) else np.asarray(a)
            if a.shape != b.shape:
                # tolerate (1,T,C) vs (T,1,C) permutations
                a = a.transpose(1, 0, 2) if a.ndim == 3 and a.shape[0] == 1 else a
            if a.shape == b.shape:
                d = float(np.abs(a - b).max())
                print("   output %d: shape %s, max|diff| = %.3e %s" % (i, tuple(a.shape), d, "OK" if d < atol else "MISMATCH"))
            else:
                print("   output %d: shape mismatch ours %s vs onnx %s" % (i, tuple(a.shape), tuple(b.shape)))
    else:
        a = torch_list[0].detach().numpy() if torch.is_tensor(torch_list[0]) else np.asarray(torch_list[0])
        b = onnx_list[0]
        d = float(np.abs(a - b).max()) if a.shape == b.shape else float("nan")
        print("   output 0: ours %s vs onnx %s | max|diff| = %.3e" % (tuple(a.shape), tuple(b.shape), d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", default=VENDOR)
    ap.add_argument(
        "--onnx",
        default=r"\\tsclient\E\CLionProjects\ModelDeploy\test_data\test_models\onnx\car_plate",
        help="directory holding yolov5plate.onnx / plate_recognition_color.onnx",
    )
    args = ap.parse_args()

    det_model, det_cat, det_x = check_det(args.vendor, args.onnx)
    det_onnx = os.path.join(args.onnx, "yolov5plate.onnx")
    if os.path.isfile(det_onnx):
        # the shipped detector ONNX returns a single concatenated tensor
        compare_onnx(det_onnx, det_cat, det_x, atol=3e-3)

    rec_model, rec_out, rec_x = check_rec(args.vendor, args.onnx)
    rec_onnx = os.path.join(args.onnx, "plate_recognition_color.onnx")
    if os.path.isfile(rec_onnx):
        model = rec_model
        model.export = True
        model.eval()
        with torch.no_grad():
            out = model(rec_x)
        model.export = False
        compare_onnx(rec_onnx, out, rec_x, atol=3e-3)


if __name__ == "__main__":
    main()
