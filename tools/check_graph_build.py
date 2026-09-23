"""Build every YOLO task config and run one dummy forward pass.

Usage:
    python tools/check_graph_build.py [--family v8] [--shape 320]
"""

import argparse
import glob
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import yaml  # noqa: E402

from ptcore.config import load_config  # noqa: E402
from torchkiln.models import build_arch_model  # noqa: E402

DEFAULT_SIZES = {
    "classify": 128,
    "semantic": 256,
    "depth": 256,
    "segment": 256,
    "obb": 320,
    "pose": 320,
    "detect": 320,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None, help="only configs whose name contains this")
    ap.add_argument("--shape", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(_ROOT, "configs", "yolo", "*.yml")))
    ok, fail = [], []
    for path in files:
        name = os.path.basename(path).replace(".yml", "")
        if args.family and args.family not in name:
            continue
        try:
            cfg = load_config(path)
            arch = cfg["Architecture"]
            task = str(arch.get("task", "detect"))
            model = build_arch_model(arch, task)
            size = args.shape or DEFAULT_SIZES.get(task, 320)
            model.eval()
            with torch.no_grad():
                out = model(torch.zeros(1, 3, size, size))
            n = out[0].shape[1] if isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]) else "-"
            ok.append(name)
            if args.verbose:
                print("  OK   {:26s} task={:9s} out1={}".format(name, task, n))
        except Exception as e:  # noqa: BLE001
            fail.append((name, "{}: {}".format(type(e).__name__, e)))
            print("  FAIL {:26s} {}: {}".format(name, type(e).__name__, str(e)[:110]))
            if args.verbose:
                traceback.print_exc()

    print("\n{} configs: {} OK, {} FAIL".format(len(ok) + len(fail), len(ok), len(fail)))
    for name, err in fail:
        print("  - {} -> {}".format(name, err))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
