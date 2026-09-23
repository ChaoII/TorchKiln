"""Compare every pretrained yolo*.pth against the official ultralytics .pt.

Usage (ptocr env):
    python tools/compare_pretrained_ultra.py

Prints one line per weight file: key coverage + max|diff| on matched tensors.
Exit code 1 if any *paired* file fails alignment (missing official = WARN only).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from ptcore.pretrained import load_state_dict_any  # noqa: E402

DEFAULT_OURS = r"\\tsclient\E\TorchKiln\pretrained"
DEFAULT_OFFICIAL = r"\\tsclient\D\项目资料\ultralytics_models"


def _index_official(root):
    idx = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".pt"):
                stem = os.path.splitext(fn)[0]
                idx[stem] = os.path.join(dirpath, fn)
    return idx


def _stem_ours(fn):
    # yolo11n.pth -> yolo11n
    return os.path.splitext(fn)[0]


def _tensors(sd):
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().float().cpu()
        elif isinstance(v, dict):
            out.update(_tensors(v))
    return out


def compare_pair(ours_path, official_path):
    a = _tensors(load_state_dict_any(ours_path))
    b = _tensors(load_state_dict_any(official_path))
    ka, kb = set(a), set(b)
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)
    common = sorted(ka & kb)
    shape_mismatch = []
    maxdiff = 0.0
    worst_key = None
    n_cmp = 0
    for k in common:
        if tuple(a[k].shape) != tuple(b[k].shape):
            shape_mismatch.append(
                "{} ours={} ultra={}".format(k, tuple(a[k].shape), tuple(b[k].shape))
            )
            continue
        d = (a[k] - b[k]).abs().max().item()
        n_cmp += 1
        if d > maxdiff:
            maxdiff = d
            worst_key = k
    return {
        "n_ours": len(ka),
        "n_ultra": len(kb),
        "common": len(common),
        "only_ours": only_a,
        "only_ultra": only_b,
        "shape_mismatch": shape_mismatch,
        "maxdiff": maxdiff,
        "worst_key": worst_key,
        "n_cmp": n_cmp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=DEFAULT_OURS)
    ap.add_argument("--official", default=DEFAULT_OFFICIAL)
    ap.add_argument("--tol", type=float, default=1e-5, help="max|diff| pass threshold")
    ap.add_argument("--filter", default="", help="substring filter on stem")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    official = _index_official(args.official)
    files = sorted(
        f
        for f in os.listdir(args.ours)
        if f.endswith(".pth") and f.startswith("yolo")
    )

    ok, bad, no_official = [], [], []
    for fn in files:
        stem = _stem_ours(fn)
        if args.filter and args.filter not in stem:
            continue
        ours_path = os.path.join(args.ours, fn)
        off_path = official.get(stem)
        if off_path is None:
            no_official.append(stem)
            print("NO_OFFICIAL  {}".format(stem))
            continue
        try:
            r = compare_pair(ours_path, off_path)
        except Exception as exc:  # noqa: BLE001
            bad.append((stem, "load error: {}".format(exc)))
            print("LOAD_FAIL    {} -> {}".format(stem, exc))
            continue

        issues = []
        if r["only_ours"]:
            issues.append("only_ours={}".format(len(r["only_ours"])))
        if r["only_ultra"]:
            issues.append("only_ultra={}".format(len(r["only_ultra"])))
        if r["shape_mismatch"]:
            issues.append("shape_mis={}".format(len(r["shape_mismatch"])))
        if r["maxdiff"] > args.tol:
            issues.append("maxdiff={:.3e}".format(r["maxdiff"]))

        status = "OK  " if not issues else "FAIL"
        extra = ("; ".join(issues)) if issues else (
            "keys {}/{} common={} maxdiff={:.3e}".format(
                r["n_ours"], r["n_ultra"], r["common"], r["maxdiff"]
            )
        )
        print("{}  {:28s} {}".format(status, stem, extra))
        if status == "OK  ":
            ok.append(stem)
        else:
            bad.append((stem, "; ".join(issues)))
            if args.verbose:
                if r["only_ours"][:8]:
                    print("         only_ours:", ", ".join(r["only_ours"][:8]))
                if r["only_ultra"][:8]:
                    print("         only_ultra:", ", ".join(r["only_ultra"][:8]))
                for s in r["shape_mismatch"][:8]:
                    print("         shape:", s)
                if r["worst_key"]:
                    print("         worst:", r["worst_key"])

    print(
        "\n== summary ==  OK={} FAIL={} NO_OFFICIAL={} (tol={})".format(
            len(ok), len(bad), len(no_official), args.tol
        )
    )
    if no_official:
        print("  no official .pt:", ", ".join(no_official))
    if bad:
        print("  failures:")
        for name, why in bad:
            print("   - {}: {}".format(name, why))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
