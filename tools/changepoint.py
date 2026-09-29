"""变点/漂移检测 CLI：对 CSV 的一列（或多列平均）跑检测并输出 JSON。

用法:
    python -m torchkiln changepoint -c configs/ts/changepoint_demo.yml
或直接:
    python tools/changepoint.py <csv> --method cusum --column s1 --threshold 8
"""
from __future__ import absolute_import

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchkiln.ts_changepoint import detect_changepoints  # noqa: E402


def _load_csv(path, column=None, label_col="label"):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    body = np.asarray([[float(x) if x != "" else np.nan for x in r] for r in rows[1:]],
                      dtype=np.float64)
    drop = {0}
    if label_col in header:
        drop.add(header.index(label_col))
    if column and column in header:
        return np.nan_to_num(body[:, header.index(column)])
    cols = [i for i in range(len(header)) if i not in drop]
    return np.nanmean(body[:, cols], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="变点/漂移检测（CUSUM / Page-Hinkley / ADWIN / z-score）")
    ap.add_argument("csv", nargs="?", help="CSV 路径")
    ap.add_argument("-c", "--config", help="YAML 配置（含 data_dir/csv_path/method/...）")
    ap.add_argument("--method", default="cusum",
                    choices=["cusum", "page_hinkley", "adwin", "zscore"])
    ap.add_argument("--column", default=None, help="指定列名（缺省取所有数值列平均）")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--ref-frac", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--out", default=None, help="输出 JSON 路径（缺省打印）")
    args, _ = ap.parse_known_args(argv)

    cfg = {}
    if args.config:
        import yaml
        y = yaml.safe_load(open(args.config, encoding="utf-8"))
        cfg = (y.get("Detect") or y) or {}
        csvp = cfg.get("csv_path") or args.csv
        if csvp and not os.path.isabs(csvp) and cfg.get("data_dir"):
            csvp = os.path.join(cfg["data_dir"], csvp)
        args.csv = csvp
    if not args.csv:
        ap.error("需要 csv 路径（位置参数或 -c 配置里的 csv_path）")
    args.method = cfg.get("method", args.method)
    args.column = cfg.get("column", args.column)
    args.label_col = cfg.get("label_col", args.label_col)
    args.ref_frac = float(cfg.get("ref_frac", args.ref_frac))
    kw = {}
    if args.threshold is not None or cfg.get("threshold") is not None:
        kw["threshold"] = float(args.threshold if args.threshold is not None else cfg["threshold"])
    if args.delta is not None or cfg.get("delta") is not None:
        kw["delta"] = float(args.delta if args.delta is not None else cfg["delta"])
    if args.window is not None or cfg.get("window") is not None:
        kw["window"] = int(args.window if args.window is not None else cfg["window"])

    x = _load_csv(args.csv, args.column, args.label_col)
    res = detect_changepoints(x, method=args.method, ref_frac=args.ref_frac, **kw)
    out = {
        "csv": args.csv, "method": res["method"], "n": int(len(x)),
        "n_changepoints": len(res["indices"]),
        "changepoints": res["indices"][:200],
        "score_max": float(np.max(res["scores"])) if len(res["scores"]) else 0.0,
    }
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(txt)
        print("已写", args.out)
    print(txt)


if __name__ == "__main__":
    main()
