"""训练对齐对照:同一份数据、同一套超参,分别跑「我们」与「原版 ultralytics」,输出对照表。

    python tools/parity_train.py --data datasets/dx_det --config configs/yolo/yolo11_graph.yml \
        --ours-weights yolo11n_state.pth --ultra-pt _downloads/ultralytics_pt/yolo11n.pt \
        --epochs 10 --imgsz 640 --batch 8 --nc 1

* 我们侧:走平台 trainer(读 ``--config``,数据用 ``-o`` 覆盖),最终 mAP 从控制台/日志抓取;
* 原版侧:在 ultralytics 环境里调用(使用 ``--ultra-python`` 指定的解释器),结果读 ``results.csv``;
* 输出:两边 mAP50 / mAP50-95 / 训练耗时对照 + 差值(目标:差值 < 1%)。

CPU/GPU 均可,默认 GPU 0;纯对照脚本,不改动任何模型/数据。
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, cwd=None, timeout=None):
    p = subprocess.run(
        cmd, cwd=cwd or ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def run_ours(args):
    cfg = os.path.join(ROOT, args.config)
    out = os.path.join(ROOT, "output", "_parity_ours")
    ds_rel = args.data.replace("\\", "/").rstrip("/")
    if os.path.isabs(ds_rel):
        ds_rel = os.path.relpath(ds_rel, ROOT).replace("\\", "/")
    cmd = [sys.executable, "-m", "torchkiln", args.task, "train", "-c", args.config,
           "-o", "Train.dataset.data_dir=%s" % ds_rel,
           "-o", "Eval.dataset.data_dir=%s" % ds_rel,
           "-o", "Architecture.Head.num_classes=%d" % args.nc,
           "-o", "Global.epoch_num=%d" % args.epochs,
           "-o", "Global.save_model_dir=%s" % out,
           "-o", "Train.dataset.transform.image_size=%d" % args.imgsz,
           "-o", "Eval.dataset.transform.image_size=%d" % args.imgsz,
           "-o", "Train.dataset.loader.batch_size_per_card=%d" % args.batch,
           "-o", "Global.print_batch_step=100"]
    if args.ours_weights:
        cmd += ["-o", "Global.pretrained_model=%s" % args.ours_weights]
    rc, log = _run(cmd)
    m = re.findall(r"Best (?:mask_)?mAP50-95 = ([0-9.]+)", log)
    m50 = re.findall(r"(?:box_)?mAP50: ([0-9.]+)", log)
    return {"rc": rc, "mAP50-95": float(m[-1]) if m else None,
            "mAP50": float(m50[-1]) if m50 else None, "log_tail": log[-400:]}


def run_ultra(args):
    out = os.path.join(ROOT, "output", "ultra_parity")
    code = (
        "import os; os.environ['YOLO_OFFLINE']='True'\n"
        "from ultralytics import YOLO\n"
        "m = YOLO(r'%s')\n"
        "m.train(data=r'%s', epochs=%d, imgsz=%d, batch=%d, workers=4, device=%s,\n"
        "        project=r'%s', name='ultra_parity', exist_ok=True, plots=False, val=True,\n"
        "        verbose=False, optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937,\n"
        "        weight_decay=0.0005, warmup_epochs=3.0, nbs=64, seed=1024)\n"
        % (args.ultra_pt, args.ultra_data_yaml or os.path.join(ROOT, "datasets", os.path.basename(args.data.rstrip("/\\")), "data.yaml"),
           args.epochs, args.imgsz, args.batch, args.device, os.path.join(ROOT, "output"))
    )
    rc, log = _run([args.ultra_python, "-c", code])
    csv_path = os.path.join(out, "results.csv")
    res = {"rc": rc, "mAP50": None, "mAP50-95": None, "log_tail": log[-400:]}
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            def _pick(tag):
                for k in last:
                    if k.strip().endswith(tag):
                        try:
                            return float(last[k])
                        except Exception:
                            return None
                return None
            res["mAP50-95"] = _pick("mAP50-95(M)") or _pick("mAP50-95(B)")
            res["mAP50"] = _pick("mAP50(M)") or _pick("mAP50(B)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/dx_det")
    ap.add_argument("--config", default="configs/yolo/yolo11_graph.yml")
    ap.add_argument("--ours-weights", default="yolo11n_state.pth")
    ap.add_argument("--ultra-pt", default="_downloads/ultralytics_pt/yolo11n.pt")
    ap.add_argument("--ultra-python",
                    default=r"C:\ProgramData\miniconda3\envs\ultralytics\python.exe")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--task", default="detect")
    ap.add_argument("--ultra-data-yaml", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--skip-ultra", action="store_true")
    args = ap.parse_args()

    ours = None if args.skip_ours else run_ours(args)
    ultra = None if args.skip_ultra else run_ultra(args)

    print("=" * 64)
    print("训练对齐对照  data=%s  epochs=%d  imgsz=%d  batch=%d" %
          (args.data, args.epochs, args.imgsz, args.batch))
    print("%-28s %10s %12s" % ("", "mAP50", "mAP50-95"))
    for name, r in (("ultralytics(原版)", ultra), ("torchkiln(我们)", ours)):
        if r:
            print("%-28s %10s %12s" % (name, r["mAP50"], r["mAP50-95"]))
    if ours and ultra and ours["mAP50-95"] and ultra["mAP50-95"]:
        d = ours["mAP50-95"] - ultra["mAP50-95"]
        print("差值(mAP50-95): %+.4f   %s" % (d, "OK(<0.01)" if abs(d) < 0.01 else "需继续对齐"))
    print("=" * 64)
    if args.skip_ours is False and ours and ours["rc"] != 0:
        print("我们侧非零退出,日志尾部:\n", ours["log_tail"])
    if args.skip_ultra is False and ultra and ultra["rc"] != 0:
        print("原版侧非零退出,日志尾部:\n", ultra["log_tail"])


if __name__ == "__main__":
    main()
