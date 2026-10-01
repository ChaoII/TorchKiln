"""Unified inference for the YOLO tasks (cls / det / obb / sem / depth / segment).

Examples::

    python tools/infer/predict_yolo.py -c configs/yolo/yolo11-det.yml --weights output/yolo11_g/best_accuracy.pth --input img.jpg
    python tools/infer/predict_yolo.py -c configs/yolo/yolo26-sem.yml --weights ... --input img.jpg --output out.png
    # --input 传目录则**批量**推理（模型只加载一次）
    python tools/infer/predict_yolo.py -c configs/yolo/yolo11-det.yml --weights ... --input imgs/ --output outdir/

The image is letterboxed to the training size (``Train.dataset.transform.image_size``),
results are mapped back to the original resolution.

--input 传目录时的行为
----------------------
``--input`` 可以是单张图，也可以是**目录**。传目录时**递归**遍历其中的图片，
逐张推理，把可视化结果写进 ``--output`` 指定的目录。

输出保持输入的**相对路径**（只把扩展名统一成 ``.jpg``），这一点是必须的：数据集目录
通常是 ``images/train/a.jpg`` + ``images/val/a.jpg`` 这种布局，重名很常见；若全部平铺
到 ``--output`` 下，后写入的会直接覆盖先写入的，表现为"跑完了但少了一半结果"。

为什么需要这个：AIStation 的「预测」功能把整个数据集目录挂到 ``/data``、期望框架
批量处理后由平台收集结果图。原本这里只接受单文件，于是 ``cv2.imread(目录)`` 返回
None → ``FileNotFoundError: Cannot read image: /data``，那条链路从来没跑通过。

关键点是**模型只加载一次**（``Runtime.load`` 在循环外）——按图起一次进程的话，
每张图都要重新加载权重与 CUDA 上下文，几百张图就是几百倍的无效开销。

⚠️ 推理逻辑**不在本文件**——letterbox、后处理、坐标回映射、各任务输出格式
都在 :mod:`torchkiln.infer_api` 里，``tkiln serve``（HTTP 推理服务）调的是同一份。
本脚本只做：解析参数、枚举输入、调 ``infer``、把结果画成图存盘。
若在下面再写一遍推理，就会出现"命令行与线上服务结果不一致"这种极难排查的问题。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import cv2

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

from ptcore.metrics_sink import build_sink, sink_enabled  # noqa: E402
from torchkiln.infer_api import Runtime  # noqa: E402
from torchkiln.ocr.utils.config import flatten_opts  # noqa: E402

#: 目录批量输入时认的图片后缀（小写比较）
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _list_images(input_path: str):
    """返回待处理图片列表。

    单文件 → ``[(路径, None)]``（``None`` 表示输出用 ``--output`` 原样路径）。
    目录 → 按相对路径排序的 ``[(绝对路径, 相对路径)]``，**递归**遍历。

    为什么递归：AIStation 挂进来的是数据集目录，图片在 ``images/train`` /
    ``images/val`` 等子目录里，只扫顶层会得到空列表（表现为
    ``No readable image under: /data``，看不出是层级不对）。

    为什么输出用相对路径：``images/train/a.jpg`` 与 ``images/val/a.jpg`` 重名很常见，
    平铺会互相覆盖。

    排序是刻意的：不排序的话同一次运行的输出顺序取决于文件系统枚举顺序，
    而 AIStation 那边会按文件名收集结果图再打包，顺序抖动会让回归难以比对。
    """
    if not os.path.isdir(input_path):
        return [(input_path, None)]
    items = []
    for root, dirs, files in os.walk(input_path):
        dirs.sort()
        for name in sorted(files):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            abspath = os.path.join(root, name)
            # 相对路径统一用 ``/``：它是**跨平台标识**（会被当对象存储 key、zip 内
            # 路径、比对基准），不能带 OS 分隔符——否则同一份产出在 Windows 上算出
            # ``a\\b.jpg``、在 Linux 上算出 ``a/b.jpg``，跨平台比对全废。
            # ``os.path.join`` 接受 ``/``，所以后面拼输出路径不受影响。
            rel = os.path.relpath(abspath, input_path).replace(os.sep, "/")
            items.append((abspath, rel))
    items.sort(key=lambda it: it[1])
    return items


def main():
    ap = argparse.ArgumentParser(description="YOLO task inference")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--opt", nargs="*", action="append", default=None)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True,
                    help="单张图片，或图片目录（目录则批量推理）")
    ap.add_argument("--output", default="output/yolo_result.jpg",
                    help="单图输入时是输出文件路径；目录输入时是输出目录")
    ap.add_argument("--metrics-dir", default=None,
                    help="metrics.jsonl 的写入目录。默认取 --output "
                         "（单图时取其所在目录）。**外部调度必须显式传**："
                         "它靠这个文件判终态")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # ⚠️ Runtime.load 必须在循环**外**：放进循环等于每张图重新加载权重与 CUDA 上下文。
    runtime = Runtime.load(args.config, args.weights, device=args.device,
                           overrides=flatten_opts(args.opt))

    # 指标契约：与训练/评估同一种写法，外部服务据此判终态。
    #
    # ⚠️ 预测**没有精度指标**——它的产物是结果图，不是数值。所以这里报的是
    #   「处理了多少张图 / 写出多少张 / 耗时」，不是 mAP 之类。
    #   但**必须**写 end 事件：服务侧的 classify_exit 把「进程退出且没有 end」
    #   判成 no_end_event / failed，不写就等于每次预测都被判失败。
    # metrics.jsonl 落在哪，必须是**确定**的：外部调度靠它判终态
    # （「进程退出且没有 end 事件」会被判成 no_end_event / failed）。
    # 而 --output 语义随输入而变（单图=文件、批量=目录），没法可靠地反推它的父目录，
    # 所以显式给 --metrics-dir。
    if args.metrics_dir:
        metrics_dir = args.metrics_dir
    elif os.path.isdir(args.output):
        metrics_dir = args.output
    else:
        metrics_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    sink = build_sink(metrics_dir, enabled=sink_enabled(True))
    started = time.time()

    try:
        items = _list_images(args.input)
        if not items:
            raise FileNotFoundError(
                "No readable image under: {} (looked for {})".format(
                    args.input, ", ".join(IMAGE_EXTS)))

        if len(items) > 1:
            # 目录输入：--output 当作输出目录
            os.makedirs(args.output, exist_ok=True)

        done = 0
        for img_path, rel in items:
            img0 = cv2.imread(img_path)
            if img0 is None:
                print("skip unreadable image: {}".format(img_path))
                continue

            result = runtime.infer(img0)
            for line in result.notes:
                print(line)

            vis = runtime.visualize(img0, result)
            if rel is None:
                out_path = args.output
            else:
                # 保持相对路径（只统一扩展名），避免 train/val 同名图互相覆盖
                rel_jpg = os.path.splitext(rel)[0] + ".jpg"
                out_path = os.path.join(args.output, rel_jpg)
                out_dir = os.path.dirname(os.path.abspath(out_path))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(out_path, vis)
            print("saved to", out_path)
            done += 1

        if done == 0:
            raise RuntimeError("All {} input image(s) were unreadable".format(len(items)))
    except Exception as exc:  # noqa: BLE001
        # 异常路径也必须收尾写 end，否则外部服务会永远等不到终态。
        sink.end(exit_reason="error",
                 error="{}: {}".format(type(exc).__name__, exc))
        raise

    elapsed = time.time() - started
    sink.predict(images_total=len(items), images_done=done,
                 elapsed_sec=round(elapsed, 3), output_dir=args.output)
    # ⚠️ 顺序不能反：end 会 close 文件。
    sink.end(exit_reason="finished", images_done=done,
             elapsed_sec=round(elapsed, 3))
    if len(items) > 1:
        print("done: {}/{} images -> {}".format(done, len(items), args.output))


if __name__ == "__main__":
    main()
