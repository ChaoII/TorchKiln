"""Unified inference for the YOLO tasks (cls / det / obb / sem / depth / segment).

Examples::

    python tools/infer/predict_yolo.py -c configs/yolo/yolo11-det.yml --weights output/yolo11_g/best_accuracy.pth --input img.jpg
    python tools/infer/predict_yolo.py -c configs/yolo/yolo26-sem.yml --weights ... --input img.jpg --output out.png

The image is letterboxed to the training size (``Train.dataset.transform.image_size``),
results are mapped back to the original resolution.

⚠️ 推理逻辑**不在本文件**——letterbox、后处理、坐标回映射、各任务输出格式
都在 :mod:`torchkiln.infer_api` 里，``tkiln serve``（HTTP 推理服务）调的是同一份。
本脚本只做三件事：解析参数、读图、调 ``infer``、把结果画成图存盘。
若在下面再写一遍推理，就会出现"命令行与线上服务结果不一致"这种极难排查的问题。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import cv2

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

from torchkiln.infer_api import Runtime  # noqa: E402
from torchkiln.ocr.utils.config import flatten_opts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="YOLO task inference")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--opt", nargs="*", action="append", default=None)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="output/yolo_result.jpg")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    runtime = Runtime.load(args.config, args.weights, device=args.device,
                           overrides=flatten_opts(args.opt))

    img0 = cv2.imread(args.input)
    if img0 is None:
        raise FileNotFoundError("Cannot read image: {}".format(args.input))

    result = runtime.infer(img0)
    for line in result.notes:
        print(line)

    vis = runtime.visualize(img0, result)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.output, vis)
    print("saved to", args.output)


if __name__ == "__main__":
    main()
