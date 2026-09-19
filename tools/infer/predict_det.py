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

from pytorchx.ocr.modeling.architectures.base_model import BaseModel
from pytorchx.ocr.postprocess import build_post_process
from pytorchx.ocr.utils.config import load_config
from pytorchx.ocr.utils.precision import enable_paddle_like_precision

enable_paddle_like_precision()


class TextDetector:
    def __init__(self, config_path, weights_path, device="cuda:0"):
        self.config = load_config(config_path)
        self.device = torch.device(
            device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        self.model = BaseModel(self.config["Architecture"]).to(self.device)
        from pytorchx.ocr.utils.pretrained import resolve_pretrained

        resolved = resolve_pretrained(weights_path)
        if not resolved:
            raise FileNotFoundError("Could not resolve weights: {}".format(weights_path))
        state = torch.load(resolved, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.post_process = build_post_process(self.config["PostProcess"])
        self.normalize = _Normalize()
        self.resize = _DetResizeForTest(limit_side_len=960, limit_type="max")

    @torch.no_grad()
    def __call__(self, img_bgr):
        src_h, src_w = img_bgr.shape[:2]
        img, ratio_h, ratio_w = self.resize(img_bgr)
        x = self.normalize(img).transpose(2, 0, 1)[None].astype(np.float32)
        tensor = torch.from_numpy(x).to(self.device)
        preds = self.model(tensor)
        shape_list = np.array([[src_h, src_w, ratio_h, ratio_w]], dtype=np.float32)
        result = self.post_process(preds, shape_list)
        return result[0]["points"]


class _Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img):
        return (img.astype(np.float32) / 255.0 - self.mean) / self.std


class _DetResizeForTest:
    def __init__(self, limit_side_len=960, limit_type="max"):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type

    def __call__(self, img):
        h, w = img.shape[:2]
        if self.limit_type == "max":
            ratio = 1.0
            if max(h, w) > self.limit_side_len:
                ratio = self.limit_side_len / (h if h > w else w)
        else:
            ratio = 1.0
            if min(h, w) < self.limit_side_len:
                ratio = self.limit_side_len / (h if h < w else w)
        resize_h = max(int(round(h * ratio / 32) * 32), 32)
        resize_w = max(int(round(w * ratio / 32) * 32), 32)
        resized = cv2.resize(img, (resize_w, resize_h))
        return resized, resize_h / float(h), resize_w / float(w)


def draw_det_res(img, boxes, save_path):
    canvas = img.copy()
    for box in boxes:
        box = np.array(box, dtype=np.int32)
        cv2.polylines(canvas, [box.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
    cv2.imwrite(save_path, canvas)


def main():
    parser = argparse.ArgumentParser(description="PyTorchOCR detection inference")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "-o", "--opt", nargs="*", action="append", default=None,
        help="config overrides, repeatable: -o a=1 -o b=2",
    )
    parser.add_argument("--weights", required=True, help="path to .pth")
    parser.add_argument("--input", required=True, help="image path")
    parser.add_argument("--output", default="output/det_result.jpg")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from pytorchx.ocr.utils.config import flatten_opts, parse_args_to_config

    config = parse_args_to_config(args.config, flatten_opts(args.opt))
    detector = TextDetector(args.config, args.weights, args.device)
    img = cv2.imread(args.input)
    assert img is not None, "cannot read {}".format(args.input)
    boxes = detector(img)
    print("detected {} boxes".format(len(boxes)))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    draw_det_res(img, boxes, args.output)
    print("saved to {}".format(args.output))


if __name__ == "__main__":
    main()
