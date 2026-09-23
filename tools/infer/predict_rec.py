from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

from torchkiln.ocr.modeling.architectures.base_model import BaseModel
from torchkiln.ocr.postprocess import build_post_process
from torchkiln.ocr.utils.config import flatten_opts, parse_args_to_config
from torchkiln.ocr.utils.precision import enable_paddle_like_precision

enable_paddle_like_precision()


def resize_norm_img(img, image_shape):
    img_c, img_h, img_w = image_shape
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
    resized = cv2.resize(img, (resized_w, img_h)).astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255.0
    resized -= 0.5
    resized /= 0.5
    padding = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding[:, :, :resized_w] = resized
    return padding


def main():
    parser = argparse.ArgumentParser(description="TorchKiln recognition inference")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "-o", "--opt", nargs="*", action="append", default=None,
        help="config overrides, repeatable: -o a=1 -o b=2",
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = parse_args_to_config(args.config, flatten_opts(args.opt))
    post = build_post_process(config["PostProcess"], global_config=config["Global"])
    char_num = len(post.character)
    arch = config["Architecture"]
    if arch.get("Head", {}).get("name") == "MultiHead":
        arch["Head"]["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "SARLabelDecode": char_num + 2,
            "NRTRLabelDecode": char_num + 4,
        }
    model = BaseModel(arch)
    from torchkiln.ocr.utils.pretrained import resolve_pretrained

    resolved = resolve_pretrained(args.weights)
    if not resolved:
        raise FileNotFoundError("Could not resolve weights: {}".format(args.weights))
    state = torch.load(resolved, map_location="cpu")
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    model.load_state_dict(state, strict=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    img = cv2.imread(args.input)
    assert img is not None, "cannot read {}".format(args.input)
    shape = config["Global"].get("d2s_train_image_shape", [3, 48, 320])
    x = resize_norm_img(img, shape)[None]
    with torch.no_grad():
        y = model(torch.from_numpy(x).to(device))
    y = y.detach().cpu().numpy() if torch.is_tensor(y) else y["ctc"].detach().cpu().numpy()
    result = post(y)
    print("{} -> {}".format(args.input, result[0][0]))


if __name__ == "__main__":
    main()
