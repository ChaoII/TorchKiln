from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from pytorchx.ocr.modeling.architectures.base_model import BaseModel
from pytorchx.ocr.postprocess import build_post_process
from pytorchx.ocr.utils.config import flatten_opts, load_config, parse_args_to_config


class ExportWrapper(nn.Module):
    """Returns plain tensor(s) so TorchScript / ONNX stay simple.

    * OCR det/rec  -> probability map / logits
    * YOLO cls/sem/depth -> single tensor
    * YOLO det/obb -> tuple of per-level tensors
    * YOLO seg     -> tuple of per-level tensors + prototypes
    """

    def __init__(self, model):
        super(ExportWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, dict):
            if "maps" in y:
                return y["maps"]
            if "ctc" in y:
                return y["ctc"]
            if "feats" in y:
                return tuple(y["feats"]) + (y["protos"],)
            return tuple(y.values())
        if isinstance(y, (list, tuple)):
            return tuple(y)
        return y


def build_ocr_model(config):
    arch = config["Architecture"]
    post = build_post_process(config.get("PostProcess"), global_config=config.get("Global"))
    head = arch.get("Head", {})
    if hasattr(post, "character"):
        char_num = len(post.character)
        if head.get("name") == "MultiHead":
            head["out_channels_list"] = {
                "CTCLabelDecode": char_num,
                "SARLabelDecode": char_num + 2,
                "NRTRLabelDecode": char_num + 4,
            }
        else:
            head["out_channels"] = char_num
    return BaseModel(arch)


def build_model(config):
    """Build the model for whatever ``Architecture.model_family`` says."""
    family = (config.get("Architecture") or {}).get("model_family", "ocr")
    if family == "yolo":
        from pytorchx.trainer import build_task

        task = build_task(config)
        post = task.build_post_process(config)
        return task.build_model(config, post)
    return build_ocr_model(config)


def input_shape_for(config, dummy_size=640):
    """(1, C, H, W) dummy input for tracing / ONNX export."""
    arch = config.get("Architecture") or {}
    if arch.get("model_family") == "yolo":
        ds = (config.get("Train", {}).get("dataset") or {})
        size = (ds.get("transform") or {}).get("image_size", dummy_size)
        return (1, 3, int(size), int(size))
    if arch.get("model_type") == "rec":
        shape = config.get("Global", {}).get("d2s_train_image_shape", [3, 48, 320])
        return (1, shape[0], shape[1], shape[2])
    return (1, 3, dummy_size, dummy_size)


def fuse_model(model):
    """Fuse re-parameterizable blocks (Conv+BN, rep branches)."""
    count = 0
    for m in model.modules():
        if hasattr(m, "rep") and callable(getattr(m, "rep")):
            try:
                m.rep()
                count += 1
            except Exception:
                pass
    return count


def main():
    parser = argparse.ArgumentParser(description="PyTorchOCR export")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "-o", "--opt", nargs="*", action="append", default=None,
        help="config overrides, repeatable: -o a=1 -o b=2",
    )
    parser.add_argument("--weights", required=True, help="training checkpoint (.pth)")
    parser.add_argument("--save-dir", required=True, help="output directory")
    parser.add_argument("--fuse", action="store_true", help="fuse rep/BN blocks")
    parser.add_argument("--onnx", action="store_true", help="also export ONNX")
    parser.add_argument(
        "--slim",
        action="store_true",
        help="run onnxslim on the exported ONNX (single-file, backend-friendly)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset (default: 18 for YOLO tasks, 11 for OCR)",
    )
    parser.add_argument(
        "--legacy-exporter",
        action="store_true",
        help="use the legacy torch.onnx exporter (dynamo=False)",
    )
    parser.add_argument(
        "--torchscript", action="store_true", default=True,
        help="export TorchScript (default true)",
    )
    args = parser.parse_args()

    config = parse_args_to_config(args.config, flatten_opts(args.opt))
    family = (config.get("Architecture") or {}).get("model_family", "ocr")
    opset = args.opset or (18 if family == "yolo" else 11)
    os.makedirs(args.save_dir, exist_ok=True)
    if family == "ocr":
        config.setdefault("Global", {}).setdefault("character_dict_path", None)

    model = build_model(config)
    from pytorchx.ocr.utils.pretrained import resolve_pretrained

    weights_path = resolve_pretrained(args.weights)
    if not weights_path:
        raise FileNotFoundError(
            "Could not resolve weights: {}".format(args.weights)
        )
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    model.load_state_dict(state, strict=False)
    model.eval()

    if args.fuse:
        n = fuse_model(model)
        print("fused {} rep modules".format(n))

    wrapper = ExportWrapper(model).eval()

    # 1) inference weights (state dict)
    pth_path = os.path.join(args.save_dir, "inference.pth")
    torch.save(model.state_dict(), pth_path)
    print("saved:", pth_path)

    # 2) inference config
    cfg_path = os.path.join(args.save_dir, "inference.yml")
    shutil.copyfile(args.config, cfg_path)
    print("saved:", cfg_path)

    # 3) TorchScript (traced) + 4) ONNX share the same dummy input
    dummy = torch.randn(*input_shape_for(config))
    with torch.no_grad():
        preview = wrapper(dummy)
    if isinstance(preview, tuple):
        output_names = ["output_{}".format(i) for i in range(len(preview))]
    else:
        output_names = ["output"]
    dynamic_axes = {"x": {0: "batch"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    with torch.no_grad():
        try:
            traced = torch.jit.trace(wrapper, dummy)
            ts_path = os.path.join(args.save_dir, "model.pt")
            traced.save(ts_path)
            print("saved:", ts_path)
        except Exception as e:
            print("TorchScript export failed:", e)

    if args.onnx:
        onnx_path = os.path.join(args.save_dir, "model.onnx")
        try:
            torch.onnx.export(
                wrapper,
                dummy,
                onnx_path,
                opset_version=opset,
                input_names=["x"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False if args.legacy_exporter else True,
            )
            print("saved:", onnx_path)
        except Exception as e:
            print("ONNX export failed:", e)

    if args.slim:
        try:
            import onnxslim

            src = os.path.join(args.save_dir, "model.onnx")
            dst = os.path.join(args.save_dir, "model_slim.onnx")
            in_shape = ",".join(str(int(v)) for v in dummy.shape)
            onnxslim.slim(
                src,
                dst,
                input_shapes=["x:" + in_shape],
                model_check=True,
            )
            print("saved:", dst)
        except Exception as e:
            print("onnxslim failed:", e)

    print("export done ->", args.save_dir)


if __name__ == "__main__":
    main()
