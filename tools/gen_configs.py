"""Generate TorchKiln configs from the official PaddleOCR configs.

Run with the PyTorch interpreter. Reads the official PaddleOCR configs from
E:\\TorchKiln\\_downloads\\PaddleOCR\\configs and emits adapted configs into
E:\\TorchKiln\\configs, pointing at the local example datasets.

Usage:
    python tools/gen_configs.py
"""
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "_downloads", "PaddleOCR", "configs")
DST = os.path.join(ROOT, "configs")
DET_DIR = os.path.join(ROOT, "datasets", "ocr_det_dataset_examples")
REC_DIR = os.path.join(ROOT, "datasets", "ocr_rec_dataset_examples")

DET_LIST = [
    ("PP-OCRv3/PP-OCRv3_mobile_det.yml", "PP-OCRv3_mobile_det"),
    ("PP-OCRv3/PP-OCRv3_server_det.yml", "PP-OCRv3_server_det"),
    ("PP-OCRv4/PP-OCRv4_mobile_det.yml", "PP-OCRv4_mobile_det"),
    ("PP-OCRv4/PP-OCRv4_server_det.yml", "PP-OCRv4_server_det"),
    ("PP-OCRv5/PP-OCRv5_mobile_det.yml", "PP-OCRv5_mobile_det"),
    ("PP-OCRv5/PP-OCRv5_server_det.yml", "PP-OCRv5_server_det"),
    ("PP-OCRv6/PP-OCRv6_tiny_det.yml", "PP-OCRv6_tiny_det"),
    ("PP-OCRv6/PP-OCRv6_small_det.yml", "PP-OCRv6_small_det"),
    ("PP-OCRv6/PP-OCRv6_medium_det.yml", "PP-OCRv6_medium_det"),
]

REC_LIST = [
    ("PP-OCRv3/PP-OCRv3_mobile_rec.yml", "PP-OCRv3_mobile_rec"),
    ("PP-OCRv4/PP-OCRv4_mobile_rec.yml", "PP-OCRv4_mobile_rec"),
    ("PP-OCRv4/PP-OCRv4_server_rec.yml", "PP-OCRv4_server_rec"),
    ("PP-OCRv5/PP-OCRv5_mobile_rec.yml", "PP-OCRv5_mobile_rec"),
    ("PP-OCRv5/PP-OCRv5_server_rec.yml", "PP-OCRv5_server_rec"),
    ("PP-OCRv6/PP-OCRv6_tiny_rec.yml", "PP-OCRv6_tiny_rec"),
    ("PP-OCRv6/PP-OCRv6_small_rec.yml", "PP-OCRv6_small_rec"),
    ("PP-OCRv6/PP-OCRv6_medium_rec.yml", "PP-OCRv6_medium_rec"),
]


GLOBAL_DEFAULTS = {
    "device": None,
    "distributed": False,
    "dist_backend": None,
    "dist_init_method": None,
    "seed": 1024,
    "use_ema": False,
    "ema_decay": 0.9998,
    "ema_decay_type": "threshold",
    "output": None,
    "show_eval_progress": True,
    "amp": False,
}

GLOBAL_ORDER = [
    "model_name",
    "debug",
    "use_gpu",
    "device",
    "epoch_num",
    "log_smooth_window",
    "print_batch_step",
    "save_model_dir",
    "save_epoch_step",
    "eval_batch_step",
    "cal_metric_during_train",
    "pretrained_model",
    "checkpoints",
    "save_inference_dir",
    "output",
    "use_visualdl",
    "infer_img",
    "character_dict_path",
    "max_text_length",
    "infer_mode",
    "use_space_char",
    "distributed",
    "dist_backend",
    "dist_init_method",
    "seed",
    "use_ema",
    "ema_decay",
    "ema_decay_type",
    "show_eval_progress",
    "amp",
    "save_res_path",
    "d2s_train_image_shape",
]


def _rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _pretrained_url(name):
    return (
        "https://www.modelscope.cn/models/ChaoII0987/PytorchOCR"
        "/resolve/master/pretrained/{}_ptocr.pth".format(name)
    )


def _reorder(d, order):
    out = {}
    for key in order:
        if key in d:
            out[key] = d[key]
    for key, value in d.items():
        if key not in out:
            out[key] = value
    return out


def _standardize(c):
    """Expose every ``-o``-overridable Global field in the emitted yml."""
    g = c.setdefault("Global", {})
    for key, value in GLOBAL_DEFAULTS.items():
        g.setdefault(key, value)
    c["Global"] = _reorder(g, GLOBAL_ORDER)
    if "Train" in c:
        c["Train"].setdefault("resume_path", None)
    return c


def gen_det(rel_src, name):
    with open(os.path.join(SRC, "det", rel_src), encoding="utf-8") as f:
        c = yaml.safe_load(f)
    g = c.setdefault("Global", {})
    g["model_name"] = name
    g["save_model_dir"] = "./output/{}".format(name)
    # Full ModelScope URL (same style as PaddleOCR configs). Downloaded into
    # ~/.torchkiln/ocr/pretrained on first use, then reused from cache.
    # Use `-o Global.pretrained_model=null` to train from scratch.
    g["pretrained_model"] = _pretrained_url(name)
    g["distributed"] = False
    for split, lbl in (("Train", "train.txt"), ("Eval", "val.txt")):
        ds = c[split]["dataset"]
        ds["data_dir"] = _rel(DET_DIR)
        ds["label_file_list"] = [_rel(os.path.join(DET_DIR, lbl))]
        ds.pop("ratio_list", None)
        ds["ratio_list"] = [1.0]
        c[split]["loader"]["num_workers"] = 4 if split == "Train" else 0
        c[split]["loader"]["prefetch_factor"] = 4
    # v6 uses a fixed 640 crop / DiceFocal loss etc; keep official values.
    return _standardize(c)

def _ensure_rec_resize(transforms, image_shape):
    for t in transforms:
        if "RecResizeImg" in t:
            return
    idx = len(transforms)
    for i, t in enumerate(transforms):
        if "KeepKeys" in t:
            idx = i
            break
    transforms.insert(idx, {"RecResizeImg": {"image_shape": image_shape}})


def gen_rec(rel_src, name):
    with open(os.path.join(SRC, "rec", rel_src), encoding="utf-8") as f:
        c = yaml.safe_load(f)
    g = c.setdefault("Global", {})
    g["model_name"] = name
    g["save_model_dir"] = "./output/{}".format(name)
    # Full ModelScope URL (same style as PaddleOCR configs). Downloaded into
    # ~/.torchkiln/ocr/pretrained on first use, then reused from cache.
    # Use `-o Global.pretrained_model=null` to train from scratch.
    g["pretrained_model"] = _pretrained_url(name)
    g["distributed"] = False
    g["character_dict_path"] = _rel(os.path.join(REC_DIR, "dict.txt"))
    image_shape = g.get("d2s_train_image_shape", [3, 48, 320])

    # Architecture: keep the official head exactly as Paddle defines it (the
    # NRTR/SAR auxiliary branch is trained together with CTC, matching Paddle).
    arch = c["Architecture"]

    for split, lbl in (("Train", "train.txt"), ("Eval", "val.txt")):
        ds = c[split]["dataset"]
        ds["name"] = "SimpleDataSet"
        ds["data_dir"] = _rel(REC_DIR)
        ds["label_file_list"] = [_rel(os.path.join(REC_DIR, lbl))]
        ds.pop("sampler", None)
        ds["ratio_list"] = [1.0]
        ds["ext_op_transform_idx"] = 1
        ds.pop("ds_width", None)
        _ensure_rec_resize(ds["transforms"], image_shape)
        c[split].pop("sampler", None)
        c[split]["loader"]["num_workers"] = 8
        c[split]["loader"].pop("shuffle", None)
        c[split]["loader"]["shuffle"] = split == "Train"
        c[split]["loader"]["num_workers"] = 4 if split == "Train" else 0
        c[split]["loader"]["prefetch_factor"] = 4
    return _standardize(c)


def main():
    for rel_src, name in DET_LIST:
        out = os.path.join(DST, "det", name + ".yml")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(gen_det(rel_src, name), f, sort_keys=False, allow_unicode=True)
        print("wrote", _rel(out))
    for rel_src, name in REC_LIST:
        out = os.path.join(DST, "rec", name + ".yml")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(gen_rec(rel_src, name), f, sort_keys=False, allow_unicode=True)
        print("wrote", _rel(out))


if __name__ == "__main__":
    main()
