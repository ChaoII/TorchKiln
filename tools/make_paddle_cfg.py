"""Create PaddleOCR training configs pointing at our local datasets.

Run with the PyTorch interpreter (no Paddle needed).
"""
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "_downloads", "PaddleOCR", "configs")
OUT = os.path.join(ROOT, "_downloads", "paddle_cfg")
DET = os.path.join(ROOT, "datasets", "ocr_det_dataset_examples")
REC = os.path.join(ROOT, "datasets", "ocr_rec_dataset_examples")


def rel(p):
    return os.path.abspath(p).replace("\\", "/")


def make(kind, src, name, epochs, output, eval_step):
    cfg = yaml.safe_load(open(os.path.join(SRC, kind, src), encoding="utf-8"))
    g = cfg["Global"]
    g["epoch_num"] = epochs
    g["eval_batch_step"] = [0, eval_step]
    g["save_model_dir"] = output
    g["distributed"] = False
    g["use_visualdl"] = False
    if kind == "det":
        d = DET
        g["save_res_path"] = os.path.join(output, "pred.txt")
        g["pretrained_model"] = rel(
            os.path.join(ROOT, "_downloads", "official",
                         "PP-OCRv4_mobile_det_pretrained.pdparams")
        )
    else:
        d = REC
        g["character_dict_path"] = rel(os.path.join(REC, "dict.txt"))
        g["pretrained_model"] = None
    for split, lbl in (("Train", "train.txt"), ("Eval", "val.txt")):
        ds = cfg[split]["dataset"]
        ds["data_dir"] = rel(d)
        ds["label_file_list"] = [rel(os.path.join(d, lbl))]
        if "sampler" in cfg[split]:
            cfg[split]["sampler"]["is_training"] = split == "Train"
        cfg[split]["loader"]["num_workers"] = 0
        if kind == "rec":
            cfg[split]["loader"]["batch_size_per_card"] = 16 if split == "Train" else 64
    cfg["Train"]["loader"]["batch_size_per_card"] = 8 if kind == "det" else 16
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, name + ".yml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print("wrote", rel(out))


if __name__ == "__main__":
    make("det", "PP-OCRv4/PP-OCRv4_mobile_det.yml", "det_v4_mobile", 20,
         rel(os.path.join(ROOT, "output", "paddle_det_v4")), 25)
    make("rec", "PP-OCRv4/PP-OCRv4_mobile_rec.yml", "rec_v4_mobile", 30,
         rel(os.path.join(ROOT, "output", "paddle_rec_v4")), 200)
