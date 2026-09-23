"""Generate demo datasets (label text + placeholder images) for self-tests.

仓库不入库 datasets/（label 与图片均本地）;真实示例图片可打包在 ModelScope(见 ``datasets/manifest.yml``)。
本脚本生成**占位图片**,让 ``tools/smoke_all.py`` 在没有真实数据时也能跑通
(图片是随机噪声 + 简单图形,不具备真实标注语义,仅用于验证数据/模型/损失/指标链路)。

    python tools/make_demo_data.py --all
    python tools/make_demo_data.py --dataset plate_rec_demo

label 约定(与 datasets/README.md 一致):
  * YOLO 系(det/obb/pose/plate_det):``train.txt`` 只列图片路径,标注写在 ``labels/*.txt``
  * 单行数据集(cls/plate_rec/attribute):``train.txt`` 每行 ``路径 <标签...>``
  * OCR det:``路径\\t[{"transcription": "...", "points": [[x,y],...]}]``
  * OCR rec:``路径\\t文本``
"""
import argparse
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "datasets")
RNG = np.random.RandomState(0)


def _dirs(name, *sub):
    d = os.path.join(DS, name)
    for s in ("",) + sub:
        os.makedirs(os.path.join(d, s) if s else d, exist_ok=True)
    return d


def _img(h, w, kind="noise"):
    img = (RNG.rand(h, w, 3) * 120 + 80).astype(np.uint8)
    if kind == "plate":
        cv2.rectangle(img, (int(w * 0.15), int(h * 0.3)), (int(w * 0.85), int(h * 0.7)),
                      (40, 40, 220), -1)
    elif kind == "person":
        cv2.rectangle(img, (int(w * 0.3), int(h * 0.08)), (int(w * 0.7), int(h * 0.95)),
                      (200, 180, 160), -1)
    elif kind == "text":
        for c in range(6):
            cv2.putText(img, chr(65 + c), (6 + c * 22, int(h * 0.72)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def _save_img(d, rel, img):
    p = os.path.join(d, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    assert cv2.imwrite(p, img), "写图失败: " + p


def _write(d, split, lines):
    with open(os.path.join(d, split + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- YOLO 系 ----
def yolo_like(name, n_cls=2, kpt_dim=0, hw=(240, 320), kind="noise",
              box=(0.25, 0.45), angle=False, n_train=24, n_val=8):
    d = _dirs(name, "images", "labels")
    for split, n in (("train", n_train), ("val", n_val)):
        paths = []
        for i in range(n):
            h, w = hw
            cx, cy = RNG.uniform(0.3, 0.7), RNG.uniform(0.3, 0.7)
            bw, bh = RNG.uniform(*box), RNG.uniform(0.08, 0.15)
            vals = [float(RNG.randint(0, n_cls)), cx, cy, bw, bh]
            if angle:
                vals.append(float(RNG.uniform(0, np.pi / 3)))
            if kpt_dim:
                x1, y1 = (cx - bw / 2), (cy - bh / 2)
                x2, y2 = (cx + bw / 2), (cy + bh / 2)
                for kx, ky in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                    vals += [kx, ky] + ([1.0] if kpt_dim == 3 else [])
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(h, w, kind))
            with open(os.path.join(d, "labels", fn + ".txt"), "w", encoding="utf-8") as f:
                f.write(" ".join("%.6f" % v for v in vals) + "\n")
            paths.append("images/%s.jpg" % fn)
        _write(d, split, paths)
    print("  %-26s images + labels/" % name)


# ------------------------------------------------------- 单行标签数据集 ----
def cls_like(name, n_cls=3):
    d = _dirs(name, "images")
    for split, n in (("train", 12), ("val", 6)):
        lines = []
        for i in range(n):
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(96, 96))
            lines.append("images/%s.jpg %d" % (fn, i % n_cls))
        _write(d, split, lines)
    print("  %-26s images + 单行标签" % name)


def cls_ml_like(name, n_cls=4):
    """多标签 classify demo: path v1 v2 ... vC (0/1 multi-hot)."""
    d = _dirs(name, "images")
    for split, n in (("train", 12), ("val", 6)):
        lines = []
        for i in range(n):
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(96, 96))
            lb = (RNG.rand(n_cls) < 0.35).astype(int)
            if lb.sum() == 0:
                lb[i % n_cls] = 1
            lines.append("images/%s.jpg %s" % (fn, " ".join(str(v) for v in lb)))
        _write(d, split, lines)
    print("  %-26s images + 多标签(%d 维)" % (name, n_cls))


def plate_rec_demo(n_train=32, n_val=8):
    d = _dirs("plate_rec_demo", "images")
    for split, n in (("train", n_train), ("val", n_val)):
        lines = []
        for i in range(n):
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(48, 168, "plate"))
            chars = [RNG.randint(1, 78) for _ in range(7)]
            lines.append("images/%s.jpg %s %d"
                         % (fn, " ".join(str(c) for c in chars), RNG.randint(0, 5)))
        _write(d, split, lines)
    print("  %-26s images + 单行标签" % "plate_rec_demo")


def attribute_demo(name, n_attr, n_train=24, n_val=8):
    d = _dirs(name, "images")
    portrait = name.startswith("pedestrian")
    h, w = (256, 192) if portrait else (192, 256)
    for split, n in (("train", n_train), ("val", n_val)):
        lines = []
        for i in range(n):
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(h, w, "person"))
            lb = (RNG.rand(n_attr) < 0.3).astype(int)
            if lb.sum() == 0:
                lb[RNG.randint(n_attr)] = 1
            lines.append("images/%s.jpg %s" % (fn, " ".join(str(v) for v in lb)))
        _write(d, split, lines)
    print("  %-26s images + 单行标签(%d 属性)" % (name, n_attr))


# ------------------------------------------------------------------- OCR ----
def ocr_det_demo(n_train=8, n_val=4):
    d = _dirs("ocr_det_dataset_examples", "images")
    for split, n in (("train", n_train), ("val", n_val)):
        lines = []
        for i in range(n):
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(48, 200, "text"))
            pts = [[6, 6], [190, 6], [190, 44], [6, 44]]
            lines.append("images/%s.jpg\t%s" % (fn, json.dumps(
                [{"transcription": "ABCDEF", "points": pts}], ensure_ascii=False)))
        _write(d, split, lines)
    print("  %-26s images + OCR det 标签" % "ocr_det_dataset_examples")


def ocr_rec_demo(n_train=12, n_val=4):
    d = _dirs("ocr_rec_dataset_examples", "images")
    for split, n in (("train", n_train), ("val", n_val)):
        lines = []
        for i in range(n):
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(32, 160, "text"))
            text = "".join(chr(65 + RNG.randint(0, 6)) for _ in range(6))
            lines.append("images/%s.jpg\t%s" % (fn, text))
        _write(d, split, lines)
    print("  %-26s images + OCR rec 标签" % "ocr_rec_dataset_examples")


BUILDERS = {
    "det_demo": lambda: yolo_like("det_demo", 2),
    "obb_demo": lambda: yolo_like("obb_demo", 2, box=(0.2, 0.4), angle=True),
    "pose_demo": lambda: yolo_like("pose_demo", 1, kpt_dim=3, kind="person", n_train=12),
    "cls_demo": lambda: cls_like("cls_demo"),
    "cls_ml_demo": lambda: cls_ml_like("cls_ml_demo"),
    "plate_det_demo": lambda: yolo_like("plate_det_demo", 2, kpt_dim=2, kind="plate", n_train=12),
    "plate_rec_demo": plate_rec_demo,
    "pedestrian_attribute_demo": lambda: attribute_demo("pedestrian_attribute_demo", 26, 12, 4),
    "vehicle_attribute_demo": lambda: attribute_demo("vehicle_attribute_demo", 19, 12, 4),
    "ocr_det_dataset_examples": ocr_det_demo,
    "ocr_rec_dataset_examples": ocr_rec_demo,
}




# ------------------------------------------- seg / sem / depth ---------------
def seg_demo(name="seg_demo", n_train=12, n_val=4):
    """labels/<split>/xxx.txt: cls x1 y1 ... xn yn(归一化多边形)"""
    d = _dirs(name, "images", "labels")
    for split, n in (("train", n_train), ("val", n_val)):
        paths = []
        for i in range(n):
            h, w = 240, 320
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(h, w))
            cx, cy = RNG.uniform(0.3, 0.7), RNG.uniform(0.3, 0.7)
            bw, bh = RNG.uniform(0.2, 0.4), RNG.uniform(0.1, 0.25)
            poly = [(cx - bw / 2, cy - bh / 2), (cx + bw / 2, cy - bh / 2),
                    (cx + bw / 2, cy + bh / 2), (cx - bw / 2, cy + bh / 2)]
            vals = [float(RNG.randint(0, 2))] + [v for xy in poly for v in xy]
            with open(os.path.join(d, "labels", fn + ".txt"), "w", encoding="utf-8") as f:
                f.write(" ".join("%.6f" % v for v in vals) + "\n")
            paths.append("images/%s.jpg" % fn)
        _write(d, split, paths)
    print("  %-26s images + labels/(多边形)" % name)


def dense_demo(name, sub, n_train=12, n_val=4, kind="mask"):
    """sem/depth:images/<split>/xxx.jpg + <sub>/<split>/xxx.png"""
    d = _dirs(name, "images", sub)
    for split, n in (("train", n_train), ("val", n_val)):
        paths = []
        for i in range(n):
            h, w = 240, 320
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(h, w))
            if kind == "mask":
                m = (RNG.rand(h, w) * 3).astype(np.uint8)
            else:
                m = (RNG.rand(h, w) * 5000 + 500).astype(np.uint16)
            fp = os.path.join(d, sub, split)
            os.makedirs(fp, exist_ok=True)
            assert cv2.imwrite(os.path.join(fp, fn + ".png"), m)
            paths.append("images/%s.jpg" % fn)
        _write(d, split, paths)
    print("  %-26s images + %s/" % (name, sub))


BUILDERS["seg_demo"] = seg_demo
BUILDERS["sem_demo"] = lambda: dense_demo("sem_demo", "masks", kind="mask")
BUILDERS["depth_demo"] = lambda: dense_demo("depth_demo", "depth", kind="depth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.dataset:
        BUILDERS[args.dataset]()
        return
    if not args.all:
        ap.print_help()
        return
    print("生成 demo 数据集到", DS)
    for name, fn in BUILDERS.items():
        fn()
    print("注意:seg/sem/depth 需要真实掩码/深度数据(见 datasets/README.md)")


if __name__ == "__main__":
    main()
