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


def dense_demo(name, sub, n_train=12, n_val=4, kind="mask", n_cls=3):
    """sem/depth/lane_seg: images/<split>/xxx.jpg + <sub>/<split>/xxx.png(同名)。"""
    d = _dirs(name, "images", sub)
    for split, n in (("train", n_train), ("val", n_val)):
        paths = []
        for i in range(n):
            h, w = 240, 320
            fn = "%03d" % i
            _save_img(d, "images/%s/%s.jpg" % (split, fn), _img(h, w))
            if kind == "mask":
                m = (RNG.rand(h, w) * n_cls).astype(np.uint8)
            else:
                m = (RNG.rand(h, w) * 5000 + 500).astype(np.uint16)
            fp = os.path.join(d, sub, split)
            os.makedirs(fp, exist_ok=True)
            assert cv2.imwrite(os.path.join(fp, fn + ".png"), m)
            paths.append("images/%s/%s.jpg" % (split, fn))
        _write(d, split, paths)
    print("  %-26s images + %s/" % (name, sub))


BUILDERS["seg_demo"] = seg_demo
BUILDERS["sem_demo"] = lambda: dense_demo("sem_demo", "masks", kind="mask")
BUILDERS["depth_demo"] = lambda: dense_demo("depth_demo", "depth", kind="depth")
# lane_seg 默认 Head.num_classes=2(bg+lane) -> 掩码只含 0/1
BUILDERS["lane_seg_demo"] = lambda: dense_demo(
    "lane_seg_demo", "masks", kind="mask", n_cls=2
)


def lane_row_demo(n_train=12, n_val=4, num_lanes=6, num_rows=100):
    """lane_row:images/<split>/*.jpg + labels/<split>/*.txt(每行 valid + R 个 x∈[0,1])。"""
    d = _dirs("lane_row_demo", "images", "labels")
    for split, n in (("train", n_train), ("val", n_val)):
        paths = []
        for i in range(n):
            h, w = 240, 320
            fn = "%s_%03d" % (split, i)
            _save_img(d, "images/%s.jpg" % fn, _img(h, w))
            lines = []
            for li in range(num_lanes):
                # 合成近似竖直的弯曲车道线;前 2~4 条有效
                if li < 2 + (i % 3):
                    xs = [
                        0.12 + 0.08 * li + 0.05 * np.sin(ri / max(num_rows - 1, 1) * np.pi)
                        for ri in range(num_rows)
                    ]
                    xs = [min(max(v, 0.0), 1.0) for v in xs]
                    lines.append("1 " + " ".join("%.4f" % v for v in xs))
                else:
                    lines.append("0 " + " ".join("0" for _ in range(num_rows)))
            with open(os.path.join(d, "labels", fn + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            paths.append("images/%s.jpg" % fn)
        _write(d, split, paths)
    print("  %-26s images + labels/(行式 L=%d R=%d)" % ("lane_row_demo", num_lanes, num_rows))


BUILDERS["lane_row_demo"] = lane_row_demo


def pc_seg_demo(n_train=12, n_val=4, n_points=4096, n_classes=5):
    """pc_seg demo: clouds/<split>/*.npy (N,4) + labels/<split>/*.npy + train/val.txt。

    空间分区生成多类点云,便于 pillar 多数投票后有可学的 mIoU。
    """
    d = _dirs("pc_demo", "clouds", "labels")
    for split, n in (("train", n_train), ("val", n_val)):
        paths = []
        for i in range(n):
            fn = "%s_%03d" % (split, i)
            # 地面平面 + 若干类簇,范围约 [-16,16]^2 × [-3,3]
            ground = np.column_stack(
                [
                    RNG.uniform(-16, 16, n_points // 2),
                    RNG.uniform(-16, 16, n_points // 2),
                    RNG.uniform(-1.2, -0.8, n_points // 2),
                    RNG.uniform(0, 0.2, n_points // 2),
                ]
            ).astype(np.float32)
            rest = n_points - ground.shape[0]
            # 簇中心随机,类 id = 1..n_classes-1
            centers = RNG.uniform(-12, 12, size=(n_classes - 1, 2))
            pts = []
            labs = []
            for ci, (cx, cy) in enumerate(centers, start=1):
                k = max(rest // (n_classes - 1), 1)
                pts.append(
                    np.column_stack(
                        [
                            RNG.normal(cx, 1.5, k),
                            RNG.normal(cy, 1.5, k),
                            RNG.uniform(-0.5, 2.5, k),
                            RNG.uniform(0.2, 1.0, k),
                        ]
                    ).astype(np.float32)
                )
                labs.append(np.full(k, ci, dtype=np.int64))
            pc = np.concatenate([ground] + pts, axis=0)
            lab = np.concatenate(
                [np.zeros(ground.shape[0], dtype=np.int64)] + labs, axis=0
            )
            # 裁到 range 内
            m = (
                (pc[:, 0] >= -16)
                & (pc[:, 0] < 16)
                & (pc[:, 1] >= -16)
                & (pc[:, 1] < 16)
                & (pc[:, 2] >= -3)
                & (pc[:, 2] < 3)
            )
            pc, lab = pc[m], lab[m]
            cp = os.path.join(d, "clouds", split)
            lp = os.path.join(d, "labels", split)
            os.makedirs(cp, exist_ok=True)
            os.makedirs(lp, exist_ok=True)
            np.save(os.path.join(cp, fn + ".npy"), pc)
            np.save(os.path.join(lp, fn + ".npy"), lab)
            paths.append("clouds/%s/%s.npy" % (split, fn))
        _write(d, split, paths)
    print("  %-26s clouds + labels/(N×4, %d cls)" % ("pc_demo", n_classes))


BUILDERS["pc_demo"] = pc_seg_demo


def det3d_demo(n_train=12, n_val=4, n_points=4096, n_cls=3):
    """det3d demo: clouds + labels/<split>/*.txt (cls x y z l w h yaw)."""
    names = ["car", "ped", "cyc"]
    d = _dirs("det3d_demo", "clouds", "labels")
    for split, n in (("train", n_train), ("val", n_val)):
        paths = []
        for i in range(n):
            fn = "%s_%03d" % (split, i)
            ground = np.column_stack(
                [
                    RNG.uniform(-16, 16, n_points // 2),
                    RNG.uniform(-16, 16, n_points // 2),
                    RNG.uniform(-1.2, -0.8, n_points // 2),
                    RNG.uniform(0.0, 0.2, n_points // 2),
                ]
            ).astype(np.float32)
            boxes = []
            pts = [ground]
            rest = n_points - ground.shape[0]
            k_each = max(rest // max(n_cls, 1), 1)
            for ci in range(n_cls):
                cx = float(RNG.uniform(-10, 10))
                cy = float(RNG.uniform(-10, 10))
                l = float(RNG.uniform(3.5, 4.5)) if ci == 0 else float(RNG.uniform(0.5, 0.8))
                w = float(RNG.uniform(1.5, 2.0)) if ci == 0 else float(RNG.uniform(0.5, 0.8))
                h = float(RNG.uniform(1.4, 1.8))
                z = float(RNG.uniform(-0.5, 0.5))
                yaw = float(RNG.uniform(-np.pi, np.pi))
                boxes.append([ci, cx, cy, z, l, w, h, yaw])
                pts.append(
                    np.column_stack(
                        [
                            RNG.normal(cx, max(l, 1) * 0.25, k_each),
                            RNG.normal(cy, max(w, 1) * 0.25, k_each),
                            RNG.normal(z, max(h, 1) * 0.2, k_each),
                            RNG.uniform(0.3, 1.0, k_each),
                        ]
                    ).astype(np.float32)
                )
            pc = np.concatenate(pts, axis=0)
            m = (
                (pc[:, 0] >= -16)
                & (pc[:, 0] < 16)
                & (pc[:, 1] >= -16)
                & (pc[:, 1] < 16)
                & (pc[:, 2] >= -3)
                & (pc[:, 2] < 3)
            )
            pc = pc[m]
            cp = os.path.join(d, "clouds", split)
            lp = os.path.join(d, "labels", split)
            os.makedirs(cp, exist_ok=True)
            os.makedirs(lp, exist_ok=True)
            np.save(os.path.join(cp, fn + ".npy"), pc)
            with open(os.path.join(lp, fn + ".txt"), "w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(
                        "%s %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"
                        % (names[int(b[0])], b[1], b[2], b[3], b[4], b[5], b[6], b[7])
                    )
            paths.append("clouds/%s/%s.npy" % (split, fn))
        _write(d, split, paths)
    print("  %-26s clouds + box txt (%d cls)" % ("det3d_demo", n_cls))


BUILDERS["det3d_demo"] = det3d_demo


def lane_bev_demo(n_train=12, n_val=4, hw=(576, 1024), bev=(200, 48),
                  out2d=(144, 256)):
    """BEV-LaneDet demo: camera image (.npy) + BEV lane GT (.npz)."""
    d = _dirs("lane_bev_demo", "images", "bev_gt")
    h, w = hw
    hb, wb = bev
    h2, w2 = out2d

    def make(stem):
        np.save(os.path.join(d, "images", stem + ".npy"),
                RNG.rand(3, h, w).astype(np.float32))
        seg = np.zeros((1, hb, wb), np.float32)
        inst = np.zeros((1, hb, wb), np.float32)
        for li in range(1, 5):
            c = RNG.randint(0, wb)
            seg[0, :, c] = 1.0
            inst[0, :, c] = li
        iseg = np.zeros((1, h2, w2), np.float32)
        for li in range(1, 5):
            iseg[0, :, RNG.randint(0, w2)] = 1.0
        np.savez(
            os.path.join(d, "bev_gt", stem + ".npz"),
            bev_seg=seg, bev_inst=inst,
            bev_off=RNG.rand(1, hb, wb).astype(np.float32),
            bev_z=RNG.rand(1, hb, wb).astype(np.float32),
            img_seg=iseg, img_inst=RNG.randint(0, 5, (1, h2, w2)).astype(np.float32))

    tr = ["%03d" % i for i in range(n_train)]
    va = ["%03d" % i for i in range(n_train, n_train + n_val)]
    for s in tr + va:
        make(s)
    _write(d, "train.txt", ["images/%s.npy" % s for s in tr])
    _write(d, "val.txt", ["images/%s.npy" % s for s in va])


BUILDERS["lane_bev_demo"] = lane_bev_demo


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
