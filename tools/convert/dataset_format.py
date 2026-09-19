"""Dataset format converter (CPU only, no training).

支持的格式:
  ``yolo_det``  原生 ultralytics YOLO:``images/<split>/x.jpg`` + ``labels/<split>/x.txt``
                (每行 ``cls cx cy w h``,归一化)+ 可选 ``data.yaml``
  ``ocr_det``   PaddleOCR 检测:``train.txt``/``val.txt`` 每行
                ``image_path<TAB>[{"transcription": "", "points": [[x,y]*4], "difficult": false}]``
  ``ocr_rec``   PaddleOCR 识别:``train.txt``/``val.txt`` 每行 ``image_path<TAB>文本``
  ``cls``       单标签 ``image_path<TAB>类别号``

用法::

    # 原生 YOLO -> PaddleOCR 检测
    python tools/convert/dataset_format.py --from yolo_det --to ocr_det --src E:/dx_ocr/ultralytics --dst E:/dst/dx_det_ocr

    # PaddleOCR 检测 -> 原生 YOLO(需要类别号映射,默认全 0)
    python tools/convert/dataset_format.py --from ocr_det --to yolo_det --src ... --dst ... --nc 1 --names plate

    # 任意来源 -> 规整成原生 YOLO 布局(等于整理/校验)
    python tools/convert/dataset_format.py --from yolo_det --to yolo_det --src ... --dst ... [--hardlink]

所有读写都是文本/图片头操作,不占 GPU、不训练。
"""

import argparse
import io
import json
import os
import shutil

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val")


# --------------------------------------------------------------------- utils
def _rel_images(root, split):
    """返回 (相对路径列表, 图片绝对路径列表);兼容 images/<split> 与平铺 images/。"""
    for sub in (os.path.join("images", split), "images", ""):
        d = os.path.join(root, sub) if sub else root
        if not os.path.isdir(d):
            continue
        files = sorted(
            f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)
        )
        if files:
            rel = [os.path.join(sub, f).replace("\\", "/") if sub else f for f in files]
            return rel, [os.path.join(d, f) for f in files]
    return [], []


def _img_size(path):
    try:
        import cv2

        img = cv2.imread(path)
        if img is None:
            return None
        return img.shape[1], img.shape[0]  # w, h
    except Exception:
        return None


def _link_or_copy(src, dst, hardlink=True):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        if hardlink:
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _read_lines(p):
    if not os.path.isfile(p):
        return []
    with io.open(p, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _split_label_path(root, split, img_rel):
    """原生 YOLO:images/<split>/x.jpg -> labels/<split>/x.txt"""
    parts = img_rel.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return os.path.join(root, os.path.splitext("/".join(parts))[0] + ".txt")


# ------------------------------------------------------------ readers (解析)
def read_yolo_det(src, split):
    """-> [(img_rel, [(cls, cx, cy, w, h), ...])]"""
    rels, _ = _rel_images(src, split)
    out = []
    for rel in rels:
        lp = _split_label_path(src, split, rel)
        boxes = []
        for line in _read_lines(lp):
            p = line.split()
            if len(p) >= 5:
                boxes.append((int(float(p[0])), *[float(v) for v in p[1:5]]))
        out.append((rel, boxes))
    if not out:
        # 退化:标注内联在 <split>.txt
        for line in _read_lines(os.path.join(src, split + ".txt")):
            p = line.split()
            if len(p) >= 5:
                out.append((p[0], [(int(float(p[1])), *[float(v) for v in p[2:6]])]))
    return out


def read_ocr_det(src, split):
    """-> [(img_rel, [(cls, cx, cy, w, h), ...])]"""
    out = []
    for line in _read_lines(os.path.join(src, split + ".txt")):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        img_rel = parts[0].strip()
        try:
            insts = json.loads(parts[1])
        except Exception:
            continue
        w = h = None
        for cand in (os.path.join(src, img_rel), os.path.join(src, "images", os.path.basename(img_rel))):
            if os.path.isfile(cand):
                sz = _img_size(cand)
                if sz:
                    w, h = sz
                break
        if not w or not h:
            continue
        boxes = []
        for it in insts:
            pts = it.get("points") or []
            if len(pts) < 2:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            boxes.append((0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
        out.append((img_rel, boxes))
    return out


READERS = {"yolo_det": read_yolo_det, "ocr_det": read_ocr_det}


# ------------------------------------------------------------ writers (输出)
def write_yolo_det(dst, split, samples, src, hardlink=True):
    os.makedirs(os.path.join(dst, "labels", split), exist_ok=True)
    for img_rel, boxes in samples:
        name = os.path.basename(img_rel)
        s = None
        for cand in (os.path.join(src, img_rel), os.path.join(src, "images", name)):
            if os.path.isfile(cand):
                s = cand
                break
        if s is None:
            continue
        _link_or_copy(s, os.path.join(dst, "images", split, name), hardlink)
        with io.open(os.path.join(dst, "labels", split, os.path.splitext(name)[0] + ".txt"),
                     "w", encoding="utf-8", newline="") as f:
            for cls, cx, cy, bw, bh in boxes:
                f.write("%d %.6f %.6f %.6f %.6f\n" % (cls, cx, cy, bw, bh))
    io.open(os.path.join(dst, split + ".txt"), "w", encoding="utf-8", newline="").write(
        "\n".join("images/%s/%s" % (split, os.path.basename(r)) for r, _ in samples) + "\n"
    )


def write_ocr_det(dst, split, samples, src):
    lines = []
    for img_rel, boxes in samples:
        name = os.path.basename(img_rel)
        s = None
        for cand in (os.path.join(src, img_rel), os.path.join(src, "images", name)):
            if os.path.isfile(cand):
                s = cand
                break
        if s is None:
            continue
        _link_or_copy(s, os.path.join(dst, "images", name), hardlink=False)
        sz = _img_size(s)
        if not sz:
            continue
        w, h = sz
        insts = []
        for _cls, cx, cy, bw, bh in boxes:
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            insts.append({"transcription": "", "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                          "difficult": False})
        lines.append("images/%s\t%s" % (name, json.dumps(insts, ensure_ascii=False)))
    io.open(os.path.join(dst, split + ".txt"), "w", encoding="utf-8", newline="").write(
        "\n".join(lines) + "\n"
    )


WRITERS = {"yolo_det": write_yolo_det, "ocr_det": write_ocr_det}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src_fmt", required=True, choices=sorted(READERS))
    ap.add_argument("--to", dest="dst_fmt", required=True, choices=sorted(WRITERS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--names", default="object")
    ap.add_argument("--copy", action="store_true", help="默认硬链接,指定后复制")
    args = ap.parse_args()

    hardlink = not args.copy
    os.makedirs(args.dst, exist_ok=True)
    total = {}
    for split in args.splits.split(","):
        samples = READERS[args.src_fmt](args.src, split)
        n_box = sum(len(b) for _, b in samples)
        total[split] = (len(samples), n_box)
        WRITERS[args.dst_fmt](args.dst, split, samples, args.src, hardlink) if args.dst_fmt == "yolo_det" \
            else WRITERS[args.dst_fmt](args.dst, split, samples, args.src)
        print("  %-5s %d 图 / %d 框" % (split, len(samples), n_box))

    if args.dst_fmt == "yolo_det":
        names = [s.strip() for s in args.names.split(",")]
        io.open(os.path.join(args.dst, "data.yaml"), "w", encoding="utf-8", newline="").write(
            "# ultralytics dataset\n"
            "path: %s\ntrain: images/train\nval: images/val\nnc: %d\nnames:\n%s\n"
            % (args.dst.replace("\\", "/"), args.nc,
               "".join("  %d: %s\n" % (i, n) for i, n in enumerate(names)))
        )
    print("完成: %s -> %s  %s" % (args.src_fmt, args.dst_fmt, args.dst))


if __name__ == "__main__":
    main()
