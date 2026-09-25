"""Convert a KITTI 3D-detection subset into the torchkiln det3d layout.

KITTI ``label_2`` stores 3D boxes in the **rectified camera** frame; torchkiln
``det3d`` expects boxes in the **velodyne** frame as::

    cls  x y z  l w h  yaw

This script:
  1. picks the extracted velodyne frames,
  2. converts each ``label_2`` box (camera) -> velodyne via ``calib``,
  3. writes ``clouds/<split>/<id>.bin`` + ``labels/<split>/<id>.txt``
     + ``train.txt`` / ``val.txt`` under ``--out``.

Usage::

    python tools/convert/kitti_to_det3d.py \
        --kitti-root _downloads/paddle3d/kitti_unzip \
        --out datasets/kitti_det3d \
        --classes Car Pedestrian Cyclist
"""
import argparse
import os
import shutil

import numpy as np

KEEP_DEFAULT = ["Car", "Pedestrian", "Cyclist"]


def parse_calib(path):
    """Return 4x4 ``T_velo_from_cam`` (inverse of R0_rect @ Tr_velo_to_cam)."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ":" not in ln:
                continue
            key, rest = ln.split(":", 1)
            data[key.strip()] = np.array([float(t) for t in rest.split()])
    R0 = np.eye(4)
    R0[:3, :3] = data["R0_rect"].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3, :4] = data["Tr_velo_to_cam"].reshape(3, 4)
    T_cam_from_velo = R0 @ Tr
    return np.linalg.inv(T_cam_from_velo)


def convert_box(parts, name_to_id, T_velo_from_cam, drop_dontcare=True):
    name = parts[0]
    if name == "DontCare":
        return None
    if name not in name_to_id:
        return None
    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
    ry = float(parts[14])
    # KITTI location is the bottom face; lift to the box centre (camera y is down).
    center_cam = np.array([x, y - h / 2.0, z, 1.0])
    center_velo = T_velo_from_cam @ center_cam
    # KITTI box length runs along the camera x-axis (devkit compute_box_3d:
    # x_corners = +-l/2).  In the velodyne frame the length axis is therefore
    # ``ry + pi/2``; torchkiln det3d stores ``yaw`` as the length direction so
    # that ``l`` is the extent along ``yaw`` (matches Paddle3D CenterPoint,
    # whose box is ``(w, l, h, ry)`` with ``l`` along ``ry + pi/2``).
    yaw = ry + np.pi / 2.0
    return [
        name_to_id[name],
        center_velo[0],
        center_velo[1],
        center_velo[2],
        l,
        w,
        h,
        yaw,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kitti-root",
        default="_downloads/paddle3d/kitti_unzip",
        help="root containing training/{velodyne,label_2,calib}",
    )
    ap.add_argument("--out", default="datasets/kitti_det3d")
    ap.add_argument("--classes", nargs="+", default=KEEP_DEFAULT)
    args = ap.parse_args()

    root = args.kitti_root
    velo_dir = os.path.join(root, "training", "velodyne")
    lab_dir = os.path.join(root, "training", "label_2")
    calib_dir = os.path.join(root, "training", "calib")
    name_to_id = {c: i for i, c in enumerate(args.classes)}

    iset = os.path.join(os.path.dirname(root), "kitti", "ImageSets")
    if not os.path.isdir(iset):
        iset = os.path.join(root, "ImageSets")

    def read_ids(fn):
        p = os.path.join(iset, fn)
        if not os.path.isfile(p):
            return []
        return [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]

    # Only emit frames whose cloud was actually extracted.
    avail = {
        os.path.splitext(f)[0]
        for f in os.listdir(velo_dir)
        if f.endswith(".bin")
    }
    splits = {
        "train": [i for i in read_ids("train.txt") if i in avail],
        "val": [i for i in read_ids("val.txt") if i in avail],
    }

    for split, ids in splits.items():
        cloud_out = os.path.join(args.out, "clouds", split)
        label_out = os.path.join(args.out, "labels", split)
        os.makedirs(cloud_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)
        for i in ids:
            src = os.path.join(velo_dir, i + ".bin")
            dst = os.path.join(cloud_out, i + ".bin")
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)  # hardlink when on the same volume
                except OSError:
                    shutil.copyfile(src, dst)
            T = parse_calib(os.path.join(calib_dir, i + ".txt"))
            rows = []
            with open(os.path.join(lab_dir, i + ".txt"), encoding="utf-8") as f:
                for ln in f:
                    parts = ln.split()
                    if len(parts) < 15:
                        continue
                    box = convert_box(parts, name_to_id, T)
                    if box is not None:
                        rows.append(box)
            with open(os.path.join(label_out, i + ".txt"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(
                        "%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"
                        % (int(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7])
                    )
        with open(os.path.join(args.out, split + ".txt"), "w", encoding="utf-8") as f:
            for i in ids:
                f.write("clouds/%s/%s.bin\n" % (split, i))
        print("%s: %d frames" % (split, len(ids)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
