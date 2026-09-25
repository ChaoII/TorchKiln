"""为每个任务生成"数据集格式示例"(纯 CPU,不训练、不占 GPU)。

产物在 ``datasets/_format_examples/``:每个任务一个子目录,内含**可直接照抄的**:
  * 组织结构(images/ labels/ masks/ depth/ ... + train.txt / val.txt)
  * 标注文件示例(1~3 行真实内容)
  * ``README.txt``:该任务的字段说明 + 对应配置片段

    python tools/make_format_examples.py            # 全部
    python tools/make_format_examples.py --task pose --task detect
"""

import argparse
import io
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "datasets", "_format_examples")


def _w(path, lines):
    p = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")


def _img(path, w=64, h=48):
    """占位图(可换成真实图;这里只为让示例目录可直接跑通)。"""
    try:
        import cv2

        p = os.path.join(OUT, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        cv2.imwrite(p, np.full((h, w, 3), 160, np.uint8))
    except Exception:
        pass


def _note(task, todo, extra):
    return [
        "任务: %s" % task,
        "",
        "组织形式:",
        todo,
        "",
        extra,
        "",
        "(本目录由 tools/make_format_examples.py 生成,可整体拷走当作模板)",
    ]


def ocr_det():
    _img("ocr_det/images/0001.jpg")
    _w("ocr_det/train.txt", [
        'images/0001.jpg\t[{"transcription": "ABC123", "points": [[10, 10], [50, 10], [50, 30], [10, 30]], "difficult": false}]'])
    _w("ocr_det/val.txt", [
        'images/0001.jpg\t[{"transcription": "ABC123", "points": [[10, 10], [50, 10], [50, 30], [10, 30]], "difficult": false}]'])
    _w("ocr_det/README.txt", _note(
        "ocr_det",
        "images/<任意名>.jpg  +  train.txt / val.txt(内联 JSON)",
        "train.txt 每行: 图片相对路径 <TAB> [{\"transcription\": 文本, \"points\": [[x,y]×4](像素), \"difficult\": false}]\n"
        "配置: Architecture.task=det(data_dir=datasets/_format_examples/ocr_det)"))


def ocr_rec():
    _img("ocr_rec/images/w0001.jpg", 160, 32)
    _w("ocr_rec/train.txt", ["images/w0001.jpg\tABC123"])
    _w("ocr_rec/val.txt", ["images/w0001.jpg\tABC123"])
    _w("ocr_rec/dict.txt", ["blank", "A", "B", "C", "1", "2", "3"])
    _w("ocr_rec/README.txt", _note(
        "ocr_rec",
        "images/*.jpg + train.txt / val.txt + dict.txt(字符集)",
        "train.txt 每行: 路径 <TAB> 文本\n配置: Architecture.task=rec,Global.character_dict_path=.../dict.txt"))


def yolo_det():
    for s in ("train", "val"):
        _img("det/images/%s/0001.jpg" % s)
        _w("det/labels/%s/0001.txt" % s, ["0 0.500000 0.500000 0.300000 0.200000"])
        _w("det/%s.txt" % s, ["images/%s/0001.jpg" % s])
    _w("det/data.yaml", ["path: datasets/_format_examples/det", "train: images/train",
                         "val: images/val", "nc: 1", "names:", "  0: plate"])
    _w("det/README.txt", _note(
        "detect / obb",
        "images/<split>/*.jpg + labels/<split>/*.txt + data.yaml + train.txt/val.txt",
        "每行: cls cx cy w h   (全部归一化 0~1)\n"
        "旋转框(obb): cls x1 y1 x2 y2 x3 y3 x4 y4(4角点归一化;配置加 Train.dataset.box_format=xywhr)",
        "  与 detect 相同目录;box_format=xywhr 时每行必须 9 个数,<9 会被整行丢弃(torchkiln/data/det.py)",
        "  内部再经 poly2rbox 转成 xywhr;不要写 cls cx cy w h angle 的 6 字段格式"))


def yolo_pose():
    for s in ("train", "val"):
        _img("pose/images/%s/0001.jpg" % s)
        _w("pose/labels/%s/0001.txt" % s, [
            "0 0.5 0.5 0.4 0.6 0.46 0.12 2 0.54 0.12 2 0.46 0.30 2 0.54 0.30 2"])
        _w("pose/%s.txt" % s, ["images/%s/0001.jpg" % s])
    _w("pose/README.txt", _note(
        "pose / plate_det",
        "images/<split>/*.jpg + labels/<split>/*.txt",
        "每行: cls cx cy w h 然后 K 组 (px py v);v: 0=未标注 1=遮挡 2=可见\n"
        "配置: Architecture.Head.kpt_shape=[17,3]\n"
        "车牌四角点(无可见性): kpt_shape=[4,2] -> cls cx cy w h p1x p1y p2x p2y p3x p3y p4x p4y"))


def yolo_seg():
    for s in ("train", "val"):
        _img("segment/images/%s/0001.jpg" % s)
        _w("segment/labels/%s/0001.txt" % s, ["0 0.10 0.20 0.40 0.20 0.40 0.55 0.10 0.55"])
        _w("segment/%s.txt" % s, ["images/%s/0001.jpg" % s])
    _w("segment/README.txt", _note(
        "segment",
        "images/<split>/*.jpg + labels/<split>/*.txt(多边形)",
        "每行: cls x1 y1 x2 y2 ...(归一化多边形,≥3 点)\n"
        "或用位图掩码: masks/<split>/0001.png(255=前景)"))


def yolo_cls():
    for s in ("train", "val"):
        _img("classify/images/%s/0001.jpg" % s, 96, 96)
        _w("classify/%s.txt" % s, ["images/%s/0001.jpg 0" % s])
    _w("classify/README.txt", _note(
        "classify", "images/<split>/*.jpg + train.txt / val.txt",
        "每行: 路径 类别号(0 起)"))


def yolo_sem():
    for s in ("train", "val"):
        _img("semantic/images/%s/0001.jpg" % s)
        try:
            import cv2
            p = os.path.join(OUT, "semantic/masks/%s/0001.png" % s)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, np.zeros((48, 64), np.uint8))   # 像素值=类别索引
        except Exception:
            pass
        _w("semantic/%s.txt" % s, ["images/%s/0001.jpg" % s])
    _w("semantic/README.txt", _note(
        "semantic",
        "images/<split>/*.jpg + masks/<split>/*.png(同名)+ train.txt/val.txt",
        "masks: uint8 单通道,像素值=类别索引;忽略像素由 ignore_index 指定(默认 255)"))


def yolo_lane_seg():
    for s in ("train", "val"):
        _img("lane_seg/images/%s/0001.jpg" % s)
        try:
            import cv2
            p = os.path.join(OUT, "lane_seg/masks/%s/0001.png" % s)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            m = np.zeros((48, 64), np.uint8)
            m[:, 20:26] = 1   # 竖直车道带
            cv2.imwrite(p, m)
        except Exception:
            pass
        _w("lane_seg/%s.txt" % s, ["images/%s/0001.jpg" % s])
    _w("lane_seg/README.txt", _note(
        "lane_seg(车道线-分割式)",
        "images/<split>/*.jpg + masks/<split>/*.png(同名)+ train.txt/val.txt",
        "masks: uint8,0=背景,1..C=车道类(可多类);ignore_index 默认 255\n"
        "配置: Architecture.task: lane_seg, Head.num_classes=C+1\n"
        "Loss: LaneSegLoss(CE+dice+focal); Metric main_indicator: lane_IoU"))


def yolo_lane_row():
    num_lanes, num_rows = 6, 100
    for s in ("train", "val"):
        _img("lane_row/images/%s/0001.jpg" % s)
        lines = []
        for li in range(num_lanes):
            if li < 2:
                xs = [0.15 + 0.2 * li + 0.05 * (ri / max(num_rows - 1, 1)) for ri in range(num_rows)]
                lines.append("1 " + " ".join("%.4f" % v for v in xs))
            else:
                lines.append("0 " + " ".join("0" for _ in range(num_rows)))
        _w("lane_row/labels/%s/0001.txt" % s, lines)
        _w("lane_row/%s.txt" % s, ["images/%s/0001.jpg" % s])
    _w("lane_row/README.txt", _note(
        "lane_row(车道线-行式/UFLD)",
        "images/<split>/*.jpg + labels/<split>/*.txt(同名)+ train.txt/val.txt",
        "每行一条车道(固定 num_lanes 行): valid x0 x1 ... x_{R-1}\n"
        "valid∈{0,1}; x∈[0,1] 归一化横坐标; R=num_rows(默认 100)\n"
        "配置: Architecture.task: lane_row, Head.num_lanes/num_rows/num_bins\n"
        "Loss: LaneRowLoss; Metric main_indicator: F1(threshold_px=50)"))


def yolo_depth():
    for s in ("train", "val"):
        _img("depth/images/%s/0001.jpg" % s)
        try:
            import cv2
            p = os.path.join(OUT, "depth/depth/%s/0001.png" % s)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, np.full((48, 64), 1000, np.uint16))   # 1000 = 1m (depth_scale=1000)
        except Exception:
            pass
        _w("depth/%s.txt" % s, ["images/%s/0001.jpg" % s])
    _w("depth/README.txt", _note(
        "depth",
        "images/<split>/*.jpg + depth/<split>/*.png(同名)+ train.txt/val.txt",
        "depth: uint16 单通道,数值=深度;Train.dataset.depth_scale 默认 1000(=毫米)"))


def plate_rec():
    _img("plate_rec/images/0001.jpg", 168, 48)
    _w("plate_rec/train.txt", ["images/0001.jpg 5 53 52 60 49 45 43 1"])
    _w("plate_rec/val.txt", ["images/0001.jpg 5 53 52 60 49 45 43 1"])
    _w("plate_rec/README.txt", _note(
        "plate_rec",
        "images/*.jpg(48×168)+ train.txt / val.txt",
        "每行: 路径 c1..c7 颜色号\n"
        "c: 字符表下标(torchkiln/nn/plate.py::PLATE_CHARSET,78 项,0=CTC blank)\n"
        "颜色: 0黑 1蓝 2绿 3白 4黄;双层牌设 Train.dataset.double_plate=true"))


def attribute():
    _img("attribute/images/0001.jpg", 192, 256)
    lb = " ".join(str(int(i in (1, 3, 10))) for i in range(26))
    _w("attribute/train.txt", ["images/0001.jpg " + lb])
    _w("attribute/val.txt", ["images/0001.jpg " + lb])
    _w("attribute/README.txt", _note(
        "attribute(行人 26 / 车辆 19)",
        "images/*.jpg + train.txt / val.txt",
        "每行: 路径 v1..vC(每位 1=有该属性,0=无;可用 0~1 软标签)\n"
        "行人 C=26(pedestrian_attribute_label_list.txt),车辆 C=19"))


def pose_action():
    os.makedirs(os.path.join(OUT, "pose_action/seqs"), exist_ok=True)
    np.save(os.path.join(OUT, "pose_action/seqs/0001.npy"), np.zeros((60, 17, 2), np.float32))
    _w("pose_action/train.txt", ["seqs/0001.npy 0"])
    _w("pose_action/val.txt", ["seqs/0001.npy 0"])
    _w("pose_action/README.txt", _note(
        "pose_action(骨架行为)",
        "seqs/*.npy + train.txt / val.txt",
        "每行: 路径 行为类别号\n"
        "npy: (T, V, C) 或 (C,T,V);C 默认 2(x,y);V 默认 17(COCO)\n"
        "配置: transform.clip_len 统一时长,Backbone.num_joints,Head.num_classes"))


def video_cls():
    for s in ("train", "val"):
        for i in range(4):
            _img("video_cls/frames/%s/0001/%04d.jpg" % (s, i))
        _w("video_cls/%s.txt" % s, ["frames/%s/0001 0" % s])
    _w("video_cls/README.txt", _note(
        "video_cls(视频行为)",
        "视频文件(*.mp4)或帧目录 + train.txt / val.txt",
        "每行: 路径(视频或帧目录) 行为类别号\n"
        "采样: transform.num_segments × frames_per_seg 帧均匀分段;image_size/crop_size 控制分辨率"))


def yolo_pc_seg():
    """pc_seg: clouds/<split>/*.npy (N,4) + labels/<split>/*.npy (N,) + train/val.txt"""
    for s in ("train", "val"):
        pc = np.column_stack(
            [
                np.linspace(-8, 8, 256),
                np.zeros(256),
                np.zeros(256),
                np.ones(256),
            ]
        ).astype(np.float32)
        lab = np.zeros(256, dtype=np.int64)
        lab[128:] = 1
        try:
            import os as _os

            for sub, arr in (("clouds", pc), ("labels", lab)):
                p = os.path.join(OUT, "pc_seg", sub, s, "0001.npy")
                os.makedirs(_os.path.dirname(p), exist_ok=True)
                np.save(p, arr)
        except Exception:
            pass
        _w("pc_seg/%s.txt" % s, ["clouds/%s/0001.npy" % s])
    _w("pc_seg/README.txt", _note(
        "pc_seg(点云 pillar 分割)",
        "clouds/<split>/*.npy|.bin + labels/<split>/*.npy|.txt + train.txt/val.txt",
        "clouds: float32 N×3(x,y,z) 或 N×4(x,y,z,intensity);.bin 为 KITTI 风格\n"
        "labels: 每点 int 类别(与点一一对应);缺省视为全 0\n"
        "配置: Architecture.task: pc_seg, Head.num_classes=C\n"
        "Train.dataset: pc_range=[xmin,xmax,ymin,ymax,zmin,zmax], pillar_size=[dx,dy,dz]\n"
        "Loss: SemLoss(pillar CE); Metric main_indicator: mIoU\n"
        "注意: 图像预训练权重不适用,Global.pretrained_model 置 null"))


def yolo_det3d():
    """det3d: clouds + labels/*.txt (cls x y z l w h yaw) + train/val.txt"""
    for s in ("train", "val"):
        pc = np.column_stack(
            [
                np.linspace(-8, 8, 256),
                np.zeros(256),
                np.zeros(256),
                np.ones(256),
            ]
        ).astype(np.float32)
        p = os.path.join(OUT, "det3d", "clouds", s, "0001.npy")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, pc)
        _w(
            "det3d/labels/%s/0001.txt" % s,
            [
                "car 5.0 -3.0 0.5 4.0 1.8 1.6 1.57",
                "ped 2.0 4.0 0.8 0.6 0.6 1.7 0.10",
            ],
        )
        _w("det3d/%s.txt" % s, ["clouds/%s/0001.npy" % s])
    _w("det3d/README.txt", _note(
        "det3d(LiDAR 3D 检测, CenterPoint 简化)",
        "clouds/<split>/*.npy|.bin + labels/<split>/*.txt + train.txt/val.txt",
        "labels 每行一个框: cls x y z l w h yaw  (cls=类名或 int; LiDAR 系)\n"
        "可选第 9 列 difficulty(忽略)\n"
        "配置: Architecture.task: det3d, Head.num_classes=C, names: [...]\n"
        "Train.dataset: pc_range=[xmin,ymin,zmin,xmax,ymax,zmax], pillar_size=[dx,dy,dz]\n"
        "Loss: Det3DLoss(heatmap focal + L1); Metric: Det3DMetric(mAP BEV IoU)\n"
        "PostProcess: Det3DPostProcess(score_thres, nms_thres)\n"
        "注意: 图像预训练权重不适用,Global.pretrained_model 置 null"))


BUILDERS = {
    "ocr_det": ocr_det, "ocr_rec": ocr_rec, "det": yolo_det, "pose": yolo_pose,
    "segment": yolo_seg, "classify": yolo_cls, "semantic": yolo_sem, "depth": yolo_depth,
    "lane_seg": yolo_lane_seg, "lane_row": yolo_lane_row,
    "pc_seg": yolo_pc_seg,
    "det3d": yolo_det3d,
    "plate_rec": plate_rec, "attribute": attribute, "pose_action": pose_action,
    "video_cls": video_cls,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=None,
                    help="只生成指定任务(可重复);默认全部: %s" % ", ".join(BUILDERS))
    args = ap.parse_args()
    tasks = args.task or list(BUILDERS)
    print("生成到", OUT)
    for t in tasks:
        if t not in BUILDERS:
            print("  skip(未知任务):", t)
            continue
        BUILDERS[t]()
        print("  %-12s ok" % t)
    print("每个目录都有 README.txt(字段说明 + 配置片段),可直接当模板")


if __name__ == "__main__":
    main()
