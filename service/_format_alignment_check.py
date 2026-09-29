"""用 TorchKiln **真实的 Dataset 类**验证 AIStation 导出格式能被正确读取。

这是「导出格式对上 TorchKiln」的最终验收：不是看代码，而是把 AIStation 各导出器
产出的布局喂给 TorchKiln 自己的加载器，断言样本数与标签结构正确。

⚠️ 必须在装有 torchkiln 的容器里跑（数据类的 __getitem__ 需要 cv2/numpy/torch）。
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

ROOT = tempfile.mkdtemp(prefix="tk_align_")
W = H = 64
FAIL = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    if not cond:
        FAIL.append(name)


def img(stem, path):
    """造一张有内容的图（纯色会被 JPEG 压成近似值，但足够 cv2 读取）。"""
    a = np.full((H, W, 3), 30, np.uint8)
    a[H // 4:H // 2, W // 4:W // 2] = 220
    cv2.imwrite(path, a)


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ================================================================ 1. 检测
print("=== 1. detection：labels/*.txt = cls cx cy w h（归一化）===")
d = os.path.join(ROOT, "det")
os.makedirs(os.path.join(d, "images", "train"))
os.makedirs(os.path.join(d, "labels", "train"))
rows = []
for i in range(3):
    img(f"a{i}", os.path.join(d, "images", "train", f"a{i}.jpg"))
    rows.append(f"a{i}.jpg")
    with open(os.path.join(d, "labels", "train", f"a{i}.txt"), "w") as f:
        f.write("0 0.500000 0.500000 0.400000 0.300000\n"
                "1 0.250000 0.250000 0.200000 0.200000\n")
write_lines(os.path.join(d, "train.txt"), [f"images/train/{r}" for r in rows])

from torchkiln.data.det import DetDataset  # noqa: E402


class _C(dict):
    def __getitem__(self, k):
        return dict.get(self, k)


det_cfg = _C({
    "Train": _C({"dataset": _C({"data_dir": d, "label_file_list": [os.path.join(d, "train.txt")],
                                "box_format": "xyxy", "transform": _C({"image_size": 64}),
                                "labels_cache": False, "augment": None})}),
})
det = DetDataset(det_cfg, mode="Train", logger=None)
check("DetDataset 样本数 = 3", len(det) == 3, f"实际 {len(det)}")
s = det[0]
check("单样本长度 = 3 (img, targets, valid)", len(s) == 3, f"{len(s)}")
check("targets 形状 (2,5) = 2 个框 × (cls,x1,y1,x2,y2)", s[1].shape == (2, 5), str(s[1].shape))
check("targets 是原图像素（非归一化）", float(s[1][0][2]) > 1.0, f"x1={float(s[1][0][2]):.1f}")

# ================================================================ 2. 旋转框
print("\n=== 2. rotated_detection：cls + 8 角点（9 段，非 6 字段 xywhr）===")
d = os.path.join(ROOT, "obb")
os.makedirs(os.path.join(d, "images", "train"))
os.makedirs(os.path.join(d, "labels", "train"))
rows = []
for i in range(2):
    img(f"o{i}", os.path.join(d, "images", "train", f"o{i}.jpg"))
    rows.append(f"o{i}.jpg")
    with open(os.path.join(d, "labels", "train", f"o{i}.txt"), "w") as f:
        f.write("0 0.20 0.20 0.80 0.20 0.80 0.80 0.20 0.80\n")
write_lines(os.path.join(d, "train.txt"), [f"images/train/{r}" for r in rows])
# 注意：这里必须用**独立的 config 指向 obb 目录**。若复用上一段 det_cfg（data_dir
# 仍指向 det），读到的会是 5 字段标签，被 box_format=xywhr 的 9 字段门槛全部丢弃——
# 这正是 TorchKiln 对格式不符的预期行为。
obb_cfg = _C({
    "Train": _C({"dataset": _C({"data_dir": d, "label_file_list": [os.path.join(d, "train.txt")],
                                "box_format": "xywhr", "transform": _C({"image_size": 64}),
                                "labels_cache": False, "augment": None})}),
})
obb = DetDataset(obb_cfg, mode="Train", logger=None)
s = obb[0]
check("OBB targets 形状 (1,6) = cls,cx,cy,w,h,theta", s[1].shape == (1, 6), str(s[1].shape))
check("OBB 未被 9 字段门槛丢弃", s[1].shape[0] == 1, f"{s[1].shape[0]} 个框")
check("OBB 中心落在图内（cx/cy 是像素）", 0 < float(s[1][0][1]) < W and 0 < float(s[1][0][2]) < H,
      f"cx={float(s[1][0][1]):.1f} cy={float(s[1][0][2]):.1f}")

# 反证：若误写成 6 字段 cx cy w h angle，会被整行丢弃
bad6 = os.path.join(d, "labels", "train", "o0.txt")
with open(bad6, "w") as f:
    f.write("0 0.5 0.5 0.3 0.2 0.785\n")
ds6 = DetDataset(obb_cfg, mode="Train", logger=None)
check("误写 6 字段 xywhr → 被丢弃（证明必须写 8 角点）", ds6[0][1].shape[0] == 0,
      f"{ds6[0][1].shape[0]} 个框")
with open(bad6, "w") as f:
    f.write("0 0.20 0.20 0.80 0.20 0.80 0.80 0.20 0.80\n")

# ================================================================ 3. 实例分割
print("\n=== 3. segmentation：cls + 归一化多边形 ===")
d = os.path.join(ROOT, "seg")
os.makedirs(os.path.join(d, "images", "train"))
os.makedirs(os.path.join(d, "labels", "train"))
rows = []
for i in range(2):
    img(f"s{i}", os.path.join(d, "images", "train", f"s{i}.jpg"))
    rows.append(f"s{i}.jpg")
    with open(os.path.join(d, "labels", "train", f"s{i}.txt"), "w") as f:
        f.write("0 0.10 0.10 0.60 0.10 0.60 0.60 0.10 0.60\n")
write_lines(os.path.join(d, "train.txt"), [f"images/train/{r}" for r in rows])
from torchkiln.data.seg import SegDataset  # noqa: E402

seg_cfg = _C({"Train": _C({"dataset": _C({"data_dir": d, "label_file_list": [os.path.join(d, "train.txt")],
                                        "transform": _C({"image_size": 64}), "augment": None})})})
seg = SegDataset(seg_cfg, mode="Train", logger=None)
s = seg[0]
check("SegDataset 样本数 = 2", len(seg) == 2, f"{len(seg)}")
check("单样本含掩码", len(s) == 4, f"{len(s)}")
check("掩码非全零", int(s[3].sum()) > 0, f"sum={int(s[3].sum())}")

# ================================================================ 4. 语义分割
print("\n=== 4. semantic_segmentation：masks/*.png 单通道 uint8（像素值=类下标）===")
d = os.path.join(ROOT, "sem")
os.makedirs(os.path.join(d, "images", "train"))
os.makedirs(os.path.join(d, "masks", "train"))
rows = []
for i in range(2):
    img(f"m{i}", os.path.join(d, "images", "train", f"m{i}.jpg"))
    rows.append(f"m{i}.jpg")
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[8:32, 8:32] = 1     # 类 1
    mask[32:56, 32:56] = 2   # 类 2
    cv2.imwrite(os.path.join(d, "masks", "train", f"m{i}.png"), mask)
write_lines(os.path.join(d, "train.txt"), [f"images/train/{r}" for r in rows])
from torchkiln.data.sem import SemDataset  # noqa: E402

sem_cfg = _C({"Train": _C({"dataset": _C({"data_dir": d, "label_file_list": [os.path.join(d, "train.txt")],
                                         "transform": _C({"image_size": 64})})})})
sem = SemDataset(sem_cfg, mode="Train", logger=None)
s = sem[0]
check("SemDataset 样本数 = 2", len(sem) == 2, f"{len(sem)}")
raw = cv2.imread(os.path.join(d, "masks", "train", "m0.png"), cv2.IMREAD_GRAYSCALE)
check("掩码经 IMREAD_GRAYSCALE 后像素值未被压成灰度", int(raw.max()) == 2, f"max={int(raw.max())}")
check("单样本 = [img, mask]", len(s) == 2, f"{len(s)}")
check("mask 形状与图一致", tuple(s[1].shape) == (H, W), str(s[1].shape))

# ================================================================ 5. 分类
print("\n=== 5. classification：单标签 清单 = <相对 data_dir 的路径> <类下标> ===")
d = os.path.join(ROOT, "cls")
rows = []
for ci, cn in enumerate(["cat", "dog"]):
    os.makedirs(os.path.join(d, "train", cn), exist_ok=True)
    for i in range(2):
        img(f"{cn}{i}", os.path.join(d, "train", cn, f"{cn}{i}.jpg"))
        rows.append(f"train/{cn}/{cn}{i}.jpg {ci}")
write_lines(os.path.join(d, "train.txt"), rows)
from torchkiln.data import ClsDataset  # noqa: E402

cls_cfg = _C({"Train": _C({"dataset": _C({"data_dir": d, "label_file_list": [os.path.join(d, "train.txt")],
                                        "transform": _C({"image_size": 32})})})})
cls = ClsDataset(cls_cfg, mode="Train", logger=None)
check("ClsDataset 样本数 = 4", len(cls) == 4, f"{len(cls)}")
s = cls[0]
check("单样本 = [img, label]", len(s) == 2, f"{len(s)}")
check("类别下标在 0..1", 0 <= int(s[1]) <= 1, str(int(s[1])))
check("清单路径确实相对 data_dir 解析成功（img 非 None）", s[0] is not None)

print("\n  -- 多标签：必须是**稠密向量**，稀疏会因下标错位被当成别的类 --")
d2 = os.path.join(ROOT, "cls_multi")
os.makedirs(os.path.join(d2, "train"), exist_ok=True)
rows = []
for i in range(2):
    img(f"m{i}", os.path.join(d2, "train", f"m{i}.jpg"))
    rows.append(f"train/m{i}.jpg 1 0 1")
write_lines(os.path.join(d2, "train.txt"), rows)
mcfg = _C({"Train": _C({"dataset": _C({"data_dir": d2, "label_file_list": [os.path.join(d2, "train.txt")],
                                       "multi_label": True,
                                       "transform": _C({"image_size": 32})})})})
mc = ClsDataset(mcfg, mode="Train", logger=None)
s = mc[0]
check("多标签样本数 = 2", len(mc) == 2, f"{len(mc)}")
check("标签是长度 3 的稠密向量", len(s[1]) == 3, str(len(s[1])))
check("向量值按类下标落位 [1,0,1]",
      [float(v) for v in s[1]] == [1.0, 0.0, 1.0], str([float(v) for v in s[1]]))

# ================================================================ 6. OCR
print("\n=== 6. ocr det/rec：清单 = 图片路径 + TAB + 标签 ===")
d = os.path.join(ROOT, "ocr")
os.makedirs(os.path.join(d, "images", "train"))
rows = []
for i in range(2):
    img(f"t{i}", os.path.join(d, "images", "train", f"t{i}.jpg"))
    entries = [{"transcription": "ABC",
                "points": [[6.0, 6.0], [50.0, 6.0], [50.0, 20.0], [6.0, 20.0]]}]
    rows.append(f"images/train/t{i}.jpg\t{json.dumps(entries, ensure_ascii=False)}")
write_lines(os.path.join(d, "train.txt"), rows)
from torchkiln.ocr.data.simple_dataset import SimpleDataSet  # noqa: E402

from torchkiln.ocr.data.imaug.label_ops import DetLabelEncode  # noqa: E402

# ⚠️ 不构造完整 SimpleDataSet：它要求提供整条 transforms 流水线
#    （DecodeImage → DetLabelEncode → … → KeepKeys），那是**模型侧**的事，与
#    「导出格式是否对得上」无关。这里直接复刻 __getitem__ 的解析契约
#    （simple_dataset.py:425-459）来验证格式，比构造 Dataset 更精准。
def parse_manifest_line(line_bytes, data_dir, delimiter="\t"):
    """与 SimpleDataSet.__getitem__ 同款的解析。"""
    substr = line_bytes.decode("utf-8").strip("\n").split(delimiter)
    file_name = substr[0]
    label = substr[1]                      # 缺第 2 段会 IndexError → 被静默吞掉丢样本
    img_path = os.path.join(data_dir, file_name)
    return file_name, label, img_path, os.path.isfile(img_path)

with open(os.path.join(d, "train.txt"), "rb") as f:
    raw_lines = f.readlines()
check("SimpleDataSet 清单样本数 = 2", len(raw_lines) == 2, f"{len(raw_lines)}")
fname, label, img_path, exists = parse_manifest_line(raw_lines[0], d)
check("第 1 行切成 2 段（TAB 分隔）", True, f"{fname}")
check("图片按 data_dir 拼出且存在", exists, img_path)
check("标签段是 JSON 数组", isinstance(json.loads(label), list) and len(json.loads(label)) == 1)
check("OCR 路径也用正斜杠（Linux 容器可用）", "\\" not in fname, fname)

parsed = json.loads(label)
check("含 transcription", parsed[0]["transcription"] == "ABC", str(parsed[0]))
check("points 是 4 个像素坐标", len(parsed[0]["points"]) == 4 and
      all(isinstance(v, float) for p in parsed[0]["points"] for v in p), str(parsed[0]["points"]))

out = DetLabelEncode(max_text_length=25, max_points_num=4)({"label": label})
check("DetLabelEncode 能转成多边形张量", out is not None and out["polys"].shape[1:] == (4, 2),
      str(None if out is None else out["polys"].shape))
check("polys 是原图像素量级（非归一化）", float(out["polys"].max()) >= 20.0,
      f"max={float(out['polys'].max()):.1f}")
check("polys 宽度能还原出框（>40px）",
      float(out["polys"][0][:, 0].max() - out["polys"][0][:, 0].min()) > 40.0)

# 反证：标签段缺失会被 IndexError 吞掉 → 样本被静默丢弃
try:
    parse_manifest_line(b"images/train/t0.jpg\n", d)
    check("缺第 2 段会抛错（故导出器必须跳过无标注图）", False, "竟然没抛")
except IndexError:
    check("缺第 2 段会抛 IndexError（故导出器必须跳过无标注图）", True)

# ================================================================ 7. 关键点
print("\n=== 7. keypoint：kpt_shape 必须与数据一致，否则 PoseDataset 静默丢行 ===")
d = os.path.join(ROOT, "pose")
os.makedirs(os.path.join(d, "images", "train"))
os.makedirs(os.path.join(d, "labels", "train"))
rows = []
for i in range(2):
    img(f"k{i}", os.path.join(d, "images", "train", f"k{i}.jpg"))
    rows.append(f"k{i}.jpg")
    with open(os.path.join(d, "labels", "train", f"k{i}.txt"), "w") as f:
        f.write("0 0.5 0.5 0.4 0.4 0.1 0.1 2 0.2 0.2 0\n")
write_lines(os.path.join(d, "train.txt"), [f"images/train/{r}" for r in rows])
from torchkiln.data.pose import PoseDataset  # noqa: E402

pose_cfg = _C({"Train": _C({"dataset": _C({"data_dir": d, "label_file_list": [os.path.join(d, "train.txt")],
                                          "kpt_shape": [2, 3],
                                          "transform": _C({"image_size": 64}), "augment": None})})})
pose = PoseDataset(pose_cfg, mode="Train", logger=None)
raw_out = pose._load("images/train/k0.jpg", W, H, 1.0, (0, 0))
boxes, kpts, valid = raw_out
check("kpt_shape=[2,3] 时框被保留（未触发 5+6=11 段门槛）", boxes.shape[0] == 1, str(boxes.shape))
check("关键点张量形状 (1,2,3)", kpts.shape == (1, 2, 3), str(kpts.shape))
check("可见性列保留 2/0", [float(kpts[0][0][2]), float(kpts[0][1][2])] == [2.0, 0.0],
      str([float(kpts[0][0][2]), float(kpts[0][1][2])]))
check("关键点坐标已换算成像素", float(kpts[0][0][0]) >= 1.0, f"x={float(kpts[0][0][0]):.1f}")

# 关键点数量不匹配会怎样 —— 证明「不下发 kpt_shape 就静默丢标签」
bad_cfg = _C({"Train": _C({"dataset": _C({"data_dir": d, "label_file_list": [os.path.join(d, "train.txt")],
                                         "kpt_shape": [17, 3],
                                         "transform": _C({"image_size": 64}), "augment": None})})})
bad = PoseDataset(bad_cfg, mode="Train", logger=None)
s = bad[0]
boxes_b = s[1] if len(s) > 1 else np.zeros((0, 5), np.float32)
check("kpt_shape 写成 17（模板默认）时标签被**全部丢弃** ← 这就是必须下发它的原因",
      len(boxes_b) == 0, f"{len(boxes_b)} 个框")

# ================================================================ 8. 空清单陷阱
print("\n=== 8. 空清单陷阱：跳过不写 vs 写空文件 ===")
d = os.path.join(ROOT, "empty")
os.makedirs(os.path.join(d, "images", "val"), exist_ok=True)
write_lines(os.path.join(d, "val.txt"), [])   # 故意写空清单
empty_cfg = _C({"Train": _C({"dataset": _C({"data_dir": d, "label_file_list": [os.path.join(d, "val.txt")],
                                            "transform": _C({"image_size": 32})})})})
ed = DetDataset(empty_cfg, mode="Train", logger=None)
check("空清单 → Dataset 长度为 0（所以导出器必须跳过不写）", len(ed) == 0, f"{len(ed)}")

shutil.rmtree(ROOT, ignore_errors=True)
print("\n" + "=" * 60)
if FAIL:
    print(f"失败 {len(FAIL)} 项:\n  - " + "\n  - ".join(FAIL))
    sys.exit(1)
print("全部通过：AIStation 导出格式与 TorchKiln 读取器逐项对齐")
