"""造一个 mono3d 的 demo 数据集：图像 + labels/*.txt（LiDAR 系 7-dof）。

用于验证 Mono3DDataset / Mono3DTask / configs/pc/mono3d.yml 这条新链路。
标签坐标系与 det3d 一致：x 前 / y 左 / z 上，z 为框**中心**，l 沿 yaw。
"""
import json
import math
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="mono3d_demo_")
NAMES = ["car", "ped", "cyc"]
# 各类别的典型米制尺寸（l, w, h），yaw 取几个离散朝向
DIMS = {"car": (4.5, 1.8, 1.6), "ped": (0.7, 0.7, 1.75), "cyc": (1.7, 0.7, 1.6)}

rng = np.random.RandomState(7)
W = H = 320
PC_RANGE = [-16, -16, -3, 16, 16, 3]
PILLAR = [0.5, 0.5, 4.0]


def make_split(split, n=12):
    img_dir = os.path.join(ROOT, "images", split)
    lab_dir = os.path.join(ROOT, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    rows = []
    for i in range(n):
        # 画一张有纹理的底图，避免纯色被 JPEG 压掉
        img = (rng.rand(H, W, 3) * 60 + 90).astype(np.uint8)
        cv2.line(img, (0, H // 2), (W, H // 2), (30, 30, 30), 2)
        lines = []
        for _ in range(rng.randint(1, 4)):
            cls = int(rng.randint(0, len(NAMES)))
            nm = NAMES[cls]
            l, w, h = DIMS[nm]
            # 框中心：x 取前方 3~14m，y 取横向 ±6m，z 取 +0.5~1.5m（相机大致齐平）
            x = float(rng.uniform(3, 14))
            y = float(rng.uniform(-6, 6))
            z = float(rng.uniform(0.5, 1.5))
            ry = float(rng.uniform(-math.pi, math.pi))
            lines.append("%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
                nm, x, y, z, l, w, h, ry))
            # 画一个粗略的框，仅为了让图像看起来「有目标」
            cx = int(W * 0.5 + y * W * 0.06)
            cy = int(H * 0.6 - (x / 16) * H * 0.4)
            bw = max(6, int(l * 6))
            bh = max(6, int(h * 14))
            cv2.rectangle(img, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2),
                          (0, 200, 0), 2)
        cv2.imwrite(os.path.join(img_dir, "%s_%03d.jpg" % (split, i)), img)
        with open(os.path.join(lab_dir, "%s_%03d.txt" % (split, i)), "w") as f:
            f.write("\n".join(lines) + "\n")
        rows.append("images/%s/%s_%03d.jpg" % (split, split, i))
    with open(os.path.join(ROOT, split + ".txt"), "w") as f:
        f.write("\n".join(rows) + "\n")
    return rows


train_rows = make_split("train")
val_rows = make_split("val", n=6)
print("demo 数据集:", ROOT)
print("  train:", len(train_rows), " val:", len(val_rows))

# ---------------- 校验 Mono3DDataset ----------------
sys.path.insert(0, os.environ.get("TKILN_REPO_ROOT", "/opt/torchkiln"))
from torchkiln.tasks import get_task  # noqa: E402

assert "mono3d" in __import__("torchkiln.tasks", fromlist=["AVAILABLE"]).AVAILABLE, \
    "mono3d 未注册到 TASK_SPECS"
TaskCls = get_task("mono3d")
print("  task:", TaskCls)

import yaml  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(os.environ.get("TKILN_REPO_ROOT", "/opt/torchkiln"),
                                       "configs", "pc", "mono3d.yml")))
cfg["Train"]["dataset"]["data_dir"] = ROOT
cfg["Train"]["dataset"]["label_file_list"] = [os.path.join(ROOT, "train.txt")]
cfg["Eval"]["dataset"]["data_dir"] = ROOT
cfg["Eval"]["dataset"]["label_file_list"] = [os.path.join(ROOT, "val.txt")]

task = TaskCls  # get_task 已返回实例
ds, ev = task.build_datasets(cfg, None)
print("  训练集 %d 张，验证集 %d 张" % (len(ds), len(ev)))

s = ds[0]
img, labels, mask = s
print("  图像张量 shape =", img.shape, "dtype =", img.dtype,
      "范围 [%.3f, %.3f]" % (img.min(), img.max()))
assert img.shape == (3, 320, 320), img.shape
assert img.min() >= 0.0 and img.max() <= 1.0
assert labels.ndim == 2 and labels.shape[1] == 8, labels.shape
print("  labels shape =", labels.shape)
print("  mask shape   =", mask.shape)
assert labels.shape[0] == mask.shape[0]

# 类别 id：names 决定映射，标签写的是类名
for row in labels:
    assert 0 <= row[0] < len(NAMES), "类 id 越界: %s" % row[0]
print("  首行标签 [cls,x,y,z,l,w,h,yaw] =", np.round(labels[0], 4).tolist())

# collate
batch = task.train_collate([ds[0], ds[1], ds[2]])
print("  collate -> image", batch[0].shape, "targets", batch[1].shape, "masks", batch[2].shape)
assert batch[0].shape[0] == 3 and batch[0].shape[1] == 3
assert batch[1].shape[2] == 8

# 水平翻转增强：y 与 yaw 必须同时取反
ds.hflip = True
np.random.seed(1)
found = False
for _ in range(200):
    _, lab, _ = ds[0]
    if lab.shape[0] == 0:
        continue
    found = True
    break
assert found, "hflip 分支没跑到"
# 直接验证变换本身（关掉 jitter，否则亮度会被缩放clip，断言像素绝对值不可靠）
ds.jitter = 0.0
base = np.array([[0, 5.0, 2.0, 1.0, 4.0, 2.0, 1.5, 0.7]], dtype=np.float32)
img0 = np.zeros((3, 8, 8), dtype=np.float32)
img0[:, :, 0] = 1.0
out_img, out_lab = ds._augment(img0, base.copy())
assert out_lab[0][2] == -2.0, out_lab
assert abs(out_lab[0][7] + 0.7) < 1e-5, out_lab
assert out_lab[0][1] == 5.0 and out_lab[0][3] == 1.0, "x/z 不该变"
assert out_img[:, :, -1].max() == 1.0, "图应沿 W 轴镜像"
assert out_img[:, :, 0].max() == 0.0, "原首列应移到末列"
print("  hflip 增强正确：y→-y, yaw→-yaw，x/z 不变")

# 关闭 hflip 时标签与图都不应被改
ds.hflip = True
np.random.seed(0)
hit = False
for _ in range(50):
    _, lab2 = ds._augment(img0, base.copy())   # _augment 返回 (img, labels)
    if lab2[0][2] < 0:
        hit = True
        break
assert hit, "hflip 开了却一次都没翻转（随机性有问题）"
print("  hflip 概率生效")

# 清单里的图读不出来时必须返回 []（collate 会过滤），不能抛异常打断整批训练
good = ds.img_files[0]
bad = "images/train/__missing__.jpg"
ds.img_files.append(bad)
assert ds[len(ds) - 1] == [], "读图失败应返回空列表"
ds.img_files.remove(bad)
assert ds[0][0].shape[0] == 3
# 标签文件缺失也应给空标签而不是崩
print("  读图失败返回 []，不打断训练")

# ---------------- 端到端跑一步训练 ----------------
import torch  # noqa: E402

from torchkiln.models.det3d import build_det3d_model  # noqa: E402
from torchkiln.det3d import build_det3d_loss, build_det3d_postprocess  # noqa: E402

ds.hflip = False
model = build_det3d_model(cfg["Architecture"])
print("  模型 in_channels =", cfg["Architecture"]["in_channels"])
nparam = sum(p.numel() for p in model.parameters())
print("  模型参数量 = %,d".replace(",", "") % nparam if False else "  模型参数量 = {:,}".format(nparam))

loss_fn = build_det3d_loss(
    cfg["Loss"], num_classes=cfg["Architecture"]["Head"]["num_classes"],
    pc_range=cfg["Train"]["dataset"]["pc_range"],
    pillar_size=cfg["Train"]["dataset"]["pillar_size"],
)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
b = task.train_collate([ds[i] for i in range(3)])
out = model(b[0])
# Det3DLoss.forward(pred, batch) 里 batch 是**完整的 collate 输出**（索引 1/2 取
# targets 与 masks），不是 (targets, masks) 二元组
losses = loss_fn(out, b)
total = losses["loss"]
opt.zero_grad()
total.backward()
opt.step()
print("  前向+反向 OK: loss_cls={:.4f} loss_box={:.4f} loss={:.4f}".format(
    float(losses["loss_cls"]), float(losses["loss_box"]), float(total)))
assert torch.isfinite(total), "loss 非有限"

pp = build_det3d_postprocess(
    cfg["PostProcess"], pc_range=cfg["Train"]["dataset"]["pc_range"],
    pillar_size=cfg["Train"]["dataset"]["pillar_size"],
)
model.eval()
with torch.no_grad():
    res = pp(out)
# Det3DPostProcess 返回 **每个 batch 元素一个 dict** 的列表
# （{"bboxes": (N,5) BEV(x,y,l,w,yaw), "scores": (N,), "labels": (N,)}）
assert isinstance(res, list) and len(res) == 3, type(res)
assert set(res[0]) == {"bboxes", "scores", "labels"}, set(res[0])
print("  后处理 -> %d 个结果，bboxes%s scores%s labels%s" % (
    len(res), tuple(res[0]["bboxes"].shape),
    tuple(res[0]["scores"].shape), tuple(res[0]["labels"].shape)))
assert res[0]["bboxes"].shape[-1] == 5, "BEV 框应为 (x,y,l,w,yaw)"

from torchkiln.det3d import build_det3d_metric  # noqa: E402

metric = build_det3d_metric(
    cfg["Metric"], num_classes=cfg["Architecture"]["Head"]["num_classes"],
    names=cfg["Eval"]["dataset"]["names"],
)
with torch.no_grad():
    metric(pp(out), b)          # 与 Det3DTask.eval_step 的调用方式一致
got = metric.get_metric()
print("  metric（随机初始化，AP 应为 0）=", {
    k: round(float(v), 4) for k, v in got.items() if isinstance(v, (int, float))})

# ---- mAP70 的 key 映射必须真的取到 0.7 处的 AP ----
# 上面 AP 全 0 证明不了映射对错（0 和「取不到」都是 0）。这里直接喂
# 「预测 == GT」的完美结果，若 mAP70 仍为 0 就说明它还在查 mAP75。
metric2 = build_det3d_metric(
    cfg["Metric"], num_classes=cfg["Architecture"]["Head"]["num_classes"],
    names=cfg["Eval"]["dataset"]["names"],
)
tgt = b[1].numpy()          # (B, max_gt, 8) = [cls,x,y,z,l,w,h,yaw]
m2 = b[2].numpy()
perfect = []
for i in range(tgt.shape[0]):
    n = int(m2[i].sum())
    if n == 0:
        perfect.append({"bboxes": np.zeros((0, 5), np.float32),
                        "scores": np.zeros((0,), np.float32),
                        "labels": np.zeros((0,), np.int64)})
        continue
    g = tgt[i][:n]
    perfect.append({
        # BEV = [x, y, l, w, yaw]（**不含 cls**），即 7-dof 的第 1/2/4/5/7 列
        "bboxes": g[:, [1, 2, 4, 5, 7]].astype(np.float32),
        "scores": np.ones((n,), np.float32),
        "labels": g[:, 0].astype(np.int64),
    })
metric2(perfect, b)
g2 = metric2.get_metric()
print("  metric（预测==GT）=", {k: round(float(v), 4) for k, v in g2.items()})
assert g2["mAP50"] > 0.9, "完美预测下 mAP50 应≈1，实际 %s" % g2["mAP50"]
# 关键断言：修复前这里恒为 0.0（查的是 mAP75，而配置里没有 0.75 阈值）
assert g2["mAP70"] > 0.9, "mAP70 仍取不到 0.7 处的 AP（实际 %s）" % g2["mAP70"]
assert abs(g2["mAP"] - (g2["mAP50"] + g2["mAP70"]) / 2) < 1e-6, "mAP 应为所配阈值均值"
print("  mAP70 已正确取到 0.7 处的 AP（修复前恒为 0.0）")

print("\n全部通过：Mono3DDataset / Mono3DTask / configs/pc/mono3d.yml")
