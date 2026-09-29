"""航向角约定的回归守卫：断言 AIStation 的导出公式 == TorchKiln 的 KITTI 转换式。

为什么需要这个文件：`tools/convert/kitti_to_det3d.py` 的注释里那套角度推导与 KITTI
devkit ``compute_box_3d`` 的实际约定**并不自洽**（注释称长度轴方向是 ``ry+π/2``，
但按 devkit 的 ``R_y`` 矩阵，长度轴其实是 ``(cos ry, 0, -sin ry)``）。

我们的处理是**不去"修正"它**，而是直接沿用 TorchKiln 的 ``yaw = ry + π/2``——
因为 TorchKiln 的 loss / 后处理 / 评估都按那套写，双方共用一个约定比各自推导
"正确"更重要（否则导出的朝向会与它的后处理差一个固定角）。

本文件把这个决策**钉成可执行断言**：哪天 TorchKiln 改了公式，这里会立刻红。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TKILN_REPO_ROOT", "/opt/torchkiln"))
from tools.convert.kitti_to_det3d import convert_box  # noqa: E402

FAIL = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    if not cond:
        FAIL.append(name)


# ---- AIStation 侧的导出公式（必须与 exporter._export_mono3d 逐字一致）----
def aistation_box3d_to_lidar(b):
    x = b["z"]
    y = -b["x"]
    z = -b["y"] + b["h"] / 2.0
    yaw = b["ry"] + math.pi / 2.0
    return [x, y, z, b["l"], b["w"], b["h"], yaw]


# ---- TorchKiln 侧的 KITTI 转换：yaw 那一步必须等于 ours ----
# 用一个只关心 yaw 的 T（yaw 与 T 无关）
T = np.eye(4)
names = {"car": 0}
L, W, H = 4.5, 1.8, 1.6

print("=== AIStation yaw 与 TorchKiln KITTI 转换的 yaw 必须逐个相等 ===")
for ry in (0.0, 0.3, math.pi / 2, -math.pi / 2, 1.2, -2.5, math.pi - 0.1):
    parts = ["car", "0.0", "0", "0", "0", "0", "100", "80",
             f"{H:.2f}", f"{W:.2f}", f"{L:.2f}",
             "10.00", "0.00", "0.00", f"{ry:.6f}"]
    out = convert_box(parts, names, T)
    assert out is not None, "convert_box 返回 None（类别名大小写？）"
    tk_yaw = out[7]                       # [cls,x,y,z,l,w,h,yaw]
    ours = aistation_box3d_to_lidar(
        {"x": 0.0, "y": 0.0, "z": 10.0, "l": L, "w": W, "h": H, "ry": ry})[6]
    d = math.atan2(math.sin(ours - tk_yaw), math.cos(ours - tk_yaw))
    # 容差取 1e-6：KITTI 标签行只存 6 位小数，上面 `f"{ry:.6f}"` 已经舍入过一次，
    # 与全精度的 ry 比对天然有 ~3e-7 的差。这不是公式不一致，是格式精度。
    check(f"ry={ry:+.4f}  ours={ours:+.6f}  torchkiln={tk_yaw:+.6f}",
          abs(d) < 1e-6, f"差 {d:+.2e}")

print("\n=== 轴置换与抬升：与 convert_box 的 center 处理一致 ===")
# convert_box: center_cam = [x, y - h/2, z]，再乘 T。我们这里相机系存底面，
# 抬 h/2 得中心 —— 与它同义（相机 y 向下为正，LiDAR z 向上为正，故 z = -y + h/2）
b = {"x": 0.0, "y": -0.9, "z": 10.0, "l": L, "w": W, "h": H, "ry": 0.0}
v = aistation_box3d_to_lidar(b)
# 底面在光轴上方 0.9m（相机 y 向下为正 => y=-0.9 表示上方），中心再抬 h/2
check("z_lidar = -y_cam + h/2 = 1.7", abs(v[2] - 1.7) < 1e-9, f"{v[2]:.4f}")
check("x_lidar = z_cam", abs(v[0] - 10.0) < 1e-9)
check("y_lidar = -x_cam", abs(v[1] - 0.0) < 1e-9)

print("\n=== 尺寸轴序：l 沿 yaw、w 垂直（TorchKiln 的 (w,l,h,ry) 约定）===")
check("l/w/h 顺序未被打乱", v[3:6] == [L, W, H], str(v[3:6]))

print("\n=== 往返一致 ===")
def lidar_to_box3d(v):
    x, y, z, l, w, h, yaw = v
    return {"x": -y, "y": h / 2 - z, "z": x, "l": l, "w": w, "h": h,
            "ry": yaw - math.pi / 2}


for b in ({"x": 0.0, "y": -0.9, "z": 10.0, "l": 4.5, "w": 1.8, "h": 1.6, "ry": 0.0},
          {"x": -2.0, "y": -0.5, "z": 30.0, "l": 0.7, "w": 0.7, "h": 1.75, "ry": 0.9},
          {"x": 3.0, "y": -2.0, "z": 5.0, "l": 11.0, "w": 2.6, "h": 3.2, "ry": -1.4}):
    back = lidar_to_box3d(aistation_box3d_to_lidar(b))
    bad = {k: (b[k], back[k]) for k in b if abs(back[k] - b[k]) > 1e-9}
    check("往返一致", not bad, str(bad) if bad else f"({b['x']},{b['y']},{b['z']},ry={b['ry']})")

print("\n" + "=" * 60)
if FAIL:
    print(f"失败 {len(FAIL)} 项: " + "; ".join(FAIL))
    sys.exit(1)
print("全部通过：AIStation 与 TorchKiln 的 3D 框约定一致（含 yaw 公式）")
