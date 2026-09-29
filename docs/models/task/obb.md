# 旋转框检测（OBB / OBBU / OBB26）

> **定位**：在检测头之上加一条 **角度塔（`cv4`）**，回归 `(cx, cy, w, h, theta)` 旋转框；
> 用 **ProbIoU** 替代 IoU 做匹配与回归，用 **probioU + 上三角抑制** 做旋转 NMS。
> **任务**：`obb` · **头**：`OBB`（旧式 Conv，DFL）/ `OBBU`（新 DWConv）/ `OBB26`（别名）
> **权重**：`yolo11{n,s,m,l,x}-obb.pth`、`yolov8*-obb.pth`、`yolo26*-obb.pth`

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Ultralytics 的 OBB 头**无独立论文**。核心量 **ProbIoU** 来自 *Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection*（arXiv:2106.06072，见 `det/rbox.py` 源码注释） |
| 机构 | Ultralytics；OBB 数据集常用 **DOTA**（Xia et al., CVPR 2018） |
| 官方代码 | https://github.com/ultralytics/ultralytics |
| 官方权重 | `yolo11n-obb.pt` / `yolov8n-obb.pt` / `yolo26n-obb.pt`（DOTA 15 类） |
| 本框架实现 | 头：`torchkiln/nn/modules.py::OBB`(`OBBU`)<br>几何：`torchkiln/det/rbox.py`（`poly2rbox`/`dist2rbox`/`probiou`/`nms_rotated`/`rbox2dist`）<br>损失：`torchkiln/det/loss.py::ObbLoss`<br>任务：`torchkiln/tasks/obb.py` |
| 移植方式 | 权重逐键对齐 + 单步 loss/梯度对齐 + 旋转 NMS 输出与 ultra `TorchNMS.fast_nms` **完全一致** |

### 要解决的问题

- 遥感/文字/仓储等场景的物体**不是轴对齐**，水平框 IoU 对旋转目标严重失效。
- 直接对旋转多边形算 IoU 慢且不可导。**ProbIoU** 把每个 `xywhr` 框建模为
  **二维高斯**（均匀分布方差 `w²/12`、`h²/12`、协方差由 `theta` 决定），
  用 Bhattacharyya 距离给出**闭式、可导**的相似度，可同时用于匹配、损失、NMS。
- Ultralytics 的工程要点：角度用 `n` 码迭代平滑表示、angle loss 用 `sin(2Δθ)²` 并按宽高比加权。

---

## 2. 网络结构

### 2.1 整体框图（yolo11n-obb）

```
Input → backbone（Conv/C3k2/SPPF/C2PSA）→ neck（PAN）→ P3,P4,P5
  ▼ Head（nl=3）
  ┌───────────────────────────────────────────────────────┐
  │  cv2 回归塔 → 4*reg_max                               │
  │  cv3 分类塔 → nc                                       │
  │  cv4 角度塔 → ne(=1)   ← ★ OBB 独有                   │
  │      c4 = max(ch[0]//4, ne)                            │
  └───────────────────────────────────────────────────────┘
  每尺度输出 = cat([cv2, cv3, cv4], 1)（upstream 布局 [reg, cls, angle]）
  解码：dist2rbox([l,t,r,b, angle]) → xywhr
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | ultra 名字 |
|---|---|---|---|---|
| `cv2` | `Conv(k3)→Conv(k3)→Conv2d(1)` | →`4*reg_max` | `l,t,r,b` 距离 | `cv2` |
| `cv3` | 旧 `Conv×3` / 新 `DWConv+Conv`×2 | →`nc` | 分类 | `cv3` |
| **`cv4`** | `Conv(x,c4,3)→Conv(c4,c4,3)→Conv2d(c4,ne,1)`，`c4=max(ch[0]//4, ne)`，bias 初始化为 0 | →`ne=1` | ★ 角度 | `cv4` |
| `OBBU` | 新 DWConv 头，layout `[reg, cls, angle]`，支持 `end2end`（`one2one_cv2/cv3/cv4`） | | v26 头 | `OBB26` |

### 2.3 与 detect 头的差异

| 项 | Detect | **OBB** | 效果 |
|---|---|---|---|
| 输出通道 | `4*reg_max+nc` | **`4*reg_max+nc+ne`** | 多 1 个角度 |
| 多出的塔 | 无 | **`cv4`（角度）** | 270 度连续表示 |
| 匹配 IoU | CIoU / IoU | **ProbIoU** | 旋转不变 |
| 回归损失 | CIoU | **`1 - ProbIoU`** | |
| 额外损失 | 无 | **angle loss（`sin(2Δθ)²·γ`）** | |
| NMS | 水平 NMS | **旋转 NMS（ProbIoU）** | |

---

## 3. 输出与后处理（★ 重点）

### 3.1 角度解码（两种约定）

```
# v8 / v11：角度通道是 logit
theta = (sigmoid(angle_logit) - 0.25) * pi          # 范围 (-pi/4, 3pi/4)

# v26：角度通道是 raw
theta = angle_raw                                     # raw_angle=True
```

```
# dist2rbox：把 (l,t,r,b,theta) 解码为网格单位 xywhr
xf, yf = (rb - lt)/2
x = xf*cos(theta) - yf*sin(theta) + anchor_x          # 中心偏移按角度旋转
y = xf*sin(theta) + yf*cos(theta) + anchor_y
w, h = lt + rb                                         # ★ w=l+r, h=t+b
out = (x, y, w, h, theta)   →  × stride_tensor 得像素
```

### 3.2 ObbLoss（`det/loss.py`）

```
# 匹配用 probiou + points_in_rboxes（旋转内点判定）
iou  = probiou(pred_rbox, gt_rbox, floor=0.01)
box  = ((1 - iou) * weight).sum() / scores_sum            # probiou 回归

# DFL（reg_max>1）或 L1（reg_max==1）
target_ltrb = rbox2dist(gt_xywh, anchor, gt_angle, reg_max=reg_max-1)  # 只 clamp ltrb 距离

# angle loss：宽高比加权，对 pi 周期取最短差
log_ar = log((w+eps)/(h+eps)); scale_weight = exp(-(log_ar)^2/3^2)
delta  = pred_theta - gt_theta; delta = delta - round(delta/pi)*pi
angle  = (sin(2*delta)^2 * scale_weight * weight).sum() / scores_sum

loss = box_gain*box + cls_gain*cls + dfl_gain*dfl + angle_gain*angle
#      (7.5)          (0.5)           (1.5)           (1.0)
```

> 默认超参对齐 ultra OBB hyp：`dfl_gain=1.5`、`angle_gain=1.0`；`reg_max` 从 1 改为 **16（DFL）**。

### 3.3 旋转 NMS（`det/rbox.py::nms_rotated`）

```
ious = probiou(boxes.unsqueeze(0), boxes.unsqueeze(0))[0].squeeze_(-1)  # (N,N)，方差 w^2/12
ious.triu_(diagonal=1)                       # 上三角
pick = nonzero((ious >= iou_thres).sum(0) <= 0)   # 未被任何更高分框抑制
```

| 项 | 值 | 说明 |
|---|---|---|
| 实现 | ProbIoU 上三角矩阵（GPU 向量化） | 原 O(N²) 多边形裁剪：2000 框 **87s** → 现 **0.13s** |
| `max_candidates` | 3000 | 仿 ultra `max_nms`，防密集图 OOM |
| `iou_thres` | 0.7（评估） | |
| `conf_thres` | 0.001（评估） | |
| **一致性** | 与 ultra `TorchNMS.fast_nms(iou_func=batch_probiou)` **输出索引 maxdiff=0.0** | |

> ⚠️ **ProbIoU 方差必须用 `w²/12`**（均匀分布），不是 `(w/2)²`；且返回值已去尾维
> `(N,)`/`(B,A,M)`，调用方不要再 `squeeze`。

---

## 4. 配置与用法

### 4.1 最小配置（yolo11-obb）

```yaml
Architecture:
  task: obb
  yaml_file: torchkiln/cfg/models/11/yolo11-obb.yaml
  scale: n
  Head: {num_classes: 15, reg_max: 16}
Loss:  {name: ObbLoss, topk: 10, alpha: 0.5, dfl_gain: 1.5, angle_gain: 1.0}
Metric:       {name: DetMetric, box_format: xywhr, main_indicator: mAP50-95}
PostProcess:  {name: DetPostProcess, box_type: xywhr, reg_layout: upstream, iou_thres: 0.7}
Train:
  dataset: {name: DetDataset, data_dir: datasets/obb_demo, box_format: xywhr, ...}
```

**v26 额外需要**（raw angle + end2end）：
```bash
-o Loss.raw_angle=true -o Loss.use_one2one=true \
-o PostProcess.end2end=true -o PostProcess.angle_raw=true \
-o Architecture.Head.reg_max=1 -o Architecture.Head.end2end=true
```

### 4.2 四条链路

```bash
tkiln train   -c configs/yolo/yolo11-obb.yml -o Global.pretrained_model=yolo11n-obb
tkiln val     -c configs/yolo/yolo11-obb.yml --weights output/yolo11-obb/best_accuracy.pth
tkiln predict -c configs/yolo/yolo11-obb.yml --weights ... --input imgs/
tkiln export  -c configs/yolo/yolo11-obb.yml --weights ... --onnx
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ | 权重体积 |
|---|---|---|
| yolo11n-obb | 2.70 | ~5.5 MB |
| yolo11s-obb | 9.74 | ~19 MB |
| yolo11m-obb | 20.96 | ~42 MB |
| yolo11l-obb | 26.22 | ~52 MB |
| yolo11x-obb | 58.88 | ~118 MB |
| yolo26n-obb | 2.72 | reg_max=1 + end2end |
| yolov8n-obb | 3.10 | 旧家族 Conv 头 |

¹ 取自 `torchkiln/cfg/models/**/*-obb.yaml` 注释（v11n=2,695,747；v26n=2,715,614）。
**FLOPs / 本机耗时未在本框架侧统计**。

---

## 6. 公开指标

### 6.1 权重加载

| 家族 | missing | unexpected |
|---|---|---|
| yolo11-obb n/s/m/l/x | **0** | **0** |
| yolov8-obb n/s/m/l/x | **0** | **0** |
| yolo26-obb n/s/m/l/x | **0** | **0**（`reg_max=1` + end2end 双头） |

### 6.2 同权重推理（DOTA128 val）

| 权重 | 本框架 mAP50-95 | ultralytics | 差异 |
|---|---|---|---|
| yolo11n-obb（预训练直接推理） | **0.8005**（mAP50 0.950） | 0.821（mAP50 0.963） | **0.021** |
| yolov8n-obb | **0.790** | 0.8021 | ~0.012 |
| yolo26n-obb | **0.806** | 0.828 | ~0.022 |

⇒ 差 ≤0.022，属 **NMS/probiou 算子级微差**（与 seg/pose 同量级）。

### 6.3 关键修复：`poly2rbox` 用 `cv2.minAreaRect`

GT 与预测的 `theta` 约定必须一致。改用 **ultralytics 同款 `cv2.minAreaRect`**
（`w` 为长边、`theta` 规范化到 `[-pi/4, 3pi/4)`），替换旧的 arctan2 + `(-pi/2, pi/2]` 约定：
**mAP50-95 从 0.754 提升到 0.8005**。

### 6.4 微调（dota128）

- 框架微调（**lr0=0.001 温和**）**mAP50-95 = 0.809**（超预训练 0.80）。
- **公平对比**：框架用默认 lr0=0.01（与 ultra 一致）时**同样过拟合退化**（v26n best=0.814@ep1，
  随后 0.657/0.691/0.680）—— 与 ultra 行为方向一致；**不存在"框架显著更强"**。
- 用框架评估 ultra 微调后的 `best.pt` 得 **0.317**，与 ultra 自报 0.32 一致 → **评估无差异**。

### 6.5 单步 loss 对齐

OBB 的单步 loss/梯度与 ultra 对齐（同权重同输入同 GT）；旋转 NMS 输出索引 **maxdiff=0.0**。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 遥感（DOTA）/ 航拍 | **yolo11n-obb / yolo26n-obb** | 权重齐全、probiou 对齐 |
| 文字/票据方向框 | **n/s** | 角度塔只多 1 通道 |
| NMS-free 低延迟 | **yolo26n-obb（end2end + raw angle）** | one2one 头 |
| 高精度离线 | **11x-obb** | |
| 只要水平框 | detect | 更省、更快 |
| 密集旋转目标 | 注意 `max_candidates=3000` 上限 | |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs detect** | 多角度塔与 ProbIoU；水平 NMS → 旋转 NMS |
| **vs 多边形 IoU** | ProbIoU 闭式可导、快（0.13s vs 87s）；多边形裁剪仅用于可视化/精度校验 |
| **vs CSL / 环形平滑标签** | 本框架/ultra 用「sigmoid/raw 角度 + sin(2Δθ)² 损失」，无需 180 维分类 |
| **v8/v11 vs v26** | v8/v11：`(sigmoid(angle)-0.25)*pi`；v26：raw angle + end2end |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`poly2rbox` 必须用 `cv2.minAreaRect`**（长边为 `w`，θ∈[-π/4,3π/4)）；用错误的
  arctan2 约定会使 mAP 掉到 0.754。
- ⚠️ **ProbIoU 方差 = `w²/12`**（非 `(w/2)²`）；返回值已去尾维，勿重复 squeeze。
- ⚠️ **只 clamp `bbox2dist`/`rbox2dist` 输出的 ltrb 距离**（`reg_max-1-0.01`），
  **不要 clamp GT 框坐标**（该 bug 影响所有 detect/OBB 训练）。
- ⚠️ **`OBBU` 的 `cv4` bias 初始化为 0**；`cv3` 的分类部分 bias 才用 `-log((1-0.01)/0.01)`。
- ⚠️ **v26 必须同时设 raw angle + end2end**（见 §4.1），否则角度/分支口径错。
- ⚠️ **`max_candidates=3000`**：密集图超过此数会被截断（防 OOM）。
- ⚠️ **评估 fp32**（`_evaluate_loop` 强制 `autocast(enabled=False)`）。
- ⚠️ 对比必须**同权重 + 同一评估器**；ultra 自报 `rect=True` 会抬高数值。
