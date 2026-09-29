# 关键点检测（Pose / PoseU / Pose26）

> **定位**：在检测头之上加一条 **关键点塔（`cv4`）**，每个目标同时回归 `nk` 个关键点的
> `(x, y [, visibility])`，用 **OKS（Object Keypoint Similarity）** 作为评价与损失核心。
> **任务**：`pose` · **头**：`Pose`（旧式 Conv，DFL）/ `PoseU`（新 DWConv）
> **权重**：`yolo11{n,s,m,l,x}-pose.pth`、`yolov8*-pose.pth`、`yolo26*-pose.pth`

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Ultralytics 的 pose 头**无独立论文**；核心评价量 **OKS** 来自 COCO（Lin et al., *Microsoft COCO*, ECCV 2014）。相关工作：YOLO-Pose（同名的第三方工作，arXiv 号未核对） |
| 机构 | Ultralytics |
| 官方代码 | https://github.com/ultralytics/ultralytics |
| 官方权重 | `yolo11n-pose.pt` / `yolov8n-pose.pt` / `yolo26n-pose.pt` |
| 本框架实现 | 头：`torchkiln/nn/modules.py::Pose`(`PoseU`)<br>结构：`torchkiln/cfg/models/{8,11,26}/*-pose.yaml`<br>损失/指标：`torchkiln/pose.py`<br>数据：`torchkiln/data/pose.py` · 任务：`torchkiln/tasks/pose.py` |
| 移植方式 | 权重逐键对齐（`missing=0/unexpected=0`）+ 单步 loss/梯度对齐 + OKS 指标复刻 ultra `PoseValidator` |

### 要解决的问题

- 检测只给框，**人体/手脸/车牌角点**需要定位结构关键点。
- 关键点误差的度量必须**按关键点类型与目标尺度归一**：手腕可以差得多、鼻子要很准；
  大目标允许的像素误差大于小目标 —— 这就是 **OKS** 的作用，也是直接 L1/L2 的替代。
- Ultralytics 的工程要点：**每个目标每个关键点回归 `2` 个偏移 + `1` 个可见性 logit**，
  偏移以「网格单位」解码，配合 OKS 式损失。

---

## 2. 网络结构

### 2.1 整体框图（yolo11n-pose）

```
Input (B,3,H,W)
  │ backbone（Conv/C3k2/SPPF/C2PSA）→ P3(8),P4(16),P5(32)
  │ neck（PAN）→ T1,T2,T3
  ▼ Head（nl=3 尺度）
  ┌─────────────────────────────────────────────────────┐
  │  cv2 回归塔 → 4*reg_max   (v8/v11: DFL reg_max=16)   │
  │  cv3 分类塔 → nc          (旧 Conv×3；新 DWConv)      │
  │  cv4 关键点塔 → nk*ndim   ← ★ pose 独有              │
  │     c4 = max(ch[0]//4, nk)；ndim=2 或 3(含可见性)     │
  └─────────────────────────────────────────────────────┘
  每尺度输出 = cat([cv2, cv3, cv4], 1)  →  (B, 4*reg_max+nc+nk*ndim, H, W)
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | ultra 名字 |
|---|---|---|---|---|
| `cv2` | `Conv(k3)→Conv(k3)→Conv2d(1)` | →`4*reg_max` | 框回归 | `cv2` |
| `cv3` | 旧 `Conv×3` / 新 `DWConv+Conv`×2 | →`nc` | 分类 | `cv3` |
| **`cv4`** | `Conv(x,c4,3)→Conv(c4,c4,3)→Conv2d(c4,nk*ndim,1)`，`c4=max(ch[0]//4, nk)` | →`nk*ndim` | ★ 关键点偏移+可见性 | `cv4` |
| `Pose26`/`PoseU` | 新 DWConv 头，`reg_max=1`，layout `[reg, cls, kpt]` | 同上 | v26 关键点头 | `Pose26` |

> ⚠️ **`cv4` 的通道口径（本轮关键修复）**：`c4 = max(ch[0]//4, nk)`，其中
> `nk = kpt_shape[0] * kpt_shape[1]`（如 `[12,2]→24`、`[17,3]→51`）。
> 此前框架写成 `c4=x`（即 c1），与 ultra 不符，导致 yolo11n-pose 权重加载不匹配。
> 改后 **missing=0 / unexpected=0**。

### 2.3 与 detect 头的差异

| 项 | Detect | **Pose** | 效果 |
|---|---|---|---|
| 输出通道 | `4*reg_max+nc` | **`4*reg_max+nc+nk*ndim`** | 多出关键点 |
| 多出的塔 | 无 | **`cv4`** | 每点 2 偏移 (+1 可见性) |
| 损失 | box+cls+dfl | **+ keypoint(OKS) + visibility BCE** | |
| 指标 | box mAP | **关键点 mAP（OKS 匹配）** | 度量完全不同 |

---

## 3. 输出与后处理（★ 重点）

### 3.1 关键点解码（训练 vs 推理，单位不同！）

```
# ---- 推理（PosePostProcess，像素单位）----
raw = kpt_channel.view(B, -1, nk, ndim)
xy  = (raw[..., 0:2] * 2.0 + (anchor - 0.5)) * stride      # ★ 乘 stride
vis = sigmoid(raw[..., 2:3])        # ndim==3 时；ndim==2 无可见性

# ---- 损失（PoseLoss，网格单位，不乘 stride）----
xy = raw[..., 0:2] * 2.0 + (anchor - 0.5)                  # ★ 不加 stride
tgt_xy = gt_kpt_xy / stride                                 # GT 也除 stride
```

### 3.2 关键点损失（`PoseLoss._keypoint_loss`，对齐 ultra `v8PoseLoss`）

```
area     = (t_bboxes/stride → xywh)[2:].prod()      # ★ 网格单位面积（关键修复点）
d        = (pred_xy - tgt_xy)^2 的平方距离
kpt_mask = (tgt_kpt[...,2] != 0)                    # 可见性掩码
factor   = nk / (kpt_mask.sum(1)+eps)
e        = d / ((2*sigma)^2 * area * 2)
loss_pose = mean( factor * (1 - exp(-e)) * kpt_mask )      # pose_gain=12
loss_kobj = BCEWithLogits(raw_vis, kpt_mask)               # kobj_gain=1
```

> ⚠️ **最关键 bug（已修）**：`target_bboxes` 必须**先 `/stride` 转网格单位再算 `area`**，
> 否则面积是像素面积（约 1.7 万倍），`e` 被稀释到 ~1e-4，`1-exp(-e)≈e`，
> **关键点头几乎不学习 → 从零 30ep mAP = 0**。
> 修复后（`t_bboxes[fg] / stride_fg`）从零 30ep **mAP50-95 = 0.164**。

### 3.3 后处理与指标

| 项 | 说明 |
|---|---|
| NMS | 分类型 `torchvision.ops.nms`（`iou_thres=0.7` 评估） |
| `conf_thres` | 0.001（评估）/ 0.25（推理） |
| 关键点 OKS | `kpt_iou`：`e = d/((2σ)²·area·2)`，`area = w*h*0.53`（COCO 约定） |
| σ | COCO 17 点用 `OKS_SIGMA/10`；否则 `ones(nk)/nk` |
| 匹配 | 复刻 ultra `match_predictions`：按 OKS 降序 → 去重预测列 → 去重 GT 行 |
| 指标 | `mAP50` / `mAP50-95`（OKS 阈值 0.5:0.05:0.95，101 点插值） |

---

## 4. 配置与用法

### 4.1 最小配置（yolo11-pose）

```yaml
Architecture:
  task: pose
  yaml_file: torchkiln/cfg/models/11/yolo11-pose.yaml
  scale: n
  Head: {num_classes: 1, kpt_shape: [12, 2], reg_max: 16}
Loss:  {name: PoseLoss, topk: 10, alpha: 0.5, pose_gain: 12.0, kobj_gain: 1.0}
Metric: {name: PoseMetric, main_indicator: mAP50-95}
PostProcess: {name: PosePostProcess, conf_thres: 0.001, iou_thres: 0.7, kpt_shape: [12, 2]}
Train:
  dataset: {name: PoseDataset, data_dir: datasets/tiger-pose, kpt_shape: [12, 2], ...}
```
> `kpt_shape: [nk, 2]`（无可见性）或 `[nk, 3]`（x,y,visibility）。
> **数据管线会自动补可见性列**：`x/y < 0 → 0，否则 1`（对齐 ultra `verify_image_label`），
> 所以 GT 关键点内部恒为 `(N, nk, 3)`。

### 4.2 四条链路

```bash
tkiln train   -c configs/yolo/yolo11-pose.yml -o Global.pretrained_model=yolo11n-pose
tkiln val     -c configs/yolo/yolo11-pose.yml --weights output/yolo11-pose/best_accuracy.pth
tkiln predict -c configs/yolo/yolo11-pose.yml --weights ... --input imgs/
tkiln export  -c configs/yolo/yolo11-pose.yml --weights ... --onnx
tkiln check   -c configs/yolo/yolo26-pose.yml
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ | 权重体积 |
|---|---|---|
| yolo11n-pose | 2.91 | ~5.8 MB |
| yolo11s-pose | 9.95 | ~20 MB |
| yolo11m-pose | 20.97 | ~42 MB |
| yolo11l-pose | 26.23 | ~52 MB |
| yolo11x-pose | 58.89 | ~118 MB |

¹ 取自 `torchkiln/cfg/models/11/yolo11-pose.yaml` 注释。**FLOPs / 本机耗时未在本框架侧统计**。

---

## 6. 公开指标

### 6.1 权重加载

| 家族 | missing | unexpected | 说明 |
|---|---|---|---|
| yolo11-pose n/s/m/l/x | **0** | **0** | `cv4` 改为 `max(ch[0]//4, nk)` 后完全一致 |
| yolov8-pose / yolo26-pose | — | — | 同构路径（v26 用 `Pose26`，reg_max=1） |

### 6.2 端到端训练（tiger-pose，`kpt_shape=[12,2]`，imgsz=640，batch=4，SGD 同超参）

| 设置 | 本框架 pose mAP50-95 | ultralytics | 说明 |
|---|---|---|---|
| 从零 30ep | **0.164** | 0.033 | 修复 target/stride bug 前为 **0** |
| 预训练微调 30ep | **0.495** | 0.509 | mAP50 两边均 0.995 |
| 同 checkpoint 评估 | 0.0288 | 0.033 | 权重对齐验证 |

### 6.3 关键修复：`cudnn.deterministic`

`ptcore/trainers/base.py` 默认 `cudnn.deterministic=True`（`Global.cudnn_deterministic` 可关）。
非确定性算法会让本框架前向与 ultra 数值不同，cls 在 conf 0.001 附近大量翻转
→ 同权重下出 275 个检测 vs ultra 2 个，mAP 被系统性低估。
修复后同一权重评估 mAP50-95 由 **0.312 → 0.495**。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 人体姿态（COCO 17 点） | **yolo11n-pose** | 官方权重齐全、对齐最好 |
| 自定义少数关键点（车牌 4 角、手部） | **任意档 + `kpt_shape:[nk,2]`** | 数据管线自动补可见性 |
| 边缘实时 | **n** | 2.9M，姿态头只多一个塔 |
| COCO 全 17 点高精度 | **11m / 11l** | |
| 只要框/掩码 | detect / segment | 见 `docs/models/task/segment.md` |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs detect** | 多 `cv4`（`nk*ndim` 通道）；损失换 OKS + 可见性 BCE |
| **vs segment** | segment 出掩码系数→原型；pose 出每点 2D 偏移 |
| **vs HRNet / 热图法** | 热图分辨率高、精度强但慢；本头单阶段直接回归偏移，实时 |
| **v8/v11 vs v26** | v11/v8 用 Conv 头 + DFL(reg_max=16)；v26 用 DWConv 头 + reg_max=1 + end2end |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`target_bboxes` 必须先 `/stride` 转网格单位再算 area**（核心 bug，见 §3.2）。
  这是本头从零训练 mAP 为 0 的根因。
- ⚠️ **`cv4` 通道 = `max(ch[0]//4, nk)`**，不是 c1；否则权重加载不匹配。
- ⚠️ **推理与损失的关键点解码单位不同**：推理乘 stride（像素），损失不乘（网格）。
- ⚠️ **`cudnn.deterministic=True`** 默认开启，否则同权重评估被系统性低估（0.312→0.495）。
- ⚠️ **评估 fp32**：`_evaluate_loop` 强制 `autocast(enabled=False)`。
- ⚠️ **v26 训练/评估需显式指定** `-o Loss.use_one2one=true ...` 等；评估自动走 one2many+NMS。
- ⚠️ 对比必须同权重 + 同一评估器（ultra 自报用 `rect=True` 抬高数值）。
