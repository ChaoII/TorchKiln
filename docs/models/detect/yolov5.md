# YOLOv5（Ultralytics `u` 系列：yolov5nu / su / mu / lu / xu，以及 `n6u … x6u`）

> **定位**：Ultralytics 2020 年起维护的实时检测器；本框架对齐的是 **`u` 变体**——
> **anchor-free、无 objectness 的 split head**（与 v8 同款头）+ `v8DetectionLoss`。
> **任务**：`detect`
> **权重**：`yolov5n{u,s,m,l,x}u.pt`、`yolov5{n,s,m,l,x}6u.pt`
> （本机官方权重目录 `\\tsclient\D\项目资料\ultralytics_models\yolov5\`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **无正式论文**。Ultralytics 官方文档明确写明："Ultralytics has not published a formal research paper for YOLOv5" |
| 作者/机构 | Glenn Jocher（Ultralytics），2020-06-26 |
| 官方代码 | `https://github.com/ultralytics/yolov5`（v6/v7 线，**anchor-based**）<br>`https://github.com/ultralytics/ultralytics`（**v5u**，cfg `v5/yolov5.yaml`、`v5/yolov5-p6.yaml`） |
| 官方文档 | `https://docs.ultralytics.com/models/yolov5` |
| 建议引用 | `@software{yolov5, author={Glenn Jocher}, year={2020}, version={7.0}, doi={10.5281/zenodo.3908559}}` |
| 本框架实现 | `torchkiln/cfg/models/v5/yolov5.yaml`（P3-P5）、`yolov5-p6.yaml`（P3-P6）<br>`torchkiln/nn/modules.py`（`C3` / `SPPF` / `Bottleneck` / `Detect`） |
| 移植方式 | **结构逐层复刻** + **权重逐键对齐**（实测 n/s/m/l/x + n6u 全 **missing=0 / unexpected=1**） |

### 要解决的问题

原版 YOLOv5（anchor-based）的问题是：
1. **依赖预定义 anchors**（需针对数据集聚类）；
2. 检测头带 **objectness 分支**，多一层监督信号，且与 v8 的 split head 不统一。

**YOLOv5u 只改头**（官方文档原话）：把 **anchor-free、objectness-free 的 split head**（来自 v8）装到
v5 的 CSP 骨架 + PAN 颈部上，改善精度-速度折中；backbone/neck 与经典 v5 完全相同。

> ⚠️ **官方警告（重要）**：**用 `ultralytics/yolov5` 仓库（v6/v7、anchor-based）训练的权重
> 不能加载进 `ultralytics` 库**；要用 YOLOv5，必须从 **v5u** 检查点（如 `yolov5su.pt`）开始。
> 本框架对齐的正是 **v5u**（anchor-free）口径。

### 关键结论

| 项 | 结论 |
|---|---|
| 权重加载 | n/s/m/l/x + n6u 全 **missing=0 / unexpected=1**（仅函数式 `dfl.conv.weight`） |
| 单步 loss（同权重同输入同 GT） | 五档 total 差 **1e-6~1e-5**（见 §6） |
| 参数 | 本框架实测 **2.65M(n) ~ 97.28M(x)**，与官方 2.6/9.1/25.1/53.2/97.2M **几乎一致** |
| 端到端 | `dx_ocr` 6 epoch：框架 **0.779** vs ultra **0.787**（差 0.008） |

---

## 2. 网络结构

### 2.1 整体框图（`yolov5.yaml`，n 档）

```
Input (B,3,640,640)
  │  [0] Conv(3→64, k6, s2, p2)     ← ★ v5 的 "Focus 替代" 大核 stem
  ├─ [1] Conv(64→128, k3, s2)   → P2 160×160
  ├─ [2] C3(128→128, n=3)
  ├─ [3] Conv(128→256,k3, s2)   → P3  80×80
  ├─ [4] C3(256→256, n=6)
  ├─ [5] Conv(256→512,k3, s2)   → P4  40×40
  ├─ [6] C3(512→512, n=9)
  ├─ [7] Conv(512→1024,k3,s2)   → P5  20×20
  ├─ [8] C3(1024→1024, n=3)
  └─ [9] SPPF(1024, k5)         ← ★ 串联 3 次 MaxPool（等价 SPPCSPC 的快速版）
       │
       ▼ Neck：PAN（自顶向下 + 自底向上，共 4 次 Concat）
  [10] Conv(1024→512,k1)  [11] Upsample×2  [12] Concat([11],[6])  [13] C3(512,F, n=3)
  [14] Conv(512→256,k1)   [15] Upsample×2  [16] Concat([15],[4])  [17] C3(256,F, n=3) → T2 (P3)
  [18] Conv(256→256,k3,s2)[19] Concat([18],[14])                  [20] C3(512,F, n=3) → T3 (P4)
  [21] Conv(512→512,k3,s2)[22] Concat([21],[10])                  [23] C3(1024,F,n=3) → T4 (P5)
       ▼
  [24] Detect([17, 20, 23], nl=3, strides=[8,16,32])   ← legacy 头 + DFL(reg_max=16)
```

**`yolov5-p6.yaml`（`n6u…x6u`）**：在 P5 与 P6 之间再插一级（P5 变 **768** 通道、P6 变 **1024**），
`SPPF` 挪到层 11，头部变为 **4 尺度** `Detect([23,26,29,32], nl=4, strides=[8,16,32,64])`（本框架实测 head 在 `model.33`）。

### 2.2 逐模块说明

| 模块 | 框架实现 | 结构 | 作用 | ultra 名字 |
|---|---|---|---|---|
| `Conv` | `modules.py::Conv` | `Conv2d(bias=False)→BN→SiLU`；stem 用 `k6, s2, p2` | 基础卷积 | `Conv` |
| **`C3`** | `modules.py::C3` | `cv1/cv2 = Conv(c1,c_,1)`、`cv3 = Conv(2c_,c2,1)`；`m = n×Bottleneck(c_,c_,**k=((1,1),(3,3))**, e=1.0)` | v5/v9 的 CSP 单元 | `C3` |
| `Bottleneck` | `modules.py::Bottleneck` | `cv1=Conv(k[0]) → cv2=Conv(k[1])`，`add = shortcut and c1==c2` | 残差块 | `Bottleneck` |
| `SPPF` | `modules.py::SPPF` | `cv1(1×1) → MaxPool(k5) ×3 串联 → cat → cv2(1×1)` | 快速 SPP | `SPPF` |
| `Concat` / `nn.Upsample` | torch 原生 | `cat(dim=1)` / `nearest ×2` | PAN 融合 | `Concat` |
| **`Detect`（legacy）** | `modules.py::Detect` | `cv2=[Conv,Conv,Conv2d]`（回归）+ `cv3=[Conv,Conv,Conv2d]`（分类） | v3/v5/v8/v9 共用旧式 Conv 头 | `Detect` |

> `C3` 的内层 `Bottleneck` 用 **`k=((1,1),(3,3))`**（1×1 降维 + 3×3 卷积），本框架与 ultralytics 8.4.154 源码**逐行一致**。

### 2.3 与"上一代/同代"的差异

| 对比对象 | 经典 YOLOv5（anchor-based） | **v5u（本框架）** | 效果 |
|---|---|---|---|
| 检测头 | anchors + **objectness** + 多标签 | **anchor-free split head**（无 obj） | 免 anchor 调参 |
| 回归 | `tx,ty,tw,th` → anchors | **DFL**（`reg_max=16`，4 塔 × 16 bin） | 分布回归，边界更稳 |
| 损失 | v5 自带 obj/cls/coord（CIoU） | **`DetLoss`（v8DetectionLoss 移植，TAL 分配，topk=10）** | 与 v8/v11/v26 统一 |
| anchors | 3×3 组 | **无**（`Detect.anchors` 仅占位） | — |
| 骨架/颈部 | Darknet CSP（`C3`）+ SPPF + PAN | **同经典 v5** | 保留 v5 特性 |
| 官方 mAP(n) | **28.0** | **34.3** | **+6.3** |
| 官方 mAP(x) | **50.7** | **53.2** | **+2.5** |
| 头结构（与 v8 的关系） | — | **v5u 与 v8 走同一套 legacy 头 + 同一套损失** | 便于统一维护 |

---

## 3. 输出与后处理

### 3.1 输出张量

| 阶段 | 形状 | 说明 |
|---|---|---|
| 训练 | 3 尺度（P6 版 4 尺度），每个 `(B, 4*reg_max + nc, H, W)` = `(B, 144, H, W)`（nc=80, reg_max=16） | 原始 logits |
| 推理 | `(B, 144, N)`，`N = Σ H·W`（P3/P4/P5 或 +P6） | 交给 `DetPostProcess` |

### 3.2 Anchor 与解码

**anchor-free**：参考点为网格中心 `(j+0.5, i+0.5)`，直接回归 `l,t,r,b`。

```
cls = sigmoid(cls_logits)
l,t,r,b = dfl_project(reg_logits, 16)          # softmax(16)·[0..15] 期望
x1 = (cx - l)*stride ; y1 = (cy - t)*stride
x2 = (cx + r)*stride ; y2 = (cy + b)*stride
```

### 3.3 NMS 与阈值

| 项 | 值 |
|---|---|
| 实现 | `torchvision.ops.nms`（C++；换实现前 16800 框 ≈10 s/图 → 后 **2.5 s/50 图**） |
| 评估 | `conf 0.001` / `iou 0.7` / `max_det 300` / `max_nms 3000` |
| 推理 | `conf 0.25` / `iou 0.45` |
| strides | P3-P5：`[8,16,32]`；**P6 版必须 `[8,16,32,64]`**（`nl=4`，实测 head 在 `model.33`） |

---

## 4. 配置与用法

### 4.1 最小配置（`configs/yolo/yolov5-det.yml` 关键项）

```yaml
Architecture:
  model_family: yolo
  task: detect
  algorithm: yolov5
  yaml_file: torchkiln/cfg/models/v5/yolov5.yaml   # P6 版换 yolov5-p6.yaml
  scale: n                 # n/s/m/l/x（yaml 里有 scales 字典，v5.yaml 五档齐全）
  in_channels: 3
  Head:
    num_classes: 80
    reg_max: 16            # ⚠️ 加载官方 v5u 权重必须 16
Loss:   {name: DetLoss, topk: 10, alpha: 0.5, beta: 6.0, cls_gain: 0.5, box_gain: 7.5}
PostProcess: {name: DetPostProcess, conf_thres: 0.001, iou_thres: 0.7,
              strides: [8, 16, 32]}              # P6 版写 [8,16,32,64]
Optimizer:  {name: SGD, momentum: 0.937, nbs: 64, lr: {name: Linear, learning_rate: 0.01, lrf: 0.01}}
```

> `yolov5.yaml` 的缩放是 **`scales: {n: [0.33,0.25,1024], s: [0.33,0.50,1024], m: [0.67,0.75,1024], l: [1.0,1.0,1024], x: [1.33,1.25,1024]}`**（depth, width, max_channels）。
> `graph.py` 同时兼容 YOLOv5 老式的 **`depth_multiple` / `width_multiple`** 写法（v3 与 `plate/yolov5n-0.5.yaml` 用的是这种）。

### 4.2 四条链路

```bash
# 结构自检
tkiln check -c configs/yolo/yolov5-det.yml

# 训练（微调官方 v5u）
tkiln train -c configs/yolo/yolov5-det.yml \
    -o Architecture.Head.reg_max=16 -o Global.pretrained_model=weights/yolov5nu.pth

# 评估
tkiln val -c configs/yolo/yolov5-det.yml --weights output/yolov5-det/best_accuracy.pth

# 预测 / 导出
tkiln predict -c configs/yolo/yolov5-det.yml --weights ... --input imgs/ --output out.jpg
tkiln export  -c configs/yolo/yolov5-det.yml --weights ... --save-dir output/onnx --onnx

# P6（1280 输入）版本
tkiln train -c configs/yolo/yolov5-p6-det.yml -o Architecture.scale=n
```

---

## 5. 规模与速度

| 档位 | yaml + scale | 参数(M)¹ | 官方参数(M)² | 官方 FLOPs(B)² | mAP50-95³ | 官方 .pt⁴ | head idx / nl / strides |
|---|---|---|---|---|---|---|---|
| `n` (`nu`) | `v5/yolov5.yaml` n | **2.65** | 2.6 | 7.7 | 34.3 | 5.3 MB | 24 / 3 / 8-16-32 |
| `s` (`su`) | 同上 s | **9.15** | 9.1 | 24.0 | 43.0 | — | 24 / 3 / 8-16-32 |
| `m` (`mu`) | 同上 m | **25.11** | 25.1 | 64.2 | 49.0 | — | 24 / 3 |
| `l` (`lu`) | 同上 l | **53.23** | 53.2 | 135.0 | 52.2 | — | 24 / 3 |
| `x` (`xu`) | 同上 x | **97.28** | 97.2 | 246.4 | 53.2 | 186.1 MB | 24 / 3 |
| `n6` (`n6u`) | `v5/yolov5-p6.yaml` n | **4.33** | 4.3 | 31.3 | 42.1 (@1280) | 8.7 MB | **33 / 4 / 8-16-32-64** |

¹ 本框架实测（`nc=80`、`reg_max=16`）。² ultralytics 官方文档 Detection(COCO) 表（v5u）。³ 同官方表（n/s/m/l/x 为 640，P6 系列为 1280）。⁴ 本机官方权重磁盘体积（**fp16**）。

> ⚠️ **本框架未集成 FLOPs 计数**；上表 FLOPs 一列是**ultralytics 官方公布值**，不是本框架统计。
> P6 系列的官方 mAP：`n6u 42.1 / s6u 48.6 / m6u 53.6 / l6u 55.7 / x6u 56.8`（@1280，官方表）。

---

---

---

---

---

### 📊 FLOPs（实测）

| 项 | 值 |
|---|---|
| **FLOPs** | **7.72 GFLOPs** |
| **MACs** | **3.86 GMACs** |
| 参数量 | **2.655 M** |
| 输入规格 | `nc=80, 640x640（n 档）` |
| 测量工具 | `torch.utils.flop_counter.FlopCounterMode`（PyTorch 内置） |
| 复现脚本 | `_downloads/flops_measure*.py` |

> **口径**：`FLOPs` 是乘加各计 1 次（×2），**与 ultralytics 官方表的 GFLOPs 同口径**
> （已由 yolo11/v8 十个模型逐个吻合验证，见 [`_FLOPS.md`](_FLOPS.md)）；
> `MACs = FLOPs / 2`。
> ⚠️ `FlopCounterMode` **不计自定义算子**（NMS / probiou / iSTFT 等后处理）⇒
> 此处是**网络主干**的 FLOPs。
## 7. 选型建议

| 场景 | 推荐档位 | 理由 |
|---|---|---|
| 边缘设备、要极轻 | **`n` / `n6`** | 2.65M / 4.33M，官方 mAP 34.3 / 42.1(@1280) |
| 通用检测、精度-速度平衡 | `s` / `m` | 9.1M / 25.1M，43.0 / 49.0 |
| **大输入、大/小目标跨度大** | **`n6u/s6u` + `imgsz=1280`** | P6 多一级 64× 下采样；官方 n6u 42.1 |
| 追求高精度（离线） | `x` | 53.2（但 97.2M，不如 v8x/y11x 划算） |
| **新项目** | 建议改用 `yolov8` / `yolo11` / `yolo26` | 同量级参数下 mAP 更高（见 §8） |
| 需要 seg/pose/obb/cls | **v5 家族不支持** | 见 `docs/models/task/` |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs 经典 YOLOv5（anchor-based）** | 本框架**只实现 v5u**；经典 v5 的 anchors/obj 分支不存在，权重也不通用（官方警告：两库权重不互载） |
| **vs YOLOv8n** | v8n 37.3 mAP / 3.2M > v5nu 34.3 / 2.6M；v8 用 `C2f` 与 `k3/s2` stem，v5 用 `C3` 与 `k6/s2` stem |
| **vs YOLO11n** | 39.5 mAP / 2.6M：同参数下比 v5nu **+5.2 mAP**（`C3k2`+`C2PSA`+DWConv 头） |
| **vs YOLO26n** | 40.9 mAP + NMS-free 双头 + 无 DFL；v5 系列已属"历史基线" |
| **定位** | v5 系列在本仓库的价值是：**验证 legacy 头 / 老式缩放（depth,width_multiple）/ P6 多尺度** 三条路径 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`v5u` 与 `v8` 走同一条代码路径**（重要，避免重复踩坑）：
  两者都是 **legacy Conv 头**（`cv3 = [Conv, Conv, Conv2d]`）+ **anchor-free** + **`v8DetectionLoss`/TAL（`DetLoss`）** + **DFL(reg_max=16)**，
  **不需要** `build_targets` / anchors / obj 分支。所谓"v5 与 v8 的区别"只在 backbone/neck 的块类型上。
- ⚠️ **与仓库既有记录的一处出入（已实测确认）**：仓库 AGENTS 的 v5 小节曾写"v5 用 `legacy=False`（新 DWConv 头）"。
  按当前代码实测（本文件编写时验证）：v5 家族的 `yaml_file` 含 `/v5/` → `build_from_arch` 置 `_legacy=True` →
  头是 **legacy `Detect`（plain Conv 塔）**，`cv3[0]` 的模块序列为 `Conv → Conv2d/BN/SiLU`，
  **不是** `DWConv`；且官方 `yolov5nu.pt` 在此结构下 **missing=0 / unexpected=1**，说明 legacy 头才是正确口径。
  （v10/v11/v12/v26 才是新 DWConv 头。）
- ⚠️ **加载官方 v5u 权重必须 `Head.reg_max: 16`**：`configs/yolo/yolov5-det.yml` 默认是 1（无 DFL）。
- ⚠️ **P6 版必须把 strides 写成 4 档** `[8,16,32,64]`；P3-P5 版写 3 档。
- ⚠️ **v5 的 `scales` 与老式 `depth_multiple/width_multiple` 两种写法并存**：
  `v5/yolov5.yaml` 用 `scales`；`v3/*.yaml` 与 `plate/yolov5n-0.5.yaml` 用 `depth_multiple/width_multiple`。
  混用时注意 `scale` 名（`n/s/m/l/x`）是否在 `scales` 里，否则会静默回落到 `DEFAULT_SCALES["n"]`（参见 `yolov9.md` §9 的同类坑）。
- ⚠️ **通用修复同样适用**：`accumulate` 读取位置、`mosaic4` 裁剪中心 `(s,s)`、评估强制 fp32、`torchvision` NMS、
  `cudnn.deterministic`、DFL target 不 clamp 框坐标（详见 `yolo11.md` §9 与 `docs/models/infra/`）。
- ⚠️ `imgsz=1024/1280` 时显存吃紧（P6 建议 1280）：**batch 降到 4**；`Eval.loader.num_workers=0`。
- ⚠️ 本框架**未集成 FLOPs 统计**，§5 的 FLOPs 是官方值。
