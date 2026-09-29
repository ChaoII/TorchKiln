# YOLOv8

> **定位**：Ultralytics 2023-01-10 发布的"anchor-free + split head"通用检测基线；
> 本框架中它同时是**检测/分割/姿态/OBB/分类五个任务的老式实现参照**（v8 家族全部用 **legacy Conv 头**）。
> **任务**：`detect`（同家族另有 `segment`/`pose`/`obb`/`classify`/`p2`/`p6`/`ghost` 变体）
> **权重**：`yolov8{n,s,m,l,x}.pt`（ModelScope `ChaoII0987/TorchKiln → pretrained/`；本机在 `\\tsclient\D\项目资料\ultralytics_models\yolov8\`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **YOLOv8 无独立论文**。Ultralytics 官方文档明确写明："Ultralytics has not published a formal research paper for YOLOv8" |
| 作者/机构 | Glenn Jocher、Ayush Chaurasia、Jing Qiu（Ultralytics），2023-01-10 |
| 官方代码 | `https://github.com/ultralytics/ultralytics` |
| 官方权重 | `https://github.com/ultralytics/assets/releases`（`yolov8*.pt`） |
| 官方文档 | `https://docs.ultralytics.com/models/yolov8` |
| 本框架实现 | `torchkiln/cfg/models/v8/yolov8.yaml`（结构）<br>`torchkiln/nn/graph.py::parse_model`（YAML→网络）<br>`torchkiln/nn/modules.py`（`C2f`/`SPPF`/`Bottleneck`/`Detect`） |
| 移植方式 | **结构逐层复刻** + **权重逐键对齐**（实测 n/s/m/l/x 全 **missing=0 / unexpected=1**，仅差函数式 `model.22.dfl.conv.weight`） |

### 要解决的问题

v8 相对 v5 的两点核心改动（官方口径）：
1. **anchor-free、objectness-free 的 split head**：去掉 anchors 与 obj 分支，改为"网格中心 + 两塔（reg/cls）"，
   同时把回归换成 **DFL**（`reg_max=16` 的分布回归）；
2. **骨干/颈部单元换成 `C2f`**（CSP + 2 次 1×1 融合，比 v5 的 `C3` 多一条梯度通路），并把 stem 从 `k6/s2` 改为两级 `k3/s2`。

这两点后来被 **v5u（换头）**、**v11/v12/v26（`C3k2`/注意力/双头）**继承，因此 v8 是本仓库的**路径基线**：
`DetLoss`（`v8DetectionLoss` 移植）、`make_anchors`/`dist2bbox`/`dfl_project`、`DetPostProcess` 全部以它为准。

### 关键结论

| 项 | 结论 |
|---|---|
| 权重加载 | n/s/m/l/x 全 **missing=0 / unexpected=1**（参数数逐位一致，仅差函数式 dfl） |
| 单步 loss（yolov8n） | total **19.64271 vs ultra 19.64270**（差 1e-5） |
| 梯度 | maxdiff **0.00186**（worst `model.0.conv.weight`，cuDNN 算子级） |
| 端到端（`dx_ocr` 6ep） | 框架 **0.788** vs ultra **0.783**（同权重同评估器：0.7860 vs 0.7831） |

---

## 2. 网络结构

### 2.1 整体框图（`v8/yolov8.yaml`，n 档）

```
Input (B,3,640,640)
  │
  ├─[0] Conv(3→16,   k3, s2)         → P1 320×320
  ├─[1] Conv(16→32,  k3, s2)         → P2 160×160   ← ★ 两级 k3/s2 stem（v5 是一级 k6/s2）
  ├─[2] C2f(32→64,   n=1, shortcut=True)
  ├─[3] Conv(64→64,  k3, s2)         → P3  80×80
  ├─[4] C2f(64→128,  n=2, True)
  ├─[5] Conv(128→128,k3, s2)         → P4  40×40
  ├─[6] C2f(128→128, n=2, True)
  ├─[7] Conv(128→256,k3, s2)         → P5  20×20
  ├─[8] C2f(256→256, n=2, True)
  └─[9] SPPF(256, k5)
       │
       ▼ Neck（PAN：自顶向下 + 自底向上）
  [10] Upsample×2 → [11] Concat([10],[6]) → [12] C2f(384→128, n=2)
  [13] Upsample×2 → [14] Concat([13],[4]) → [15] C2f(192→64,  n=2)   → T2 (P3)
  [16] Conv(64→64, k3,s2) → [17] Concat([16],[12]) → [18] C2f(192→128, n=2) → T3 (P4)
  [19] Conv(128→128,k3,s2)→ [20] Concat([19],[9])  → [21] C2f(384→256, n=2) → T4 (P5)
       ▼
  [22] Detect([15, 18, 21], nl=3, strides=[8,16,32])   ← ★ legacy Conv 头 + DFL(reg_max=16)
```

> ⚠️ **v8 家族在本框架中一律走 legacy 头**：`graph.py::build_from_arch` 判定 `yaml_file` 含 `/v8/` →
> `spec["_legacy"]=True` → `Detect/Segment/OBB/Pose` 走 `LEGACY_HEAD`（**plain `Conv` 塔**），
> 而**不是** v11/v12/v26 的 DWConv 塔。实测：`cv3[0] = Sequential(Conv → Conv2d/BN/SiLU)`，无 `DWConv`。

### 2.2 逐模块说明

| 模块 | 框架实现 | 结构 | 作用 | ultra 名字 |
|---|---|---|---|---|
| `Conv` | `modules.py::Conv` | `Conv2d(bias=False)→BN→SiLU` | 基础卷积 | `Conv` |
| **`C2f`** | `modules.py::C2f` | `cv1=Conv(c1,2c,1)` 切两半；`m = n×Bottleneck(c,c,shortcut,k=((3,3),(3,3)),e=1.0)`；`cv2=Conv((2+n)c,c2,1)` | ★ v8 的核心 CSP 单元 | `C2f` |
| `Bottleneck` | `modules.py::Bottleneck` | `cv1(3×3) → cv2(3×3)`，`add = shortcut and c1==c2` | 残差块（`c_=int(c2*e)`） | `Bottleneck` |
| `SPPF` | `modules.py::SPPF` | `cv1(1×1) → MaxPool(k5)×3 串联 → cat → cv2(1×1)` | 扩大感受野 | `SPPF` |
| `Concat` / `nn.Upsample` | torch 原生 | `cat(dim=1)` / `nearest ×2` | PAN 融合 | — |
| **`Detect`（legacy）** | `modules.py::Detect(_BaseHead)` | 回归塔 `cv2=[Conv,Conv,Conv2d]`→`4*reg_max`；分类塔 `cv3=[Conv,Conv,Conv2d]`→`nc`；`bias_init`：reg bias=2.0、cls bias=-log((1-0.01)/0.01) | ★ 旧式 Conv 头 | `Detect` |

### 2.3 与上一代（YOLOv5u）的差异

| 对比对象 | YOLOv5u | **YOLOv8** | 为什么改 | 效果 |
|---|---|---|---|---|
| CSP 单元 | `C3`（`m = n×Bottleneck(k=((1,1),(3,3)))`） | **`C2f`**（`cv1` 输出**切两半**，每层输出都 concat） | 更多梯度通路 | 同参下 mAP↑ |
| stem | 一级 `Conv(k6,s2,p2)` | **两级 `Conv(k3,s2)`** | 减少一次大核计算 | 更快 |
| 头 | 同为 anchor-free split head | 同 | — | 两者同损失/同后处理 |
| 回归 | DFL(16) | DFL(16) | — | — |
| 官方 mAP(n) | 34.3 | **37.3** | 综合 | **+3.0** |
| 官方参数(n) | 2.6M | **3.2M** | C2f 更宽 | +0.6M |
| 任务覆盖 | 仅检测 | **检测/分割/姿态/OBB/分类** | — | v8 是"全家桶基线" |

---

## 3. 输出与后处理

### 3.1 输出张量

| 阶段 | 形状 | 含义 |
|---|---|---|
| 训练 | 3 个尺度，每个 `(B, 4*reg_max + nc, H, W)` = `(B, 144, H, W)`（nc=80, reg_max=16） | 原始 logits |
| 推理 | `(B, 144, N)`，`N = 80²+40²+20² = 8400`（imgsz=640） | 拼三尺度 |

### 3.2 Anchor 与解码

**anchor-free**：参考点为网格中心；`DetPostProcess` → `split_head`（DFL 投影）→ `dist2bbox` → ×stride。

```
cx = j + 0.5, cy = i + 0.5
cls = sigmoid(cls_logits)
l,t,r,b = dfl_project(reg_logits, 16)      # softmax(16 个 bin)·[0..15]
x1 = (cx - l)*stride ; y1 = (cy - t)*stride
x2 = (cx + r)*stride ; y2 = (cy + b)*stride
```

### 3.3 损失（`DetLoss`，对齐 ultra `v8DetectionLoss`）

| 分量 | 说明 |
|---|---|
| 分配器 | **TAL（Task-Aligned Assigner）**：`topk=10`、`alpha=0.5`、`beta=6.0`（对齐 ultra `tal_topk=10 / alpha=0.5 / beta=6.0`） |
| box | CIoU（`box_gain=7.5`，按 `t_scores` 加权归一化） |
| cls | BCE（`cls_gain=0.5`） |
| dfl | DFL 交叉熵（`dfl_gain=1.5`，仅 `reg_max>1`） |

> ⚠️ **DFL target 的关键 bug（本框架已修）**：**不能 clamp GT 框坐标**，只能 clamp `bbox2dist` 输出的
> **ltrb 距离**到 `reg_max-1-0.01`（档位默认 16/13/10 需对齐 ultra hyp 的 `topk/alpha`）。

### 3.4 NMS 与阈值

| 项 | 评估 | 推理 |
|---|---|---|
| 实现 | `torchvision.ops.nms`（C++） | 同 |
| conf | **0.001**（ultra 口径） | 0.25 |
| iou | **0.7** | 0.45 |
| max_det / max_nms | 300 / 3000 | 300 / 3000 |
| strides | `[8,16,32]` | 同 |

---

## 4. 配置与用法

### 4.1 最小配置（`configs/yolo/yolov8-det.yml` 关键项）

```yaml
Architecture:
  model_family: yolo
  task: detect
  algorithm: yolov8
  yaml_file: torchkiln/cfg/models/v8/yolov8.yaml   # 变体：-p2 / -p6 / -seg / -pose / -obb / -cls / -ghost
  scale: n                 # n/s/m/l/x
  in_channels: 3
  Head:
    num_classes: 80
    reg_max: 16            # ⚠️ 加载官方 v8 权重必须 16（demo 默认写 1）
Loss:   {name: DetLoss, topk: 10, alpha: 0.5, beta: 6.0, cls_gain: 0.5, box_gain: 7.5, dfl_gain: 1.5}
Metric: {name: DetMetric, main_indicator: mAP50-95}
PostProcess: {name: DetPostProcess, conf_thres: 0.001, iou_thres: 0.7, strides: [8, 16, 32]}
Optimizer:
  name: SGD
  momentum: 0.937
  nbs: 64
  lr: {name: Linear, learning_rate: 0.01, lrf: 0.01, warmup_epoch: 3}
Train:
  loader: {batch_size_per_card: 16, num_workers: 4}
```

### 4.2 四条链路

```bash
# 结构自检
tkiln check -c configs/yolo/yolov8-det.yml

# 训练（从官方 COCO 预训练微调）
tkiln train -c configs/yolo/yolov8-det.yml \
    -o Architecture.Head.reg_max=16 -o Global.pretrained_model=yolov8n   # 裸名→ModelScope 自动下载

# 评估（官方口径 conf=0.001 / iou=0.7）
tkiln val -c configs/yolo/yolov8-det.yml --weights output/yolov8-det/best_accuracy.pth

# 预测 / 导出 ONNX
tkiln predict -c configs/yolo/yolov8-det.yml --weights ... --input imgs/ --output out.jpg
tkiln export  -c configs/yolo/yolov8-det.yml --weights ... --save-dir output/onnx --onnx

# 变体：P2（小目标）、P6（大输入）、OBB / seg / pose / cls
tkiln train -c configs/yolo/yolov8-p2-det.yml
tkiln train -c configs/yolo/yolov8-obb.yml -o Architecture.Head.reg_max=16
tkiln train -c configs/yolo/yolov8-seg.yml -o Architecture.Head.reg_max=16
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ | 官方参数(M)² | 官方 FLOPs(B)² | 官方 mAP50-95² | 官方 .pt³ | head idx / nl / strides |
|---|---|---|---|---|---|---|
| n | **3.16** | 3.2 | 8.7 | 37.3 | 6.2 MB | 22 / 3 / 8-16-32 |
| s | **11.17** | 11.2 | 28.6 | 44.9 | — | 22 / 3 |
| m | **25.90** | 25.9 | 78.9 | 50.2 | — | 22 / 3 |
| l | **43.69** | 43.7 | 165.1 | 52.9 | — | 22 / 3 |
| x | **68.23** | 68.2 | 257.8 | 53.9 | — | 22 / 3 |

¹ 本框架实测（`nc=80`、`reg_max=16`）；与 ultralytics yaml 注释里的官方参数**只差 16**（DFL 的 `1×1` 卷积在框架里是函数式）。
² ultralytics 官方文档 Detection(COCO) 表。³ 本机 `\\tsclient\D\项目资料\ultralytics_models\yolov8\` 磁盘体积（fp16）。

> ⚠️ **本框架未集成 FLOPs 计数**；表中 FLOPs 为 **ultralytics 官方公布值**（也写在 `yolov8.yaml` 的注释里）。

**变体参数**（yaml 注释，官方统计，nc=80）：`yolov8-p6` n 4.98M / 8.8 GFLOPs；`yolov8-ghost` n 1.87M / 5.8；
`yolov8-ghost-p2` n 2.03M / 13.8；`yolov8-ghost-p6` n 2.90M / 5.8；`yolov8-obb` n 3.23M / 9.1；`yolov8-seg` n 3.4M / 12.0。

---

## 6. 公开指标

| 数据集/口径 | 指标 | ultralytics 官方 | **本框架实测** | 差异原因 |
|---|---|---|---|---|
| COCO val2017 | mAP50-95 `n/s/m/l/x` | **37.3 / 44.9 / 50.2 / 52.9 / 53.9** | 权重加载 **missing=0 / unexpected=1** | 仅函数式 `model.22.dfl.conv.weight` |
| DOTA128（OBB，同权重推理 `yolov8n-obb`） | mAP50-95 | 0.8021（官方自评） | **0.790** | 差 0.012，NMS/probiou 算子级 |
| `dx_ocr` 车牌（nc=1，6ep 训练） | mAP50-95 | 0.783 | **0.788** | 数据顺序/随机性 |

**对齐验证（单步，同权重同输入同 GT，yolov8n，`reg_max=16`）**：

| 项 | 本框架 | ultralytics | 差异 |
|---|---|---|---|
| 权重加载 | missing=0 / unexpected=1 | — | 仅函数式 `model.22.dfl.conv.weight` |
| assigner | **n_fg=10 / t_scores.sum=1.0403** | 同 | **一致** |
| box（raw） | **0.6425** | 同 | 一致 |
| cls（raw） | **20.5645** | 同 | 一致 |
| dfl（raw） | **3.0281** | 同 | 一致 |
| total | **19.642714** | **19.642702** | 1.2e-5 |
| 梯度 maxdiff | — | — | **0.00186**（worst `model.0.conv.weight`，cuDNN 算子级） |

**端到端训练（`dx_ocr`，nc=1，6 epoch，SGD/batch=8，关增广，EMA exponential 0.9999）**：

| epoch | 1 | 2 | 3 | 4 | 5 | 6（终值） |
|---|---|---|---|---|---|---|
| 本框架 mAP50-95 | 0.604 | 0.718 | 0.732 | 0.738 | 0.749 | **0.788** |
| ultralytics | 0.530 | 0.690 | 0.736 | 0.755 | 0.777 | **0.783** |

**同权重同评估器**（用 ultra 训练出的 `best.pt` 交给框架 `tkiln val`）：框架 **0.7860** vs ultra 自评 **0.7831**（差 0.003）。

> 复测记录（确认 `REPEAT_MODULES` 移除 `Bottleneck` 后无回归）：yolov8n **19.642714 ↔ 19.642702**，梯度 **0.00186**。

---

## 7. 选型建议

| 场景 | 推荐档位 | 理由 |
|---|---|---|
| 边缘/嵌入式 | **n**（3.2M / 8.7 GFLOPs） | 实时，官方 37.3 |
| 通用检测 | **m / l** | 50.2 / 52.9，参数-精度平衡点 |
| 离线高精度 | **x** | 53.9 |
| **小目标密集** | **`yolov8-p2-det.yml`** | 4 尺度（+P2/stride 4），代价是 FLOPs 翻倍（ghost-p2 n 13.8 GFLOPs） |
| **大输入 / 单图多尺度** | **`yolov8-p6-det.yml`** | +P6/stride 64，适合 1280 输入 |
| 极轻量 | `yolov8-ghost-*` | n 仅 1.87M（ghost 卷积；精度略降） |
| 分割 / 姿态 / OBB / 分类 | 对应 `yolov8-*.yml` | v8 是"五任务全家桶"，且**都用 legacy 头** |
| 迁移学习/微调 | 直接用官方 COCO 权重 | `missing=0`，且微调不退化（BN momentum 已对齐 ultra） |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs YOLOv5u** | 官方 mAP(n)：v8 **37.3** vs v5u 34.3（+3.0）；参数 3.2M vs 2.6M；v8 用 `C2f` + 两级 k3 stem |
| **vs YOLO11** | y11n **39.5 / 2.6M**：**更少参数、更高 mAP**（`C3k2`+`C2PSA`+DWConv 头）。v8 的价值在于"老管线兼容" |
| **vs YOLOv9** | v9 用 GELAN/PGI（`RepNCSPELAN4`），v9c 53.0 / 25.5M 与 v8m 50.2 / 25.9M 相比更优 |
| **vs YOLOv10** | v10 走 NMS-free 双头；v8 仍靠 NMS，但 v8 的评估/损失是本仓库其它家族的参照实现 |
| **vs YOLO26** | v26 是 v11 的端到端后继（双头 + 去 DFL）；v8 属"基线"，不建议新项目直接用 |
| **基准数据** | 本仓库所有 detect/OBB/seg/pose 的"单步对齐"记录都以 v8 风格的 `DetLoss`+`DetPostProcess` 为准 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **v8 家族一律 legacy Conv 头（本框架已实现路由）**：
  v8/v3/v5/v9 的 `yaml_file` 路径命中 `_legacy` → 头是 `Conv-Conv-Conv2d`；
  只有 v10/v11/v12/v26 才是 **DWConv 头**。历史上曾把 v8 误路由到新头，导致 det/obb/pose/seg 参数不匹配；
  改为 legacy 后 v8n-detect 才能 **missing=0 / unexpected=1**。
- ⚠️ **加载官方权重必须 `Head.reg_max: 16`**：`configs/yolo/yolov8-det.yml` 默认写 `reg_max: 1`（无 DFL）。
- ⚠️ **DFL target 不可 clamp 框坐标**（本框架已修）：只 clamp `bbox2dist` 后的 ltrb 距离（`reg_max-1-0.01`）。
  该 bug 影响**所有 detect/OBB 训练**。
- ⚠️ **TAL 的默认超参必须与 ultra 一致**：`topk=10`、`alpha=0.5`（框架早期是 `topk=13`/`alpha=1.0`，会对不上 ultra）。
- ⚠️ **BN 语义**：ultra `initialize_weights` 把 BN 设 `momentum=0.03, eps=1e-3`（训练期）；
  但**权重文件不含 eps**，ultra 推理时用构造默认 `1e-5`。框架在 `evaluate()` 里临时恢复 `eps=1e-5`，
  否则 backbone 第 0 层就有 ~0.005 的 meanabs 差异，评估被系统性低估。
- ⚠️ **对比口径**：ultra 自报 mAP 默认 `rect=True`（矩形 letterbox）会抬高数值；
  跨端对比请用"**同一评估器评双方权重**"（本仓库做法：把 ultra 的 `best.pt` 丢进 `tkiln val`）。
- ⚠️ **评估必须 fp32**：`_evaluate_loop` 强制 `autocast(enabled=False)`；否则 yolo26 等 `reg_max=1` 的框解码会在 fp16 下失真。
- ⚠️ **NMS 必须是 `torchvision.ops.nms`**：原 Python 逐框循环 + `.item()` 同步在 16800 框时要 ~10 s/图，
  换 C++ 实现后 **2.5 s/50 图（约 200×）**，mAP 逐位不变。
- ⚠️ `imgsz=1024` 时显存吃紧 → **batch=4**；`Eval.loader.num_workers=0`（Windows）。
- ⚠️ 本框架**未集成 FLOPs 统计**，§5 的 FLOPs 是官方值。
