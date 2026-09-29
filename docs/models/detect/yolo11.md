# YOLO11

> **定位**：Ultralytics 2024 年 9 月发布的主力检测家族；用 **C3k2 + C2PSA** 替换 YOLOv8 的 C2f，
> 在**几乎不增加参数**的前提下提升精度与小目标能力。
> **任务**：`detect`（同时可扩到 `segment`/`pose`/`obb`/`classify`/`semantic`）
> **权重**：`yolo11{n,s,m,l,x}.pth`（ModelScope `ChaoII0987/TorchKiln → pretrained/`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **YOLO11 无独立论文**（Ultralytics 未发表），创新点见官方文档与代码 |
| 机构 | Ultralytics |
| 官方代码 | https://github.com/ultralytics/ultralytics |
| 官方权重 | https://github.com/ultralytics/assets/releases（`yolo11*.pt`） |
| 本框架实现 | `torchkiln/cfg/models/11/yolo11.yaml`（结构）<br>`torchkiln/nn/graph.py::parse_model`（YAML→网络）<br>`torchkiln/nn/modules.py`（`C3k2`/`C2PSA`/`Detect` 等） |
| 移植方式 | **结构逐层复刻**（YAML 对齐 ultra）+ **权重逐键对齐**（missing=0/unexpected=1，仅差函数式 DFL） |

### 论文/文档要解决的问题

YOLOv8（2023）用 **C2f** 作为骨架单元，效果稳定但存在两个问题：
1. **小目标与遮挡场景**精度不足 —— 缺乏显式的**空间注意力**；
2. **感受野**受限于 3×3 堆叠，**全局上下文**建模弱。

YOLO11 的两条核心改进（官方说法）：
- **C3k2**：把 C2f 的 Bottleneck 换成**可选 C3k 块**（`c3k=True/False` 控制），
  用 **两个不同核尺寸的卷积分支**替代单一瓶颈，提高特征多样性；
- **C2PSA**：在 SPPF 之后插入 **Position-Sensitive Attention**（多头注意力 + FFN），
  以极低代价引入**全局注意力**。

### 关键结论

| 档位 | COCO mAP50-95 | 相对 YOLOv8 同档 | 参数变化 |
|---|---|---|---|
| n | **39.5** | +2.2 | -0.0M |
| s | **47.0** | +2.1 | +0.0M |
| m | **51.5** | +1.3 | +0.0M |
| l | **53.4** | +0.6 | +0.0M |
| x | **54.7** | +0.4 | +0.0M |

> ⇒ **小模型收益更大**（n 提升 2.2），因为 C2PSA 的全局注意力对小模型帮助显著。

---

## 2. 网络结构

### 2.1 整体框图（YOLO11n，n 档）

```
Input (B,3,640,640)
  │
  ├─[0] Conv(3→16, k3, s2)                     → P1  320×320
  ├─[1] Conv(16→32, k3, s2)                    → P2  160×160
  ├─[2] C3k2(32→64,  n=1, c3k=False)           → P2
  ├─[3] Conv(64→64, k3, s2)                    → P3  80×80
  ├─[4] C3k2(64→128, n=2)                      → P3
  ├─[5] Conv(128→128, k3, s2)                  → P4  40×40
  ├─[6] C3k2(128→128, n=2)                     → P4
  ├─[7] Conv(128→256, k3, s2)                  → P5  20×20
  ├─[8] C3k2(256→256, n=2)                     → P5
  ├─[9] SPPF(256→256, k5)                      → P5  ← 多尺度池化
  └─[10] C2PSA(256→256, n=2)                   → P5  ← ★ 全局注意力
       │
       ▼ Neck（PAN：自顶向下 + 自底向上）
  [11] Upsample ×2 → cat([10], [6]) → C3k2(384→128, n=2)   → T1
  [12] Upsample ×2 → cat([11], [4]) → C3k2(192→64,  n=2)   → T2
  [13] Conv(64→64, k3, s2) → cat([13],[11]) → C3k2(192→128,n=2) → T3
  [14] Conv(128→128,k3,s2) → cat([14],[10]) → C3k2(384→256,n=2) → T4
       │
       ▼ Head（新 DWConv 头，3 尺度）
  Detect([T2(64ch,80×80), T3(128ch,40×40), T4(256ch,20×20)])
      · cv2 回归分支（DWConv 塔）→ 64 ch（4×reg_max）
      · cv3 分类分支（DWConv 塔）→ nc ch
      · DFL 投影（reg_max=16）
```

### 2.2 逐模块说明（本框架实现文件）

| 模块 | 结构 | 输入→输出 | 作用 | ultra 中的名字 |
|---|---|---|---|---|
| `Conv` | `Conv2d(k,s,p,bias=False) → BN → SiLU` | 任意→任意 | 标准卷积块 | `Conv` |
| **`C3k2`** | `cv1(1×1)` 切两半；`m = n×Bottleneck`（`c3k=False`）或 `n×C3k`（`c3k=True`）；`cv2(1×1)` 融合 | (B,c1,H,W)→(B,c2,H,W) | ★ 主特征提取单元 | `C3k2` |
| **`C2PSA`** | `cv1` 切两半；`m = n×(PSABlock)`；`cv2` 融合 | 同上 | ★ 位置敏感注意力（全局） | `C2PSA` |
| `PSABlock` | `Attention(qkv=Conv(1×1), proj=Conv(1×1), pe=DWConv(3×3)) + FFN` | 同维 | 自注意力 + 前馈 | `PSABlock` |
| `SPPF` | 3 次 5×5 MaxPool（串行，等价多尺度）+ concat + `cv2` | 同维 | 扩大感受野 | `SPPF` |
| **`Detect`** | `cv2`(回归塔, DWConv×2 + Conv2d) + `cv3`(分类塔, DWConv×2 + Conv2d) + `dfl`(1×1 impl.) | 3 尺度特征→3×(B,H,W,C) | ★ 新 DWConv 头 | `Detect` |
| `DWConv` | `Conv2d(k,s,p,groups=c)`（深度可分离） | 同维 | 降参降算力 | `DWConv` |

### 2.3 与 YOLOv8 的差异（★ 核心对比）

| 对比对象 | YOLOv8 | **YOLO11** | 为什么这么改 | 效果 |
|---|---|---|---|---|
| 骨架单元 | `C2f`（Bottleneck×n） | **`C3k2`**（`C3k` 块可选） | C3k 用**两个不同核尺寸的卷积**，特征更多样 | 同参下 mAP↑ |
| 颈部单元 | `C2f` | **`C3k2`** | 同上 | |
| 全局注意力 | ❌ 无 | ✅ **`C2PSA`**（SPPF 后） | 引入**位置敏感**的全局上下文 | 小目标↑ |
| 检测头 | `cv2/cv3 = Seq[Conv,Conv,Conv2d]` | **`Seq[DWConv,Conv],[DWConv,Conv],Conv2d`** | **深度可分离**降参 | 参数↓ 精度↑ |
| 分类头输出 | `nc` | `nc` | — | |
| 回归 | reg_max=16 + DFL | reg_max=16 + DFL | — | |
| **参数（n）** | 3.2M | **2.6M** | DWConv 省参 | **-19%** |
| **mAP（n）** | 37.3 | **39.5** | 综合 | **+2.2** |

> ⚠️ **C3k2 的 `c3k` 按规模不同**：ultra `parse_model` 对 **`scale ∈ {m,l,x}` 强制 `c3k=True`**，
> n/s 用 `c3k=False`（即普通 Bottleneck）。本框架已复刻（`graph.py`），
> 否则 m/l/x 权重加载会通道/结构不符。

---

## 3. 输出与后处理

### 3.1 输出张量

训练/推理时 `Detect.forward` 返回 3 个尺度（训练）或拼接后的预测（推理）：

| 阶段 | 形状 | 含义 |
|---|---|---|
| 训练 | `[P3,P4,P5]`，每个 `(B, 4*reg_max+nl_cls, H, W)` | 各尺度原始 logits |
| 推理 | `(B, 4*reg_max + nc, N)`，`N = Σ H*W` | 展平后拼三尺度 |

### 3.2 Anchor

**YOLO11 是 anchor-free**（继承 v8）：以**网格中心**为参考点，每个位置直接回归 `l,t,r,b` 四边距离。
（框架中 `Detect.anchors`/`strides` 仅作计算辅助，不是 v5 那种候选框。）

### 3.3 解码公式

```
# 网格中心（anchor point）
cx = (j + 0.5),  cy = (i + 0.5)              # j,i 为网格坐标

# 分类
cls_score = sigmoid(cls_logits)               # 多标签 sigmoid（非 softmax）

# 框回归（DFL）
dist = Softmax(dfl_logits).matmul([0..reg_max-1])   # 期望值
l,t,r,b = dist × 4 组 → 像素单位
x1 = (cx - l) * stride,  y1 = (cy - t) * stride
x2 = (cx + r) * stride,  y2 = (cy + b) * stride
```

**DFL 原理**：把每边距离建模为 `reg_max=16` 个离散 bin 上的分布，取**期望**作为回归值。
（对齐 ultra `DFLoss`；本框架修过一个关键 bug：**GT 框坐标不能 clamp**，
只有 `bbox2dist` 输出的 ltrb 距离才 clamp 到 `reg_max-1-0.01`。）

### 3.4 NMS

| 项 | 说明 |
|---|---|
| 实现 | **`torchvision.ops.nms`**（C++ 实现） |
| 为什么换 | 原 Python 逐框循环 + 每步 `.item()` GPU 同步 → 16800 框 **10s/图**；换后 **2.5s/50图（200×）** |
| 阈值 | `iou_thres` 默认 0.7（评估）/ 0.45（推理） |
| conf 阈值 | 评估 0.001（ultra 口径）/ 推理 0.25 |
| `max_nms` | 3000（防密集图 OOM） |

---

## 4. 配置与用法

### 4.1 最小配置（COCO 检测）

```yaml
Global:
  model_name: yolo11n_coco
  device: cuda:0
  epoch_num: 100
Architecture:
  family: yolo11          # 家族（决定用哪个 model yaml）
  scale: n                # 档位 n/s/m/l/x
  task: detect
  num_classes: 80
Optimizer:
  name: SGD
  lr: {name: Linear, learning_rate: 0.01, warmup_epoch: 3}
Train:
  dataset: {data_dir: datasets/coco, ...}
  loader:  {batch_size_per_card: 16, num_workers: 4}
```

### 4.2 四条链路

```bash
# 训练（从官方预训练微调）
tkiln train -c configs/yolo/yolov8-det.yml \
    -o Global.pretrained_model=yolo11n   # 裸名 -> 自动从 ModelScope 下载

# 评估
tkiln val -c configs/yolo/yolov8-det.yml --weights output/xxx/best_accuracy.pth

# 预测
tkiln predict -c configs/yolo/yolov8-det.yml --weights ... --input imgs/

# 导出 ONNX
tkiln export -c configs/yolo/yolov8-det.yml --weights ... --onnx

# 结构自检（不训练，只验证 YAML→网络能否构建 + 权重能否加载）
tkiln check -c configs/yolo/yolov8-det.yml
```

---

## 5. 规模与速度

| 档位 | 参数(M) | COCO mAP50-95 | 权重体积(.pth) | 本机推理(ms)¹ | 训练显存(batch16,640)² |
|---|---|---|---|---|---|
| n | 2.58 | 39.5 | ~5.4 MB | ~3 | ~4 GB |
| s | 9.41 | 47.0 | ~18.5 MB | ~6 | ~6 GB |
| m | 20.05 | 51.5 | ~40 MB | ~13 | ~9 GB |
| l | 25.28 | 53.4 | ~50 MB | ~18 | ~11 GB |
| x | 56.87 | 54.7 | ~113 MB | ~30 | ~14 GB |

¹ 本机 RTX 4060 Ti 16GB，fp16，batch=1，640×640（含预处理，粗估）
² 实际可用 `batch=16`；**imgsz=1024 时 batch 需降到 4**（AGENTS 记录）

> ⚠️ **本框架未集成 FLOPs 计数**（如需可加 `thop`）。上表 FLOPs 列留空以避免编造数字。

---

## 6. 公开指标

| 数据集 | 指标 | Ultralytics 官方 | **本框架实测** | 差异原因 |
|---|---|---|---|---|
| COCO val2017 | mAP50-95 (n) | 39.5 | 权重加载 **missing=0 / unexpected=1**（仅函数式 dfl） | — |
| DOTA128（OBB，同权重推理） | mAP50-95 (v11n-obb) | 0.821 | **0.8005** | 差 **0.021**，NMS/probiou 算子级 |
| dx_ocr 车牌（自训 6ep, nc=1） | mAP50-95 | 0.791 | **0.782** | 差 0.009，数据 shuffle 差异 |
| ESC 等 | — | — | — | — |

**对齐验证（单步，同权重同输入同 GT）**：

| 项 | 本框架 | ultralytics | 差异 |
|---|---|---|---|
| 权重加载 | missing=0 / unexpected=1 | — | 仅函数式 `dfl.conv.weight` |
| assigner 匹配 | n_fg=10 / t_scores.sum=1.3955 | 同 | **一致** |
| loss（box/cls/dfl） | 0.5508 / 23.3453 / 3.1119 | 同 | **一致** |
| total loss | 20.47179 | 20.47182 | **3e-5** |
| 梯度 maxdiff | — | — | 0.0216（cuDNN 卷积反向算子级） |

---

## 7. 选型建议

| 场景 | 推荐档位 | 理由 |
|---|---|---|
| 边缘/嵌入式（Jetson、树莓派） | **n** | 2.6M / ~5 MB，实时 |
| 服务器通用检测 | **m / l** | 精度-速度平衡 |
| 追求最高精度（离线） | **x** | 54.7 mAP |
| **小目标密集**（无人机、遥感） | **n/s + P2 头** | 用 `yolo26-p2.yaml` 式的 P2 增强 |
| 需要**实例分割** | 见 `docs/models/task/segment.md`（`yolo11-seg.yaml`） | 同一 backbone，加 `cv4`+`Proto` |
| 需要**旋转框**（遥感/文字） | 见 `task/obb.md` | 加 angle 塔 |
| 需要**关键点** | 见 `task/pose.md` | 加 `cv4` 关键点塔 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs YOLOv8** | 同参数下 mAP **+0.4~2.2**（小模型收益大）；头换 DWConv 后**参数 -19%** |
| **vs YOLOv10** | v10 主打 **NMS-free**（one2one 头），推理延迟更稳定；v11 精度更高 |
| **vs YOLO12** | v12 用 **A2C2f**（区域注意力），参数量更大、更偏精度 |
| **vs YOLO26** | v26 引入 **end2end（one2one）+ reg_max=1（L1 回归）**，是 v11 的**端到端后继** |
| **vs RT-DETR** | Transformer 检测器，无 NMS 但更重；v11 在同等精度下更轻更快 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`C3k2` 的 `c3k` 按规模变化**：m/l/x 必须 `c3k=True`（本框架已复刻），
  否则 m/l/x 权重加载会结构不符。
- ⚠️ **DFL target 不可 clamp 框坐标**（本框架已修）：只 clamp `bbox2dist` 后的 ltrb 距离。
  该 bug 会影响**所有 detect/OBB 训练**。
- ⚠️ **对比必须同评估器 + 同 val 清单**：ultra 自报 mAP 默认 `rect=True`（矩形 letterbox）
  会抬高数值，与本框架方图不可直接比。
- ⚠️ **imgsz=1024 时显存吃紧**：有记录 `batch=8` 即 OOM，**用 batch=4**。
- ⚠️ 本框架**未集成 FLOPs 统计**，上表不含 FLOPs/TOPS。
