# 车牌检测（plate_det）

> **定位**：车牌场景的**专用轻量检测** —— YOLOv5n 的 **0.5 宽度版**，
> **1.08 MB**，单类别（plate），针对**固定长宽比**的车牌目标做了裁剪。
> **任务**：`plate_det` | **权重**：`plate_detect.pth`（1.08 MB，✅ 已托管 ModelScope）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **无独立论文**（基于 YOLOv5 结构 + 业务定制） |
| 上游 | YOLOv5n 架构（`torchkiln/cfg/models/v5/` 家族）+ 宽度缩放 |
| 官方代码 | https://github.com/ultralytics/ultralytics（v5 家族） |
| 本框架实现 | `torchkiln/cfg/models/plate/yolov5n-0.5.yaml` + `torchkiln/nn/graph.py` |
| 权重来源 | `weights/plate_detect.pth`（1.08 MB）→ ModelScope `pretrained/` ✅ |

### 要解决的问题

通用检测器（yolo11n 等 5~10 MB）在车牌场景上**性价比不高**：

| 特点 | 车牌 | 通用检测 |
|---|---|---|
| **类别数** | **1**（plate） | 80（COCO） |
| **长宽比** | **固定且极端**（320×80 ≈ 4:1） | 各类各异 |
| **尺度** | 较小（远距离拍摄） | 各类 |
| **背景** | 变化大，但**目标密集度高**（多车排队） | 稀疏 |
| 精度需求 | **极高**（执法/闸机） | 通用 |

⇒ 把宽度减半（0.5×）可把模型压到 **1/4 体积**，而单类别场景下精度损失有限。

---

## 2. 网络结构

### 2.1 整体框图（宽度 0.5）

```
Input (B, 3, 640, 640)
 │
 ├─[0] Conv(3→16, k3, s2)              → P1  320×320     （width×0.5 后减半）
 ├─[1] Conv(16→32, k3, s2)             → P2  160×160
 ├─[2] C3(32→64)                        → P2
 ├─[3] Conv(64→64, k3, s2)              → P3   80×80
 ├─[4] C3(64→128)                       → P3
 ├─[5] Conv(128→128, k3, s2)            → P4   40×40
 ├─[6] C3(128→256)                      → P4
 ├─[7] Conv(256→256, k3, s2)            → P5   20×20
 ├─[8] C3(256→512)                      → P5
 ├─[9] SPPF(512→512, k5)                → P5   多尺度池化
 │
 └─ Neck（PANet，C3）
   [10] Upsample ×2 → cat([9],[6]) → C3(768→256)     → T1
   [11] Upsample ×2 → cat([10],[4]) → C3(384→128)    → T2
   [12] Conv(128→128,k3,s2) → cat([12],[10]) → C3(384→256) → T3
   [13] Conv(256→256,k3,s2) → cat([13],[9])  → C3(768→512)  → T4
 │
 └─ Head（**legacy Conv 头**，anchor-free，reg_max=16）
    Detect([T2(128, 80×80), T3(256, 40×40), T4(512, 20×20)])
      · cv2 回归分支: [Conv, Conv, Conv2d] → 4 × 16 通道
      · cv3 分类分支: [Conv, Conv, Conv2d] → 1 通道（单类别）
      · DFL 投影（reg_max=16）
```

> ⚠️ **通道数是 YOLOv5n 的一半**（v5n 正常是 16/32/64/128/256/512，
> 这里是 8/16/32/64/128/256 的等价缩放；上图为便于对照按「逻辑层」标注，
> 实际 `width_multiple=0.5` 由 `make_divisible` 在 `parse_model` 里施加）。

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `Conv` | `Conv2d(k,s,p,bias=False) → BN → SiLU` | 标准卷积块 |
| `C3` | `cv1` 切两半 + `m = n×Bottleneck` + `cv2` 融合 | v5 家族主特征单元 |
| `Bottleneck` | `cv1(1×1) → cv2(3×3)` + shortcut | C3 内部 |
| `SPPF` | 3 次 5×5 MaxPool（串行）+ concat + `cv2` | 扩感受野 |
| **`Detect`** | `cv2`(回归塔) + `cv3`(分类塔) + `dfl` | **legacy 头**（`cv3=[Conv,Conv,Conv2d]`） |

> ⭐ **为什么是 legacy 头**：ultralytics 的 **v5 家族用旧式 Conv 头**，
> 而 yolo11/12/26 用**新 DWConv 头**（`cv3 = [Seq[DWConv,Conv],...,Conv2d]`）。
> 框架的 `build_from_arch` 对 `v3/v5/v8/v9` 置 `_legacy=True`，
> `parse_model` 据此路由到旧式头 —— **否则权重结构不匹配**。

### 2.3 与 yolo11n 的差异

| 对比项 | **plate_det（v5n-0.5）** | yolo11n |
|---|---|---|
| 家族 | **v5**（C3 + legacy 头） | **v11**（C3k2 + C2PSA + DWConv 头） |
| 宽度 | **0.5×** | 1.0× |
| 类别 | **1（plate）** | 80 |
| 参数 | **≈0.35 M** | 2.58 M |
| 体积 | **1.08 MB** | 5.4 MB |
| 权重加载 | missing=0 | missing=0 |
| 精度 | 单类别专用 | 通用 |

> ⭐ **选型逻辑**：类别单一 + 目标形态固定 ⇒ **窄而深不如宽而浅** ⇒ 减宽度是最划算的压缩方式。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| **Anchor** | **anchor-free**（v5u 变体 / legacy v5 头）—— 以**网格中心**为参考点 |
| **输出张量** | 训练：3 个尺度 `(B, 4*16+1, H, W)`；推理：`(B, 65, N)`，`N = 80²+40²+20² = 8400` |
| **解码** | `xywh = (raw*2 + (anchor - 0.5)) * stride` |
| **DFL** | `reg_max = 16`，`dist = Softmax(logits) · [0..15]`，`ltrb = dist × 4` |
| **NMS** | **`torchvision.ops.nms`**（C++ 实现，见下） |
| iou_thres | 评估 0.7 / 推理 0.45 |
| conf_thres | 评估 **0.001**（算 mAP 必需）/ 推理 0.25 |
| max_nms | **3000**（防密集图 OOM） |

### 3.1 ⭐ NMS 优化（重要，影响所有检测）

框架的 NMS 用的是 **torchvision 的 C++ 实现**，而非 Python 逐框循环：

| 实现 | 16800 框耗时 | 说明 |
|---|---|---|
| Python 逐框 + 每步 `.item()` | **10 s/图** | 每步 GPU 同步，402 张 val 要 ~1 小时（表现为"评估卡死"） |
| **`torchvision.ops.nms`** | **2.5 s / 50 图** | **快 200×** |

**mAP 逐位不变**（纯实现替换，不改数学）。

> ⚠️ 旋转框（OBB）NMS 另用 `probiou` + `fast_nms`（见 `docs/models/task/obb.md`）。

---

## 4. 配置与用法

### 4.1 最小配置

```yaml
Global:
  model_name: plate_detect
  pretrained_model: plate_detect        # 裸名，自动从 ModelScope 下载
  device: cuda:0
  epoch_num: 100
Architecture:
  family: v5                            # v5 家族（决定 legacy 头）
  scale: 0.5                            # 宽度 0.5
  task: detect
  num_classes: 1                         # ★ 单类别
  Head:
    reg_max: 16                          # ★ DFL
    legacy: true                         # ★ v5 用旧式 Conv 头
Optimizer:
  name: SGD
  lr: {name: Linear, learning_rate: 0.01, warmup_epoch: 3}
Train:
  dataset:
    data_dir: datasets/plate
    augment:                             # ★ 不能写 {} （falsy！）
      mosaic: 1.0
      hsv: 0.015
      affine: 1.0
      fliplr: 0.5
      close_mosaic: 10
  loader:
    batch_size_per_card: 16
    num_workers: 4
Metric:
  main_indicator: mAP50-95
  conf_thres: 0.001                     # ★ 评估必须 0.001
  iou_thres: 0.7
```

### 4.2 四条链路

```bash
# 训练（从预训练微调）
tkiln train -c configs/plate/plate_det.yml

# 评估
tkiln val -c configs/plate/plate_det.yml --weights output/plate_det/best_accuracy.pth

# 预测
tkiln predict -c configs/plate/plate_det.yml --weights ... --input imgs/

# 导出 ONNX
tkiln export -c configs/plate/plate_det.yml --weights ... --onnx

# 结构自检（不训练，只验证 YAML→网络→权重加载）
tkiln check -c configs/plate/plate_det.yml
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| **参数量** | **≈0.35 M** |
| **权重大小** | **1.08 MB**（`plate_detect.pth`） |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |
| 本机推理 | 毫秒级（1 MB 模型，远快于 yolo11n） |
| 训练显存 | batch16@640 < 2 GB |
| 对比 yolo11n | 参数 **1/7.4**、体积 **1/5** |

> ⚠️ **上表的参数量为按 v5n 结构比例推算**；框架**未集成 FLOPs/精确参数统计工具**，
> 以 `tkiln check` 实际输出为准。

---

## 6. 公开指标

| 项 | 说明 |
|---|---|
| 数据集 | 车牌数据集（业务私有，`datasets/` 不入库） |
| 指标 | mAP50 / mAP50-95 / mAP75 |
| 口径纪律 | ⚠️ **必须用同一评估器评双方权重**（ultra 自报 mAP 默认 `rect=True` 会抬高数值） |

> ℹ️ 本框架**未对 plate_det 做与 ultralytics 的逐位对齐**（业务定制模型），
> 故不列「框架 vs 官方」对比数字 —— **不编造未验证的数据**。
> 结构层面的对齐已由 v5 家族整体验证（v5nu n/s/m/l/x 全 missing=0，
> 单步 loss 15.830602 ↔ 15.830601）。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| **闸机/停车场固定相机** | ✅ 本模型 | 1 MB，实时，场景固定 |
| **嵌入式摄像头** | ✅ 本模型 | 体积是关键约束 |
| **多类目标 + 车牌** | ❌ 用 yolo11n | 需同时检出车/人/牌 |
| **极端小目标**（远距离车牌） | ⚠️ 考虑加 P2 头 | 车牌在远处只有几十像素 |
| **高精度要求**（执法取证） | 考虑 yolo11s + 更大分辨率 | 1 MB 模型的召回上限有限 |
| 夜间/逆光 | 需数据侧增强 | 模型不是瓶颈 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| **vs yolo11n** | 体积 1/5、参数 1/7.4；代价是通用性与召回上限 |
| **vs yolov5nu**（标准 v5n） | 宽度 0.5 vs 1.0；同一结构、同一 legacy 头 |
| **vs PP-OCRv6_tiny_det**（1.87 MB） | 都轻；OCR 泛化到文本行，车牌版针对固定形态可再窄 |
| **vs 车牌专用检测器（如 PP-YOLOE）** | 本框架用 YOLOv5 系是历史选择；若要更高召回可换 PP-YOLOE |

---

## 9. 已知问题 / 注意事项

| 项 | 说明 |
|---|---|
| ⚠️ **`num_classes` 固定 1** | 换任务必须改；分类头 `cv3` 末层通道数随之变化 |
| ⚠️ **必须用 `legacy` 头** | v5 家族若用新 DWConv 头，**权重结构不匹配**（框架已按 family 自动路由） |
| ⚠️ **width 0.5 是极致压缩** | 精度上限低于标准 v5n；若召回不足，优先提分辨率而非加宽 |
| ⚠️ **`augment: {}` 是 falsy** | 写成空 dict **不会**建 augmenter，必须显式写 `mosaic/hsv/affine/fliplr/close_mosaic` |
| ⚠️ **评估 conf 必须 0.001** | 否则 mAP 被系统性低估（漏掉低分框） |
| ⚠️ **数据格式** | 车牌标签为 YOLO 格式 `cls cx cy w h`（归一化）；`datasets/` 不入库 |
| ⚠️ **`device: cuda:0`** | 写 `'0'` 会按非 gpu/cuda 前缀解析到 **CPU** |
| ⚠️ **密集图 NMS** | `max_nms=3000` 防 OOM；排队场景若超限会截断，必要时调大 |
