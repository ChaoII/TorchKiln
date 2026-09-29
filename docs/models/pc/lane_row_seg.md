# 车道线 (row / seg)

> **定位**：本框架内置的**两条轻量 2D 车道线方案**（不带 3D）：
> - **`lane_row`（行式，UFLD 风格）**：把图像切成 `R` 行，每行对 `W` 个横坐标 bin 做分类；
> - **`lane_seg`（分割式）**：把车道线当**语义分割掩码**学，指标是 lane IoU / mIoU。
> **任务**：`lane_row` / `lane_seg`（`Architecture.family: yolo`，backbone 复用 YOLO11）
> **权重**：**无官方预训练权重**（本框架自研任务头，需自训）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| **本框架实现** | `torchkiln/nn/modules.py::LaneRow`（头）、`torchkiln/lane.py`（4 个 loss/metric/后处理）、<br>`torchkiln/data/lane_row.py`（row 数据）、`torchkiln/data/sem.py`（seg 数据，复用语义分割）、<br>`torchkiln/tasks/lane_row.py`、`torchkiln/tasks/lane_seg.py` |
| 结构 YAML | row：`torchkiln/cfg/models/11/yolo11-lane-row.yaml`（backbone + `LaneRow` 头）<br>seg：`torchkiln/cfg/models/11/yolo11-sem.yaml`（复用语义分割 YAML） |
| 配置 | row：`configs/yolo/yolo11-lane-row.yml`；seg：`configs/yolo/yolo11-lane-seg.yml` |
| 灵感来源 | **UFLD**（Ultra Fast Structure-aware Deep Lane Detection, ECCV 2020）的「行式分类」范式；`lane_seg` 为常规语义分割 |
| **UFLD 论文** | **不确定**：本仓库**未记录** UFLD 的 arXiv 号，也未与官方 UFLD 做数值对齐 → 按仓库纪律**不写**具体编号 |
| 对齐状态 | ⚠️ **无对齐记录**：未与任何上游实现做权重/前向/损失对齐（本任务头是自研） |

### 要解决的问题

1. **车道线是细长结构**，整图逐像素分割（seg）容易**类极不平衡**；
2. **固定行采样（row）** 把「曲线回归」退化为「每行选一个 x bin」的**分类问题**，
   既避开分割的类别不平衡，又天然表达「一条车道线每行有且仅有一个点」的结构先验；
3. 需要**两种互补口径**：seg 给稠密掩码（便于可视化/下游），row 给结构化的
   `(车道, 行) → x`（便于拟合曲线、算几何误差）。

### 关键结论（UFLD 范式）

| 数据集 | 指标 | 说明 |
|---|---|---|
| TuSimple | accuracy | UFLD 在 TuSimple 上准确率很高（**具体数字以 UFLD 论文为准，本仓库未记录**） |
| CULane | F1 | 同上；本仓库**未记录** |

> ⚠️ 本节**没有本框架实测数字**（本仓库只在 demo 数据上跑通链路，未在公开基准上评测）。

---

## 2. 网络结构

### 2.1 `lane_row` 整体框图（YOLO11 backbone + LaneRow 头）

```
图像 (B, 3, H, W=256)
  ▼ YOLO11 backbone（yolo11-lane-row.yaml，scale=n，宽度减半）
    [0] Conv(3→32,3,2)        [1] Conv(32→64,3,2)
    [2] C3k2(64→128,n=2)      [3] Conv(128→128,3,2)
    [4] C3k2(128→256,n=2)     [5] Conv(256→256,3,2)
    [6] C3k2(256→256,c3k)     [7] Conv(256→512,3,2)
    [8] C3k2(512→512,c3k)     [9] SPPF(512→512,k5)
    [10] C2PSA(512→512,n=2)
  ▼ Neck（PAN）
    [11] Up ×2 → cat[6] → C3k2(768→256,n=2)      → T1(P4)
    [12] Up ×2 → cat[4] → C3k2(384→128,n=2)      → T2(P3)
    [13] Conv(128→128,3,2) → cat[11] → C3k2(384→256,n=2) → T3(P4)
    [14] Conv(256→256,3,2) → cat[10] → C3k2(768→512,n=2) → T4(P5)
  ▼ LaneRow 头（P3=13, P4=16, P5=19 的索引 16/19/22 输入）
    lat3/lat4/lat5 = Conv(1×1) 统一到 hidden=128
    y = lat3(P3) + up(lat4(P4)) + up(lat5(P5))
    y = fuse(y)                       # 2× Conv3×3(128→128)
    y = row_pool(y)                   # AdaptiveAvgPool2d((num_rows,1)) → (B,128,R)
    y = to_lane(y.transpose(1,2))     # Linear(128 → L*W)
    y.view(B, R, L, W).permute(0,2,1,3) → (B, L, R, W)
```

- 头输出 **`(B, num_lanes, num_rows, num_bins)`** 的 bin logits。
- 默认 `num_lanes=6`、`num_rows=100`、`num_bins=101`、`hidden=128`。

### 2.2 `lane_seg` 整体框图

```
图像 (B,3,H,W) → YOLO11 backbone+neck（yolo11-sem.yaml）→ Semantic 头
  语义分割头：P3/P4/P5 → 融合 → 逐像素 (B, num_classes, H, W) logits
  输出 argmax → (B, H, W) 掩码     # 0=背景, 1..C=车道类
```

### 2.3 逐模块说明（`LaneRow` 头）

| 模块 | 结构 | 输入→输出 | 作用 |
|---|---|---|---|
| `lat3/lat4/lat5` | `Conv(c, hidden, 1, 1)` | P3/P4/P5 → 128ch | 三尺度对齐 |
| `fuse` | 2× `Conv(128,128,3,1)` | 128ch → 128ch | 融合 |
| **`row_pool`** | `nn.AdaptiveAvgPool2d((num_rows, 1))` | (B,128,H,W)→(B,128,R,1) | ★ **压成 R 行**（列全池化） |
| **`to_lane`** | `nn.Linear(128, L*W)` | (B,R,128)→(B,R,L·W) | ★ 每行分类到 `(车道, bin)` |
| reshape | `view(B,R,L,W).permute(0,2,1,3)` | → `(B,L,R,W)` | 便于 loss/metric |

### 2.4 与同类的差异

| 对比对象 | 差异点 | 为什么 | 效果 |
|---|---|---|---|
| **车道线分割（`lane_seg`）** | row 是**结构化分类**（每行 1 点/车道） | 解决细长结构类不平衡 | 更省、更稳 |
| **曲线回归（多项式/Bezier）** | row 用**离散 bin** 而非连续回归 | 避免回归不稳定 | 分类更易训 |
| **3D 车道（BEV-LaneDet）** | row/seg 都**只有 2D** | 轻量、无相机标定 | 但**无 z / 无 BEV** |
| **UFLD 官方** | 本实现是**自研头**，非逐层复刻 | 与本框架 YOLO backbone 集成 | **未做数值对齐** |

---

## 3. 输出与后处理

### 3.1 `lane_row`

| 项 | 说明 |
|---|---|
| 输出张量 | `(B, L, R, W)` bin **logits** |
| 标签 | `labels/<split>/*.txt`，每行 `valid x0 x1 ... x_{R-1}`（`x∈[0,1]`；`valid∈{0,1}`），共 `L` 行 |
| 损失 | `LaneRowLoss` = 每 `(车道,行)` 的 **CrossEntropy**(`ignore_index=-100`)，非 valid 行设为 ignore；`x → bin = round(x*(W-1))` |
| 后处理 | `LaneRowPostProcess`：`argmax(-1)` → `x = bin / (W-1)`（`ignore_value=-1.0` 表示无效） |
| 指标 | `LaneRowMetric`：`threshold_px=50`（换算成归一化阈值 `thr = 50/image_width`）；逐行误差 ≤ 阈值记 **hit**（→ `acc`）；逐车道**≥50% 有效行命中**记 TP（→ `precision`/`recall`/`F1`） |
| ⚠️ 指标口径 | 实现里 **`precision == recall`**（只统计 GT 车道，**未统计 FP 车道**）→ `F1` 实际等于两者；属简化口径 |

解码公式：

```
x_norm = argmax_w(logits[b, l, r, :]) / (W - 1)         # ∈ [0, 1]
# 还原到像素：x_px = x_norm * image_width  （image_width 来自 dataset.transform.image_size）
```

### 3.2 `lane_seg`

| 项 | 说明 |
|---|---|
| 输出张量 | `(B, num_classes, H, W)` logits（0=背景，1..C=车道类） |
| 损失 | `LaneSegLoss` = `CE(ignore_index=255)` + `0.5 * Dice` + `0.5 * Focal(gamma=2.0)` |
| 后处理 | `SemPostProcess`：`argmax(1)` → `(B,H,W)`（支持 `interpolate` 到标签尺寸） |
| 指标 | `LaneSegMetric`：**前景（lane）IoU**（main）+ 每类 mIoU，`ignore_index=255` |
| 配置 | `dice_weight: 0.5`、`focal_weight: 0.5`、`focal_gamma: 2.0` |

---

## 4. 配置与用法

```yaml
# ------ lane_row ------
Architecture:
  model_family: yolo
  task: lane_row
  algorithm: yolo11
  yaml_file: torchkiln/cfg/models/11/yolo11-lane-row.yaml
  scale: n
  in_channels: 3
  Head: {num_classes: 6, num_lanes: 6, num_rows: 100, num_bins: 101}
Loss:        {name: LaneRowLoss, ignore_index: -100}
Metric:      {name: LaneRowMetric, main_indicator: F1, threshold_px: 50}
PostProcess: {name: LaneRowPostProcess, ignore_value: -1.0}
Train:
  dataset:
    name: LaneRowDataset
    data_dir: datasets/lane_row_demo
    label_file_list: [datasets/lane_row_demo/train.txt]
    num_lanes: 6
    num_rows: 100
    transform: {image_size: 256}      # 须为 32 倍数
```

```yaml
# ------ lane_seg ------
Architecture:
  model_family: yolo
  task: lane_seg
  algorithm: yolo11
  yaml_file: torchkiln/cfg/models/11/yolo11-sem.yaml
  scale: n
  Head: {num_classes: 2}              # 0=bg, 1=lane
Loss:        {name: LaneSegLoss, dice_weight: 0.5, focal_weight: 0.5, focal_gamma: 2.0, ignore_index: 255}
Metric:      {name: LaneSegMetric, main_indicator: lane_IoU}
PostProcess: {name: SemPostProcess, conf_thres: 0.001, iou_thres: 0.7, strides: [8, 16, 32]}
Train:
  dataset:
    name: SemDataset
    data_dir: datasets/lane_seg_demo
    label_file_list: [datasets/lane_seg_demo/train.txt]
    transform: {image_size: 256}
```

目录布局（row）：

```
<data_dir>/images/<split>/xxx.jpg
<data_dir>/labels/<split>/xxx.txt     # 每行 "valid x0 ... x_{R-1}"
<train.txt / val.txt>                 # 图片相对路径
```

```bash
tkiln check -c configs/yolo/yolo11-lane-row.yml
tkiln train -c configs/yolo/yolo11-lane-row.yml
tkiln val   -c configs/yolo/yolo11-lane-row.yml --weights output/yolo11-lane-row/best_accuracy.pth

tkiln check -c configs/yolo/yolo11-lane-seg.yml
tkiln train -c configs/yolo/yolo11-lane-seg.yml
```

---

## 5. 规模与速度

| 模型 | 参数 | 来源 |
|---|---|---|
| `yolo11-lane-row`（n 档 + `LaneRow` 头 0.4317 M） | **2.5909 M** | 本框架实算 |
| `yolo11-lane-seg`（n 档 + 语义头，nc=2） | **2.5130 M** | 本框架实算 |
| 推理耗时（本机 RTX 4060 Ti） | **未统计** | — |
| 训练显存 | **未统计**（demo `image_size=256`、batch 8 可跑） | — |

> 权重体积按 fp32 粗估约 10 MB / 20 MB；本框架**未统计** `.pth` 实际体积。

---

## 6. 公开指标

| 数据集 | 指标 | 官方 | 本框架实测 |
|---|---|---|---|
| TuSimple / CULane | accuracy / F1 | UFLD 论文数字（**本仓库未记录**） | **未统计**（未在公开基准评测） |
| demo（`datasets/lane_row_demo`，12 train / 4 val） | F1 / acc | — | 仅跑通链路，**无正式指标记录** |

> ⚠️ **如实说明**：本仓库**没有任何 `lane_row` / `lane_seg` 的对齐记录或公开基准数字**。
> 已确认的只有：`tkiln check` 能构建模型并加载数据集（row：12 train / 4 val）。

---

## 7. 选型建议

| 场景 | 建议 | 理由 |
|---|---|---|
| 只需 2D 车道线曲线 | **`lane_row`** | 结构化输出、好拟合、类不平衡不敏感 |
| 需要车道线**掩码** | **`lane_seg`** | 稠密输出，可直接可视化 |
| 需要 **3D / 距离** | 用 `bev_lanedet`（见 `bev_lanedet.md`） | 本两任务**无 z** |
| 极端边缘设备 | `lane_row`（2.59M） | 比 `lane_seg` 略重一点（差不多） |
| 车道类多（左/右/...） | `lane_seg` 的 `num_classes` 直接加 | row 的 `num_lanes` 也可加 |
| 需要 FP 车道（误检）评估 | 需改 `LaneRowMetric` | 现简化口径无 FP |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **row vs seg** | row = 每行选 x（分类）；seg = 逐像素分类。两者**互补**：row 好拟合、seg 好出掩码 |
| **vs BEV-LaneDet** | BEV-LaneDet 是单目 **3D**（含 z、BEV）；本两任务是 **2D** |
| **vs UFLD 官方** | 本实现是**灵感同源但自研**的头，**未对齐** |
| **数据增强** | `LaneRowDataset` 在无 augmenter 时只做 `letterbox`；**水平翻转未同步 xs**（代码里有注释标注） |

---

## 9. 已知问题 / 注意事项

- ⚠️ **无对齐记录**：本两任务头是自研，**没有**与任何上游实现做权重/前向/损失对齐，
  也**没有公开基准数字**。文档中所有「未统计」都是真实状态。
- ⚠️ **`LaneRowMetric` 是简化口径**：只算 GT 车道的命中率，
  **没有 FP 车道** → `precision == recall`，`F1` 偏乐观。生产评估建议自行补 FP。
- ⚠️ **水平翻转不同步**：`LaneRowDataset.__getitem__` 用 augmenter 时
  `img, _ = augmenter(img, zeros)`，**xs 未跟着翻转**（代码注释已标出）。
  为避免学错，**先用无翻转的增广**。
- ⚠️ **row 的 `num_rows` 必须两处一致**：`Head.num_rows` 与
  `Train/Eval.dataset.num_rows`（`LaneRowTask.build_datasets` 会从 Head 镜像过去，
  但显式写在 dataset 里更安全）。标签行数 `R` 不一致会导致 loss 与标签错位。
- ⚠️ **`image_size` 必须是 32 的倍数**（backbone 有 5 次下采样）。
- ⚠️ `lane_seg` 的标签格式与语义分割一致：`cls` 或掩码图，`0=背景`。
- ⚠️ 本框架**未集成 FLOPs 统计**。
