# 自动驾驶感知：车道线 / 点云 / 3D 检测 设计文档

> 状态：**已实施（M1a/M1b/M2/M3/M4 完成，2026-09-23）**  
> 范围：P1 车道线（两种形态）→ P2 点云底座 → P3 3D 目标检测  
> 关联：[`TASK_ARCHITECTURE.md`](TASK_ARCHITECTURE.md)、[`DATASET_FORMATS.md`](DATASET_FORMATS.md)、[`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)

---

## 0. 目标与原则

### 0.1 目标

在现有 TorchKiln 任务插件体系上，补齐自动驾驶常见感知能力：

| 期 | 任务 | 输入 | 输出 | 主指标 |
|---|---|---|---|---|
| **P1** | 车道线 `lane_seg` | 前视 RGB | 逐像素车道 mask | lane IoU / mIoU |
| **P1** | 车道线 `lane_row` | 前视 RGB | 行式回归点列（UFLD 风格） | F1 / 车道准确率 |
| **P2** | 点云分割 `pc_seg` | LiDAR 点云 (N,C) | 逐点类别 | mIoU |
| **P3** | 3D 检测 `det3d` | LiDAR（可扩展多模态） | 3D 框 (x,y,z,l,w,h,yaw,cls) | BEV AP / 3D AP（KITTI 简化） |

### 0.2 设计原则（与平台一致）

1. **不改训练循环**：全部差异进 `TaskAdapter` 11 钩子（`ptcore/task.py`）。
2. **两级分派**：`TASK_REGISTRY` + `TRAINER_REGISTRY` + CLI 别名，与 depth/semantic 同构。
3. **配置八段**：`Global / Architecture / Optimizer / Loss / Metric / PostProcess / Train / Eval`。
4. **`datasets/` 不入库**：demo 用 `tools/make_demo_data.py` 再生；真实数据走 ModelScope 或本地路径。
5. **SOTA 只移植算法核心**，不 vendor 整仓（OpenPCDet / MMDet3D / PytorchAutoDrive 作参考实现）。
6. **显存约束**：16GB 卡，点云/3D 默认 `batch≤4`；图像任务沿用现有约定。

### 0.3 非目标（本期不做）

- BEV 端到端（BEVFormer/UniAD）、多传感器融合训练、在线建图/拓扑（OpenLane-V2）。
- 稀疏卷积 CUDA 扩展（spconv/Minkowski）——P2 先用 **dense pillar/MLP**，后续可插拔。
- nuScenes 官方 NDS 全套（先 KITTI 简化 AP；NDS 列为 P3.5 可选）。

---

## 1. SOTA 参考与选型

### 1.1 车道线

| 流派 | 代表 | 选型 |
|---|---|---|
| 语义分割 | BDD100K lane、TwinLiteNet+、ERFNet 车道类 | **P1a：`lane_seg`** — 复用 `semantic` 数据/头，换 lane 专用 metric |
| 行式回归 | UFLD、CLRerNet (LaneIoU)、SCNN、LaneATT | **P1b：`lane_row`** — 新 head + 行交叉熵/结构损失 |
| 3D 车道 | PersFormer、OpenLane、BEVPointNet3D | 暂缓（需相机-BEV 变换，挂到 P3 后） |

**数据集对接（用户自备）**：TuSimple / CULane / LLAMAS（行式）；BDD100K-drivable / Cityscapes-lane（分割）。

### 1.2 点云

| 方案 | 代表 | 选型 |
|---|---|---|
| 点式 MLP | PointNet / PointNet++ | 可选 baseline，不作为主路径 |
| Pillar/BEV | **PointPillars**、SECOND | **P2 主路径**：voxel→pillar→scatter→2D CNN 或 MLP |
| 稀疏体素 | OpenPCDet PV-RCNN、Minkowski | P3 可选增强，不阻塞主路径 |

### 1.3 3D 检测

| 方案 | 代表 | 选型 |
|---|---|---|
| LiDAR 单模态 | PointPillars / SECOND / CenterPoint | **P3 主路径** |
| 相机/多模态 | Far3D、BEVFusion、TransFusion | 后续扩展（复用 P1 车道/2D 头 + lift） |
| 代码参考 | OpenPCDet `pcdet`、MMDet3D | 只抄 assigner / loss / box encode，不引入依赖 |

**数据集**：KITTI（首选，协议简单）→ nuScenes mini（可选）。

---

## 2. P1 车道线

### 2.1 两种形态并存

配置用 `Architecture.task` 区分：

| task | 数据 | 模型 | 复用度 |
|---|---|---|---|
| `lane_seg` | `masks/<split>/*.png`（uint8，0=背景，1..C=车道） | YAML 图 + `SemanticSegment` 头 | **≈semantic**：Dataset 可共用/薄包装 |
| `lane_row` | `labels/*.txt` 行式关键点 | 新头 `LaneRow` | 全新 loss/metric/postprocess |

同一套道路图可同时导出两种标签（转换脚本见 §2.5）。

### 2.2 `lane_seg`（P1a，优先落地）

**数据**  
- 直接复用 `torchkiln/data/sem.py::SemDataset`（mask 路径约定 `masks/train|val`）。  
- 或 `LaneSegDataset(SemDataset)` 薄包装：仅改默认 `num_classes`、`ignore_index=255`、可选「细线 class weight」。  
- **标签格式**：与 semantic 相同（§ DATASET_FORMATS semantic 节）。

**模型**  
- `Architecture.task: semantic` 的图模型即可（`cfg/models/26/yolo26-sem.yaml` 等）。  
- 可选：为细长车道加 **dice/focal 组合**（见 Loss），结构不改。

**Loss**  
- 默认：`SemLoss`（CE + 可选 dice）。  
- 新增 `LaneSegLoss`：`ce + dice_weight*dice + focal_weight*focal`（车道前景极稀疏，focal 有帮助）。  
  - 实现：`torchkiln/lane.py`，builder `build_lane_seg_loss`。  
  - `Loss.name: LaneSegLoss` 时走新类，否则回退 `SemLoss`。

**Metric**  
- `LaneSegMetric`：  
  - `lane_IoU`：前景（∪ 所有车道类）IoU（BDD100K 风格）  
  - `mIoU`：逐类（兼容 semantic）  
  - `main_indicator: lane_IoU`  
- 实现：混淆矩阵扩展为「背景 / 车道前景」二元 + 逐类。

**PostProcess**  
- 复用 `SemPostProcess`（argmax → (B,H,W)）。

**TaskAdapter**  
- `torchkiln/tasks/lane_seg.py::LaneSegTask`：几乎抄 `tasks/semantic.py`，仅换 `build_loss`/`build_metric` 指向 lane builder。  
- 注册：`TASK_REGISTRY["lane_seg"]`。

**Trainer**  
- `ptcore/trainers/lane_seg.py` 薄壳 → `TRAINER_REGISTRY["lane_seg"]`。

**CLI**  
- `TASK_ALIASES["lane_seg"] = "lane_seg"`（或 `lane→lane_seg` 作默认别名）  
- `FAMILY_OF["lane_seg"] = "yolo"`  
- `_config_task` 键列表增加 `"lane_seg"`（及 `"lane"`）。

### 2.3 `lane_row`（P1b，UFLD 风格）

**标签含义**  
每条车道线在 `num_rows`（默认 100）个水平行上，给出归一化横坐标 `x ∈ [0,1]` 与有效性；  
缺失/出画 = ignore（-1 或约定索引）。

**标签文件** `labels/<stem>.txt`（与图同名）：

```text
# 每条车道一行；N = num_lanes 上限（默认 6）
# 字段：valid x0 x1 ... x_{R-1}   （valid∈{0,1}，x 为 [0,1]；valid=0 时 x 可全 0）
1 0.12 0.13 ... 0.20
1 0.55 0.56 ... 0.61
0 0 0 ... 0
```

清单仍是 `train.txt`/`val.txt`（相对 `data_dir` 的图片路径），标签由 `images→labels` 推导（同 det）。

**数据**  
- `torchkiln/data/lane_row.py::LaneRowDataset`  
  - 读图 + 读行标签 → `__getitem__` 返回 `[image, target]`  
  - `target`: `FloatTensor(num_lanes, num_rows)`，无效位置填 ignore 值（实现用极大值或另附 mask）  
  - 更稳妥：`[image, xs, valid]`，`xs:(L,R)`、`valid:(L,R)` bool  
- collate：`stack image`；`xs/valid` stack（定长 L×R，无需 padding）。

**模型头 `LaneRow`**（`nn/modules.py` `@register`）

```text
输入: 多尺度 feats（取 P2/P3 或单层 FPN 融合，先做「单层简化」）
  → AdaptiveAvgPool / 固定 stride 下采样到 (B, C, R, R) 不强制
UFLD 简化结构（推荐 v0）:
  取最高层特征 (B,C,H,W)
  → conv 降到 (B, C', R, W') 再 flatten 空间
  → 或 GAP 空间宽 → (B, C', R) → Linear → (B, num_lanes, num_rows)  # 每行每车道一个 logit（横坐标分类）
v0 推荐：**行分类**（把宽离散为 W 个 bin，`num_rows × num_lanes × W` logits）
  → loss = BCE/CE（ignore 无效行）
  → 推理：argmax×bin_width → x 坐标
```

> 行分类（classification per row）比直接回归更贴 UFLD，且数值稳定；W 默认 81 或 101。  
> yaml 末层示例：`- [[], 1, LaneRow, [num_lanes, num_rows, W]]`。

**Loss `LaneRowLoss`**  
- `BCEWithLogits` 或 `CE`（每 lane × row × bin）  
- `valid` mask（无效不回传）  
- 可选结构平滑：相邻 row 惩罚跳变（`lambda * |p_r - p_{r+1}|` 对 soft 分布）— v0 可关。  
- 返回 `{"loss": ...}`，可拆 `loss_ce` 日志键。

**Metric `LaneRowMetric`**  
- 转成每行 x 坐标 → 与 GT 比较  
- **F1@50 / F1@30**（TuSimple 协议：50 像素内算命中，按行采样）  
- `main_indicator: F1`  
- 简化实现：对有效行算 `mean |x_pred - x_gt| * img_w < thr` 的 accuracy，再聚合成 F1 近似；完整 TuSimple 曲线可后补。

**PostProcess `LaneRowPostProcess`**  
- logits → softamx/argmax → `(B, L, R)` 的 x（0..1）  
- 可选输出像素坐标 `(B, L, R, 2)` 供可视化。

**TaskAdapter / Trainer / CLI**  
- `tasks/lane_row.py`、`ptcore/trainers/lane_row.py`、注册同 P1a。

### 2.4 配置模板

`configs/yolo/yolo11-lane-seg.yml`（骨架）：

```yaml
Global:
  model_name: yolo11-lane-seg
  pretrained_model: null
  device: cuda:0
  epoch_num: 100
  save_model_dir: ./output/yolo11-lane-seg
Architecture:
  model_family: yolo
  task: lane_seg
  algorithm: yolo11
  yaml_file: torchkiln/cfg/models/11/yolo11-sem.yaml   # 或复用 sem 结构
  scale: n
  Head:
    num_classes: 2          # 0=bg, 1=lane（多车道类可加大）
Loss:
  name: LaneSegLoss
  dice_weight: 0.5
  focal_weight: 0.5
Metric:
  name: LaneSegMetric
  main_indicator: lane_IoU
PostProcess:
  name: SemPostProcess
Train:
  dataset:
    data_dir: datasets/lane_demo
    label_file_list: train.txt
    ...
  loader:
    batch_size_per_card: 8
Eval: ...
```

`configs/yolo/yolo11-lane-row.yml`：

```yaml
Architecture:
  task: lane_row
  Head:
    num_lanes: 6
    num_rows: 100
    num_bins: 101          # 行分类 bin 数
Loss:
  name: LaneRowLoss
Metric:
  name: LaneRowMetric
  main_indicator: F1
  threshold_px: 50
PostProcess:
  name: LaneRowPostProcess
```

### 2.5 Demo 与格式示例

- `tools/make_demo_data.py`：`lane_seg_demo`（随机细线 mask）、`lane_row_demo`（合成弯道行标签）。  
- `tools/make_format_examples.py`：`--task lane_seg|lane_row`。  
- 真实数据：文档给 TuSimple/CULane→`lane_row`、BDD100K→`lane_seg` 的转换要点（独立脚本 `tools/convert/lane_dataset.py`，P1 后期）。

### 2.6 P1 验收

1. `python tools/make_demo_data.py --dataset lane_seg_demo`（及 row）  
2. `tkiln check -c configs/yolo/yolo11-lane-*.yml` 四行组件正常  
3. `python tools/smoke_all.py` 全绿（新配置纳入）  
4. `python tools/check_graph_build.py` 不回归  
5. demo 上 `train 1 epoch` 跑通；metric 有 `lane_IoU` / `F1` 输出  

---

## 3. P2 点云底座

### 3.1 为何单独一期

现有 collate 假定 `batch[0]=(B,3,H,W)`；点云变长、非图像。  
先打通 **数据 → 简单模型 → 分割 metric**，P3 的 3D 检测只加头与框 loss。

### 3.2 数据

**目录约定**（`data_dir` 下）：

```text
datasets/pc_demo/
  train.txt                 # 每行一个相对路径，如 clouds/train/0000.bin
  val.txt
  clouds/train/*.bin|.npy   # float32，N×3 或 N×4 (x,y,z,intensity)
  labels/train/*.txt        # 可选：每点 int 标签，一行一个；或 labels/train/*.npy int32
```

- **清单**：点云文件路径（不是图片）。  
- **扩展名**：`.npy`（推荐 demo）、`.bin`（KITTI 风格 float32 reshape）。  
- **坐标**：LiDAR 系；可选 `dataset.range: [[xmin,xmax],...]` 裁剪。  
- `PointCloudDataset`：  
  - `__getitem__` → `[pc, labels]`（`pc:(N,3|4)` float，`labels:(N,)` long；分类任务可 `labels: scalar`）  
  - 失败返回 `[]`  
- **collate**（新，不能直接 stack）：  
  - 方式 A（v0）：**体素/pillar 化后定长** → 在 dataset 内先编码，collate 只 stack  
  - 方式 B：padding + mask  
  - **选 A**：dataset 输出已 scatter 的 BEV 特征图或 pillar 索引，训练更稳。

**预处理（dataset 内，可配）**  
1. 裁剪 range、去 NaN  
2. **Pillarize**：`pc_range`、`pillar_size`（如 0.2m）、网格 `(Nx, Ny)`  
3. 每 pillar 点特征：`x,y,z,i, xc-x, yc-y, ...`（PointPillars 简化）  
4. 输出：  
   - `pillars: (P, max_points, C)` + `coords: (P, 2)` + `num_points: (P,)`  
   - 或一步 scatter 成 `(C', Ny, Nx)` dense 图（更贴 2D CNN 头）

**v0 推荐**：dense scatter `(32, Ny, Nx)`，用小型 2D backbone（复用 YOLO 图 `in_channels=32` 需改，或专用 `PillarBackbone` 手写 builder）。

### 3.3 任务 `pc_seg`

- **头**：分割头（可复用 `SemanticSegment` 思路，但输入是 BEV 特征图）。  
- **Loss**：CE per pillar/点（上采样回点：用 pillar id gather）。  
- **Metric**：点级 mIoU（按 pillar→点 展开）。  
- **PostProcess**：argmax → 每点标签。

### 3.4 配置要点

```yaml
Architecture:
  task: pc_seg
  model_family: pc          # 或 yolo + 手写 builder
  Head:
    num_classes: 5
Loss: { name: SemLoss }     # 可复用
Train:
  dataset:
    data_dir: datasets/pc_demo
    label_file_list: train.txt
    pc_range: [-40, -40, -3, 40, 40, 1]
    pillar_size: [0.2, 0.2, 4]
    max_points_per_pillar: 32
  loader:
    batch_size_per_card: 4
```

### 3.5 平台适配点（点云专用）

| 钩子 | 处理 |
|---|---|
| `forward_train(model, images, batch)` | **`images` 实为 collate 后的 pc 张量**；`model(pc)` 不假设 3 通道 |
| `eval_step` | 同样从 `batch[0]` 取点云 |
| `sample_count` | batch 内样本数（或点数，配置决定） |
| `BaseTrainer` 日志/AMP | 保持；确保输入 dtype float32 |
| 预训练 | 点云任务默认 `pretrained_model: null`（图像权重不适用） |

> 若 `BaseTrainer` 对 `images` 有硬编码 3 通道假设，**只在 TaskAdapter 层规避**，不改 base（符合原则 1）。

### 3.6 P2 验收

1. demo 点云生成 + `tkiln check`  
2. 1 epoch 训练 loss 下降趋势正常  
3. mIoU 指标输出  
4. 不破坏现有 80+ smoke  

---

## 4. P3 3D 目标检测 `det3d`

### 4.1 输入

- v0：**纯 LiDAR**（与 P2 同管线：pillars → BEV）  
- 配置可留 `modalities: [lidar]`，为 camera/fusion 预留  

### 4.2 标签格式（KITTI 简化）

`labels/<stem>.txt` 每行一个 3D 框：

```text
# cls  x y z  l w h  yaw   （LiDAR 系，z 为中心或底部——配置 metadata.z_mode）
# 可选 difficulty: 0 easy 1 moderate 2 hard
car  12.1  -3.4  0.8  3.7 1.6 1.5  1.57
ped   5.0   2.2  0.9  0.6 0.6 1.7  0.10
```

清单仍为点云路径列表。

### 4.3 模型

```text
pillars → pillar feature → scatter BEV (B,C,H,W)
  → backbone（小型 conv 或 YOLO 手写改 in_channels）
  → 检测头 Det3DHead:
      分类: num_classes
      回归: 7 dof — Δx Δy z log l log w log h sin_yaw cos_yaw（或直接 yaw）
```

- **anchor-based（PointPillars/SECOND）** 或 **anchor-free center（CenterPoint）**  
  - **v0 选 CenterPoint 式中心 heatmap**：实现更短、无 anchor 调参  
  - v1 可补 anchor 版对齐 OpenPCDet  

### 4.4 Loss

| 分量 | 形式 |
|---|---|
| cls | Focal Loss（heatmap / CE） |
| box | L1 或 SmoothL1（7 维，按 encode 归一） |
| yaw | 分类 bin 或 sin/cos L1 |

返回 `{"loss", "loss_cls", "loss_box", ...}`。

### 4.5 Metric

- **v0：BEV IoU AP**（鸟瞰 2D IoU，阈值 0.5/0.7）+ 简化 3D IoU（旋转框 IoU，可复用 `probiou` 扩到 3D 或分解）  
- 类别：与数据 `names` 对齐  
- `main_indicator: mAP`  
- KITTI 官方 40 点 recall 曲线：P3.5  
- nuScenes NDS：P3.5 可选  

### 4.6 PostProcess / NMS

- 中心 peak 提取（3×3 maxpool + 阈值）→ 解码 7 dof  
- **BEV NMS**：旋转 IoU（复用 `det/rbox.py` 思路升到 3D：BEV IoU + 高度重叠过滤）

### 4.7 配置骨架

```yaml
Architecture:
  task: det3d
  Head:
    num_classes: 3
    out_size: [256, 256]      # BEV 网格
    pc_range: [-40, -40, -3, 40, 40, 3]
Loss:
  name: Det3DLoss
  focal_gamma: 2.0
  box_weight: 0.25
Metric:
  name: Det3DMetric
  main_indicator: mAP
  iou_thresholds: [0.5, 0.7]
PostProcess:
  name: Det3DPostProcess
  score_thres: 0.1
  nms_thres: 0.2
Train:
  loader: { batch_size_per_card: 4 }
```

### 4.8 P3 验收

1. demo 3D 框数据 + check + 1 epoch  
2. mAP 可计算（哪怕随机初始化很低）  
3. 有真实 KITTI 子集时，与 OpenPCDet PointPillars 同数据量级趋势一致（**不强制数值对齐**，平台此前对齐标准针对 2D ultra）  
4. smoke / graph 回归通过  

---

## 5. 统一注册与 CLI 清单

新增 task 一览（实施时逐项勾选）：

| task | data | loss/metric/pp | TaskAdapter | Trainer | CLI alias | configs |
|---|---|---|---|---|---|---|
| `lane_seg` | 复用/薄包 sem | `lane.py` LaneSeg* | `tasks/lane_seg.py` | `trainers/lane_seg.py` | `lane_seg` | `yolo11-lane-seg.yml` |
| `lane_row` | `data/lane_row.py` | `lane.py` LaneRow* | `tasks/lane_row.py` | `trainers/lane_row.py` | `lane_row` | `yolo11-lane-row.yml` |
| `pc_seg` | `data/pc.py` | 复用 Sem* 或 pc 专属 | `tasks/pc_seg.py` | `trainers/pc_seg.py` | `pc_seg` | `pc-pointpillars-seg.yml` |
| `det3d` | `data/det3d.py` | `det3d.py` Det3D* | `tasks/det3d.py` | `trainers/det3d.py` | `det3d` | `pc-pointpillars-det3d.yml` |

**注册文件（每 task 3 处 + 可选 1 处）**：

1. `torchkiln/tasks/__init__.py` — import + `TASK_REGISTRY`  
2. `ptcore/trainers/__init__.py` — import + `TRAINER_REGISTRY`  
3. `torchkiln/cli.py` — `TASK_ALIASES`、`FAMILY_OF`、`_config_task` 键列表  
4. （可选）`tools/check_graph_build.py::DEFAULT_SIZES` 加默认输入尺寸  

**可选组件**：

- `tools/make_demo_data.py` BUILDERS  
- `tools/make_format_examples.py`  
- `tools/infer/predict_yolo.py` 渲染分支  
- `datasets/manifest.yml`（若上架 ModelScope 示例）  
- 文档：`TASK_ARCHITECTURE.md` §3 表、`DATASET_FORMATS.md`、`MODEL_ZOO.md`、`CONFIG_REFERENCE.md`、README 任务计数  

---

## 6. 目录与文件规划（新增）

```text
torchkiln/
  data/
    lane_row.py          # P1b
    pc.py                # P2 点云 Dataset + collate + pillarize
    det3d.py             # P3（可继承 pc.py）
  lane.py                # P1 loss/metric/postprocess + builders
  det3d.py               # P3 loss/metric/postprocess + builders
  tasks/
    lane_seg.py  lane_row.py  pc_seg.py  det3d.py
  nn/modules.py          # + LaneRow 头（@register）；Det3D 头
  nn/graph.py            # HEAD_CLASSES + parse_model 分支（LaneRow/Det3D）
ptcore/trainers/
  lane_seg.py  lane_row.py  pc_seg.py  det3d.py
configs/yolo/
  yolo11-lane-seg.yml  yolo11-lane-row.yml
configs/pc/              # 或 configs/yolo/ 下 pc-*.yml
  pointpillars-seg.yml  pointpillars-det3d.yml
tools/
  make_demo_data.py      # +4 builders
  convert/lane_dataset.py  # 可选，真实数据转换
docs/
  AUTONOMOUS_DRIVING.md  # 本文件
```

---

## 7. 里程碑与顺序

| 里程碑 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| **M0** | 本文档评审通过 | — | ✅ |
| **M1a** | `lane_seg` 全链路 + demo + smoke | M0 | ✅ |
| **M1b** | `lane_row` 头/loss/metric + demo | M1a（注册模式复用） | ✅ |
| **M2** | `pc.py` + pillarize + `pc_seg` + demo | M0 | ✅ |
| **M3** | `det3d` 头/loss/AP/NMS + demo | M2 | ✅ |
| **M4** | 文档同步、format examples、可选 convert 脚本 | M1–M3 | ✅ |

**验收（2026-09-23）**：`smoke_all` **80 OK, 0 FAIL**（含 lane_seg/lane_row/pointpillars-seg/pointpillars-det3d）；
`check_graph_build` **55 OK, 0 FAIL**；四任务 1 epoch 训练 + 评估均跑通；`make_format_examples` 四任务模板已生成。

**实施顺序**：M1a → M1b → M2 → M3 → M4（图像线先闭环，再开点云线）。

---

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 行式车道与 UFLD 指标细节多 | v0 用简化 F1@px；TuSimple 官方曲线后补 |
| 点云 collate 破坏 base 假设 | 全部在 TaskAdapter/collate 吸收；必要时 `forward_train` 自建 |
| BEV 网格显存 | 默认 range ±40m、0.2m → 400×400 可偏大；先 0.4m 或 800×800→降采样；batch=4 |
| 无 spconv 性能差 | dense pillar 足够 demo/小数据；文档注明可插拔后端 |
| 3D AP 实现有坑 | 复用/升维现有 `probiou`、旋转 NMS；单测小 GT 集 |
| smoke 基线被冲垮 | 新配置进 glob 但保持 0 fail；不改已有 yml 语义 |

---

## 9. 与 ultralytics / 外部对齐说明

- **车道线/点云/3D 不在 ultra 对齐范围**（ultra 无这些任务）。验收以「平台自洽 + 指标可复现」为准，不套用 AGENTS 中 2D det/seg 的「对齐 ultra mAP」标准。  
- 算法参考实现见 §1；若需数值参考，P3 可选对 **OpenPCDet PointPillars + KITTI 样例** 做量级对照（非本期门禁）。

---

## 10. 评审检查单（实施前确认）

- [x] task 命名：`lane_seg` / `lane_row` / `pc_seg` / `det3d` 是否可接受  
- [x] 车道线行式标签：bin 分类（推荐）vs 直接回归  
- [x] 点云 v0：dense pillar + 小 CNN，不引入 spconv  
- [x] 3D v0：CenterPoint 式 anchor-free，不做 nuScenes NDS  
- [x] configs 放 `configs/yolo/` 还是新建 `configs/pc/`（lane 在 yolo，pc 在 `configs/pc/`）  
- [x] demo 数据是否够 smoke；真实数据是否由用户提供路径  
- [x] 实施顺序 M1a→M1b→M2→M3 是否认可  

---

*文档版本：v1.1 · 2026-09-23 · M1–M4 已实施完毕，验收见 §7。*
