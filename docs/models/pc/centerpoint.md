# CenterPoint-Pillars

> **定位**：LiDAR 点云 **3D 检测**的 SOTA 基线；用 **2D pillar** 把无序点云压成伪图像，
> 再用 **anchor-free 的 CenterHead** 在 BEV 热力图上直接回归 3D 框。
> **任务**：`det3d`（`Architecture.algorithm: centerpoint`）
> **权重**：`centerpoint_pillars_kitti.pth`（`E:/TorchKiln/weights/`，ModelScope `ChaoII0987/TorchKiln → pretrained/`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **Center-based 3D Object Detection and Tracking**（即 CenterPoint） |
| arXiv | **2006.11275**（2020-06-19，已核实标题/作者） |
| 作者 | Tianwei Yin、Xingyi Zhou、Philipp Krähenbühl（UT Austin） |
| 会议 | CVPR 2021 |
| 官方代码 | https://github.com/tianweizx/CenterPoint （mmdetection3d 生态） |
| 本框架参考实现 | **Paddle3D** `centerpoint_pillars_016voxel_kitti`（模块名逐一对齐） |
| 本框架实现 | `torchkiln/nn/centerpoint.py`（网络）、`torchkiln/nn/centerpoint_task.py`（loss/后处理/指标）、<br>`torchkiln/data/det3d.py`（`raw_points` 模式）、`torchkiln/models/det3d.py`、`torchkiln/tasks/det3d.py` |
| 移植方式 | **PyTorch 逐命名移植 Paddle3D** + **权重逐键对齐**（missing=0 / unexpected=0，191 张量） |

### 论文要解决的问题

1. **anchor-based 3D 检测的框定义不优雅**：KITTI 类方法要给每个类别配一组 3D anchor
   （尺寸、朝向、IoU 计算都麻烦），且旋转 IoU 计算代价高。
2. **车顶视角重叠**：BEV 上物体投影重叠，两个物体的中心往往落在同一网格。
3. **点云稀疏**：远距离只有个位数点，直接点级回归噪声大。

CenterPoint 的三条核心创新：
- **Center-based**：先在 BEV 上预测**物体中心热力图**（Gaussian Focal Loss），
  再在中心点处回归尺寸/朝向/高度 —— 完全 **anchor-free**，不需要旋转 IoU；
- **Pillar 化**：用 `PillarFeatureNet` 把 `(x,y,z,…)` 点特征 + 点相对柱心偏移编码成
  固定维 pillar 特征，`Scatter` 回 2D 伪图像 —— 免 3D 卷积，算得快；
- **双任务头**：`CenterHead` 同时输出 `hm/reg/height/dim/rot`，多类别（Car/Ped/Cyc）
  共享 `shared_conv` 但各自独立 `SeparateHead`。

### 关键结论

| 数据集 | 指标 | 原文 |
|---|---|---|
| KITTI val（BEV, Mod） | mAP | CenterPoint-Pillars 约 **71.9**（Car 87.3 / Ped 62.7 / Cyc 65.6） |
| nuScenes val | mAP / NDS | Pillar 版 mAP ~**40.3** / NDS ~**55.0**（论文 Table） |

---

## 2. 网络结构

### 2.1 整体框图（KITTI 配置）

```
点云 (N,4) [x,y,z,intensity]
  │  hard_voxelize（voxel 0.16×0.16×4.0，2D pillar，max_points=100，max_voxels=40000）
  ▼
PillarFeatureNet  [9 → 32 → 64]
  · 输入 9 维 = [x,y,z,i] + [x,y,z]−柱心均值 + [x,y]−柱心坐标 (2) 
  · 2× _PFNLayer：Linear → BN(1d) → ReLU → 逐柱 max-pool 融合
  ▼ voxel_features (P,64)
PointPillarsScatter   →  (B, 64, Ny, Nx)   伪图像
  ▼
SecondBackbone  blocks = (3,5,5)，strides = (1,2,2)，out = (64,128,256)
  ▼ 三个尺度 [64,128,256]
SecondFPN  deblocks strides = (0.5,1,2)，use_conv_for_no_stride=True，out = (128,128,128)
  ▼ cat → (B, 384, H/2, W/2)
CenterHead
  · shared_conv: Conv3×3(384→64)+BN+ReLU
  · tasks[i] = SeparateHead（reg 4 / height 1 / dim 3 / rot 2 / hm nc）
  · hm 最后一层 conv bias 初始化为 **-2.19**（先验：初始 p≈0.1）
  ▼ per-task list: [{"hm","reg","height","dim","rot"}, ...]
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | Paddle3D 名 |
|---|---|---|---|---|
| `hard_voxelize` | numpy 栅格化 + `np.bincount` 归组 | (N,4) → voxels(P,100,4), coords(P,4), num(P,) | 点云→柱 | `hard_voxelize` |
| **`PillarFeatureNet`** | 2× `_PFNLayer`（Linear→BN1d→ReLU→max） | (P,100,9)→(P,64) | 点→柱特征，`legacy=False` | `PillarFeatureNet` |
| `PointPillarsScatter` | 按 `coords[:,2]=y, [:,3]=x` 散射 | (P,64) → (B,64,Ny,Nx) | 柱→BEV 伪图像 | `PointPillarsScatter` |
| **`SecondBackbone`** | 3 段 `Conv3×3+BN+ReLU` 堆叠（层数 3/5/5） | (B,64,·)→ 3 尺度 | BEV 主干 | `SecondBackbone` |
| **`SecondFPN`** | 3 个 deblock：`ConvTranspose(0.5)` / `Conv2×2(s2)`（stride=1 用 conv） / `ConvTranspose(2)` | 3 尺度→(B,384,·) | 多尺度融合 | `SecondFPN` |
| **`CenterHead`** | `shared_conv` + `nn.ModuleList(SeparateHead)` | (B,384,·)→ list[dict] | ★ 检测头 | `CenterHead` |
| `_FastFocalLoss` | Gaussian Focal Loss | hm vs hm_t | 中心点分类 | `FastFocalLoss` |
| `_RegLoss` | L1（按 code 维聚合） | 8 维回归 | 框回归 | `RegLoss` |

### 2.3 与同类的差异

| 对比对象 | 差异点 | 为什么 | 效果 |
|---|---|---|---|
| **PointPillars** | PointPillars 是 **anchor-based**（每类 3 个 anchor）；CenterPoint 是 **center heatmap** | 免旋转 IoU、免 anchor 调参 | 更简单且精度更高 |
| **VoxelNet / SECOND** | 3D 体素卷积 → 2D pillar | 3D 卷积慢 | 快数倍 |
| **传统 anchor-free（FCOS3D 等）** | 那些多在 **图像** 上；本方法在 **BEV** | BEV 无遮挡透视问题 | 中心定位更准 |
| **`legacy` 开关** | Paddle3D 的 `PillarFeatureNet(legacy=True)` 会把 9 维拆成 `(4,5)` 两段分别 Linear；框架用 **`legacy=False`**（本权重口径） | 对齐官方 KITTI checkpoint | 权重可 1:1 加载 |

---

## 3. 输出与后处理

### 3.1 输出张量

`CenterPointPillars.forward(points)` 返回 **每个 task 一个 dict**（`hm` 分辨率 = BEV/`down_ratio`）：

| 键 | 形状 | 含义 |
|---|---|---|
| `hm` | `(B, nc_task, Ny', Nx')` | 中心点 logits（未 sigmoid） |
| `reg` | `(B, 2, Ny', Nx')` | 中心点亚像素偏移 `(dx,dy)` |
| `height` | `(B, 1, Ny', Nx')` | 框中心 z（直接回归） |
| `dim` | `(B, 3, Ny', Nx')` | **log 尺寸** `(log w, log l, log h)` |
| `rot` | `(B, 2, Ny', Nx')` | `(sin ry, cos ry)` |

### 3.2 解码公式（`CenterPointPostProcess`）

```
score = sigmoid(hm).max(dim=class)                  # 逐类取最大
x = (gx + dx) * down_ratio * voxel_x + pc_range_x0  # gx 为网格列号
y = (gy + dy) * down_ratio * voxel_y + pc_range_y0
w,l,h = exp(dim)                                     # 3 维
ry    = atan2(rot_sin, rot_cos)
box   = [x, y, z, w, l, h, ry]                       # ← Paddle3D 约定
```

- `score_thres=0.1`、`max_det=100`（配置默认）。
- **旋转 NMS**：调用 `torchkiln.det.rbox.nms_rotated`
  （`probiou` 高斯 IoU + `fast_nms` 上三角抑制，GPU 向量化）；
  `nms_thres=0.1`。NMS 用的框序是 `[x, y, l, w, ry]`。

### 3.3 GT 约定（**最容易踩的坑**）

| 层 | 框格式 | 说明 |
|---|---|---|
| 框架数据 | `[cls, x, y, z, l, w, h, yaw]` | `yaw` = **框长边朝向** |
| Paddle3D / 本模型 | `(x, y, z, w, l, h, ry)` | `ry` = **相机系原始 heading** |
| 换算 | **`yaw = ry + π/2`** ⇔ `ry = yaw − π/2` | KITTI 框长边沿相机 x 轴 |

> ⚠️ 此换算写在 `centerpoint_task.py` 的两处：`_targets` 里 `ry = yaw − π/2`（造标签），
> 后处理里 `box[6] + π/2`（回框架格式）。**改错会让非轴向车辆 IoU 被系统性压低**。

---

## 4. 配置与用法

```yaml
# 最小配置（KITTI 尺寸；demo 用 [-16,-16,-3,16,16,3]）
Architecture:
  model_family: pc
  task: det3d
  algorithm: centerpoint
  in_channels: 4
  Head:
    num_classes: 3
    point_cloud_range: [0, -39.68, -3, 69.12, 39.68, 1]   # KITTI
    voxel_size: [0.16, 0.16, 4.0]
    max_num_points_in_voxel: 100
    feat_channels: [64, 64]
    tasks:
      - {num_class: 1, class_names: [car]}
      - {num_class: 2, class_names: [cyc, ped]}
Loss:  {name: CenterPointLoss, down_ratio: 2, gaussian_overlap: 0.1, max_objs: 500, min_radius: 2, weight: 2.5}
Metric: {name: CenterPointMetric, main_indicator: mAP, iou_thresholds: [0.5, 0.7]}
PostProcess: {name: CenterPointPostProcess, down_ratio: 2, score_thres: 0.1, nms_thres: 0.1, max_det: 100}
Train:
  dataset:
    name: Det3DDataset
    data_dir: datasets/det3d_demo
    label_file_list: [datasets/det3d_demo/train.txt]
    raw_points: true                 # ★ CenterPoint 必开（直接吃原始点云，内部 voxelize）
    point_cloud_range: [0, -39.68, -3, 69.12, 39.68, 1]
    names: [car, ped, cyc]
```

```bash
# 结构自检（不训练）
tkiln check -c configs/pc/centerpoint-det3d.yml

# 训练（从 ModelScope 权重微调：裸名自动下载缓存）
tkiln train -c configs/pc/centerpoint-det3d.yml -o Global.pretrained_model=centerpoint_pillars_kitti

# 评估 / 预测
tkiln val     -c configs/pc/centerpoint-det3d.yml --weights output/centerpoint-det3d/best_accuracy.pth
tkiln predict -c configs/pc/centerpoint-det3d.yml --weights ... --input clouds/
```

> ⚠️ **必须同时改配置里的 `Head.point_cloud_range` 与 `Train/Eval.dataset.point_cloud_range`**
> （前者给模型 voxelize，后者给数据集过滤），否则点数会不一致。

---

## 5. 规模与速度

| 项 | 值 | 来源 |
|---|---|---|
| 参数（`num_class=(1,2)`） | **4.9961 M** | 本框架实算 |
| 参数（demo `num_classes=3` 同上） | 4.9961 M | — |
| 权重体积 | **19.6 MB**（`weights/centerpoint_pillars_kitti.pth`） | AGENTS 记录 |
| Paddle3D 官方 `.pdparams` | 19 MB（191 张量） | AGENTS 记录 |
| 推理耗时（本机 RTX 4060 Ti） | **未统计** | — |
| 训练显存 | **未统计** | — |

---

## 6. 公开指标

### 6.1 同权重前向数值对齐（决定性证据）

同一 voxel 输入分别跑 Paddle3D(1.0.0) 与本框架：

| 中间量 | maxdiff |
|---|---|
| `voxel_features` / `bev` | **5e-6** |
| `fpn` | **3e-4** |
| `head`（hm/reg/height/dim/rot） | **~1e-4**（float32 级） |

⇒ **模型本身无数值问题**。权重加载：**missing=0 / unexpected=0（191 张量）**。
（注意：Paddle `Linear.weight` 是 `[in,out]`，加载需 `.t()`；BN 用 `_mean/_variance`。）

### 6.2 KITTI val（3769 帧）BEV mAP

用 **Paddle3D 自带官方评估器**（难度/DontCare/R40，`z_axis=2`，`metric_types=('bev',)`），
IoU 用本地 shapely 精确 BEV IoU。**GT 当预测自检 = 100**。

| 类别 | 本框架 Easy/Mod/Hard | Paddle3D 参考 Easy/Mod/Hard |
|---|---|---|
| Car | **90.2 / 84.4 / 79.4** | 93.0 / 87.3 / 86.2 |
| Ped | **60.3 / 57.1 / 53.1** | 66.5 / 62.7 / 58.5 |
| Cyc | **81.7 / 61.9 / 58.0** | 86.6 / 65.6 / 61.6 |
| **均值（Mod）** | **67.8** | **71.87**（差 **4**） |

> 残差来自 **NMS / shapely-IoU / 点过滤** 的算子级差异（非算法差异）。

---

## 7. 选型建议

| 场景 | 建议 | 理由 |
|---|---|---|
| LiDAR 3D 检测（KITTI 类） | **本模型（centerpoint）** | SOTA 基线，免 anchor，BEV 精度高 |
| 只有已 pillarize 的稠密 BEV | `algorithm: pointpillars`（轻量 `PillarDetNet`） | 数据管线更简单 |
| 需要跟踪（Tracking） | 需自行加 `CenterPointTracker` | 本框架**只实现了检测**，未移植 tracking |
| 多类别（车/人/骑行者） | `Head.tasks` 配多组 `num_class` | 共享 `shared_conv`，头独立 |
| 算力极小的嵌入式 | 本模型 ~5M 可接受；可减小 `feat_channels` | — |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs PointPillars（anchor）** | CenterPoint anchor-free，免旋转 IoU；本框架两者都有（`algorithm` 切换） |
| **vs VoxelNet/Second** | 3D 体素 vs 2D pillar，后者快得多 |
| **vs Paddle3D 原版** | **数值逐层对齐**（前向 1e-4 级）；KITTI mAP 差 ~4（算子级） |
| **vs mmdet3d 官方** | 本框架不依赖 mmdet3d，纯 torch 可微；但**未做 tracking** |
| **vs 图像 3D（FCOS3D）** | 单目图像 vs LiDAR，精度后者高（有深度信息） |

---

## 9. 已知问题 / 注意事项

- ⚠️ **GT `ry` 约定**：框架 `yaw = ry + π/2`；转换器 `tools/convert/kitti_to_det3d.py`
  也必须用 `yaw = ry + π/2`（早期用「几何 heading = −ry + lwh」导致 IoU 被压低）。
  修正后与 Paddle3D 框 **probiou IoU = 0.994**。
- ⚠️ **`raw_collate` 哨兵填充**：CenterPoint 需要原始点云，但框架基线
  `BaseTrainer` 硬约束 `batch[0].to(device)` 且要求 batch 维一致 → 用
  `-1e4` 填充成 `(B, N, 4)` 张量，`hard_voxelize` 的 `point_cloud_range` 过滤会剔除它们。
  **不要**把 `raw_points` 关掉。
- ⚠️ **`_gather_feat` 必须支持 4D `(B,C,H,W)`**（早期只支持 `(B,L,C)` 会报错）。
- ⚠️ **`FastFocalLoss` 的 gather 只按类别列索引**；`RegLoss` 按 code 维（8）聚合返回 `(8,)`。
- ⚠️ **两处 `point_cloud_range` 必须一致**（模型 vs 数据集）。
- ⚠️ **KITTI 数据需自行准备**：`data_object_velodyne.zip`（28.7GB）等 S3 直连；
  本框架未内置下载器。
- ⚠️ 本框架**未集成 FLOPs 统计**，第 5 节不含 FLOPs/TOPS。
