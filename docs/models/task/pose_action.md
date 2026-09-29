# 骨架行为识别（ST-GCN）

> **定位**：**基于骨架序列**的行为识别 —— 空间图卷积（人体关节点拓扑）
> + 时序卷积（1D TCN），无 3D 卷积，适合跌倒/打架/攀爬/挥手等安防行为。
> **任务**：`pose_action`（别名 `action`） · **模型**：`STGCN`
> **权重**：**无官方权重**（本框架自实现，未做权重对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **ST-GCN：Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**（arXiv:1801.07455，AAAI 2018，Yan et al.） |
| 机构 | 香港中文大学（CUHK）等 |
| 官方代码 | https://github.com/yysijie/st-gcn |
| 官方权重 | **本框架未使用官方权重**（自研端口，配置与原文略有简化） |
| 本框架实现 | 全部在 `torchkiln/tasks/pose_action.py`（`STGCN`/`STGCNBlock`/`SpatialGraphConv`/数据集/损失/指标）<br>配置 `configs/action/pose_action.yml` |
| 移植方式 | **按论文思想自实现**（非逐键复刻），未做对齐验证 |

### 要解决的问题

- 视频 RGB 行为识别（见 `video_cls.md`）对光照/遮挡敏感，且安防场景常已有 **pose 输出**。
- **骨架**输入（每帧每个关节点的坐标）维度低、抗外观变化，**ST-GCN** 把人体建模为**图**：
  节点=关节点、边=骨骼连接，用**图卷积**在空间维聚合、用**1D 时序卷积**在时间维建模。
- 本框架的接口取向：**直接复用现有 YOLO-pose 能力** —— 先用 `pose` 出关键点序列存成 `.npy`，
  再用本任务训练行为分类器。

---

## 2. 网络结构

### 2.1 整体框图

```
Input  (N, C, T, V)                    # C=坐标维(默认2), T=帧数, V=关节数(默认17)
  │
  ▼ data_bn: permute→(N, V*C, T) → BatchNorm1d(V*C) → 回 (N, C, T, V)
  ▼ 9 × STGCNBlock
  │   ├─ SpatialGraphConv（3 个邻接子集：自环 / 向心 / 离心）
  │   │    h = conv_k(x) (1×1)  →  h @ A_k  →  按 (1 + alpha_k) 加权求和
  │   └─ TCN: BatchNorm → ReLU → Conv2d(9×1, 时间卷积) → BatchNorm → Dropout
  │        + 残差（identity 或 1×1 Conv+BN）
  ▼ AdaptiveAvgPool2d(1) → flatten → Linear(C, num_classes)
  输出 (N, num_classes) logits
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | 论文名字 |
|---|---|---|---|---|
| `data_bn` | `BatchNorm1d(V*C)` | `(N,C,T,V)→同形` | 输入归一 | `data_bn` |
| **`SpatialGraphConv`** | 每个子集一个 `Conv2d(1×1)` + `einsum('nctv,vw->nctw', h, A_k)`；`alpha_k` 可学习残差权 | `(N,C,T,V)→(N,C',T,V)` | ★ 空间图卷积 | Spatial Graph Conv |
| **`STGCNBlock`** | `gcn → BN → [BN→ReLU→Conv2d(9×1)→BN→Dropout]` + 残差 → ReLU | 同形 | ★ 时空块 | ST-GCN block |
| `A`（3 子集） | **自环 / 向心 / 离心** 归一化邻接矩阵，注册为 buffer | `(3,V,V)` | 图拓扑 | Partition |
| 池化/分类 | `AdaptiveAvgPool2d(1)` + `Linear` | →`(N,nc)` | | |

- 通道序列：`[64, 64, 64, 128, 128, 128, 256, 256, 256]`（9 个 block）。
- 默认拓扑为 **COCO 17 点**（`coco17_edges()`），支持自定义 `num_joints`（如车牌 4 角点）。
- `SpatialGraphConv` 用 `einsum` 聚合邻接，`alpha` 参数默认 0（即初始 `1+alpha=1`）。

### 2.3 与 video_cls 的差异

| 项 | video_cls（TSM） | **pose_action（ST-GCN）** |
|---|---|---|
| 输入 | `(N,T,3,H,W)` RGB 帧 | **`(N,C,T,V)` 骨架序列** |
| 空间建模 | 2D 卷积 | **人体图卷积** |
| 时序建模 | TemporalShift + 分段共识 | **1D 时序卷积** |
| 对光照/遮挡 | 敏感 | **不敏感** |
| 前置依赖 | 无 | **需先跑 pose 得到关键点** |

---

## 3. 输出与后处理（★ 重点）

### 3.1 数据格式（`PoseActionDataset`）

```
# label file（yolo 风格）：
path/to/seq_0001.npy  <action_id>

# .npy 形状：(T, V, C) 或 (C, T, V)
#   V 默认 17（COCO/YOLO-pose），C 默认 2（x, y）
```

- 若 `.npy` 是 `(C,T,V)`（C∈{2,3}），自动转成 `(T,V,C)`。
- 若只有 1 个关键点 `(T,C)`，扩成 `(T,1,C)`。

### 3.2 采样与增广

| 项 | 说明 |
|---|---|
| 时间裁剪 | 训练：随机窗口 `clip_len`；评估：`linspace` 均匀采样到 `clip_len` |
| 水平翻转 | `augment.flip`（默认 0.5）：`x` 取反 + **左右对称点对交换** `[(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]` |
| 坐标抖动 | `augment.jitter`（默认 0.01）：加均匀噪声 |
| 输出 | `(C, T, V)` float32 |

### 3.3 输出与指标

```
logits = model(data)                 # (N, num_classes)
prob   = softmax(logits, dim=1)
label  = argmax(prob); score = max(prob)
metric = PoseActionMetric → acc(top-1) / top5
loss   = CrossEntropyLoss(logits, label)   # label_smoothing 可配
```

---

## 4. 配置与用法

### 4.1 最小配置（`configs/action/pose_action.yml`）

```yaml
Architecture:
  task: pose_action
  algorithm: STGCN
  Backbone: {name: STGCN, num_joints: 17, in_channels: 2, dropout: 0.1}
  Head: {num_classes: 3}
Loss:  {name: PoseActionLoss, label_smoothing: 0.0}
Metric: {name: PoseActionMetric, main_indicator: acc, topk: 5}
PostProcess: {name: PoseActionPostProcess}
Train:
  dataset: {name: PoseActionDataset, data_dir: datasets/pose_action_demo,
            label_file_list: [.../train.txt],
            transform: {clip_len: 60, num_joints: 17},
            augment: {flip: 0.5, jitter: 0.01}}
  loader: {batch_size_per_card: 4, num_workers: 0}
```

### 4.2 数据准备（YOLO-pose → npy）

```bash
# 1) 用 pose 任务跑关键点（或用 tkiln predict 导出）
tkiln predict -c configs/yolo/yolo11-pose.yml --weights pose_ckpt.pth --input clips/
# 2) 每段视频得到一个 (T,17,2) 序列，存成 .npy
# 3) 写 label file：seq_0001.npy 0
```

### 4.3 四条链路

```bash
tkiln train   -c configs/action/pose_action.yml
tkiln val     -c configs/action/pose_action.yml --weights output/pose_action/best_accuracy.pth
tkiln predict -c configs/action/pose_action.yml --weights ... --input seqs/
tkiln check   -c configs/action/pose_action.yml
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 模型 | ST-GCN：9 个 block，通道 `[64,64,64,128,128,128,256,256,256]` |
| 参数量 | **未统计**（本框架未记录） |
| FLOPs / 本机耗时 / 显存 | **未统计** |

> 无 3D 卷积；主要开销在小矩阵乘（`A` 为 `V×V`，V=17）。

---

## 6. 公开指标

| 项 | 状态 |
|---|---|
| **官方权重对齐** | **无官方权重可对**（自研端口，配置简化） |
| **NTU-RGB+D / Kinetics-Skeleton 公开基准** | **未统计** |
| 设计目标 | 安防场景**自有骨架数据**的行为分类可用性 |

> ⚠️ **如实说明**：本端口按 ST-GCN 思想自实现（如仅用 3 分区邻接、9 个 block、
> 未加原文的 `edge importance` 加权等细节），**未经与官方权重的对齐验证**，
> 也没有公开基准成绩。评估质量需在自有数据上报告 top-1/top-5。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 已有 pose、要判行为（跌倒/打架） | **本任务（ST-GCN）** | 抗光照/遮挡、输入维度低 |
| 只有 RGB、无 pose | 见 `docs/models/task/video_cls.md` | TSM 直接吃视频 |
| 需要人体关键点本身 | 见 `docs/models/task/pose.md` | Pose |
| 自定义关键点（车牌 4 角等） | 本任务 + `num_joints=4` | 支持任意骨架 |
| 工业设备故障（非人体） | 用 `ts_classify` / `ts_anomaly` | 时序分类/异常 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs video_cls（TSM）** | 输入骨架 vs RGB；ST-GCN 抗外观变化 |
| **vs ST-GCN 原文** | 同思想；本端口用 3 分区邻接 + 9 block + 可学习 `alpha`，细节有简化 |
| **vs 2s-AGCN / CTR-GCN** | 更强的骨架模型（自适应图/通道拓扑），本端口未实现 |
| **vs Pose（检测头）** | Pose 出关键点；本任务把关键点序列分类 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **无官方权重、无对齐验证**：报告数值须注明"未与官方对齐"。
- ⚠️ **输入形状**：`(T,V,C)` 或 `(C,T,V)`；`V` 必须等于 `num_joints`，否则样本被丢弃（返回 `[]`）。
- ⚠️ **左右翻转必须交换对称点 且 `x` 取反**，否则增广产生错误语义。
- ⚠️ **COCO 17 点拓扑固定**；用别的骨架（4 点/自定义）要替换 `edges`（当前 `build_pose_action_model` 只透传 `num_joints`，拓扑仍用 COCO——**自定义骨架需改代码或加配置**）。
- ⚠️ **数据需自行从 pose 导出**（本框架不内建视频→npy 的批处理命令）。
- ⚠️ **评估 fp32**：`_evaluate_loop` 强制 `autocast(enabled=False)`。
