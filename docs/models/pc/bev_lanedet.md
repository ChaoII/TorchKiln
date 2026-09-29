# BEV-LaneDet

> **定位**：**单目 3D 车道线检测**；用 ResNet34 提图特征，再用 **FCTransform**
> 把「图像特征图」reshape 成「BEV 空间特征图」（全连接式视图变换，不显式建相机几何）。
> **任务**：`lane_bev`（`Architecture.task: lane_bev`）
> **权重**：`bev_lanedet_apollo_576x1024.pth`（`weights/`，ModelScope `ChaoII0987/TorchKiln → pretrained/`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **BEV-LaneDet: a Simple and Effective 3D Lane Detection Baseline** |
| arXiv | **2210.06006**（2022-10-12，已核实标题/作者） |
| 作者 | Ruihao Wang、Jian Qin、Kaiying Li、Yaochen Li、Dong Cao、Jintao Xu |
| 会议 | CVPR 2023 |
| 官方代码 | https://github.com/gigo-team/bev_lane_det |
| 本框架参考实现 | **Paddle3D**（Apollo 3D Lane 训练口径） |
| 本框架实现 | `torchkiln/nn/bev_lanedet.py`（网络）、`torchkiln/lane_bev.py`（loss/后处理/指标）、<br>`torchkiln/data/lane_bev.py`（数据）、`torchkiln/tasks/lane_bev.py`、`torchkiln/models/lane_bev.py` |
| 移植方式 | PyTorch 逐命名移植 Paddle3D + 权重逐键对齐（**missing=0 / unexpected=0，372 张量**） |

### 论文要解决的问题

1. **透视→BEV 的视角变换是难点**：单目图像没有深度，传统的 IPM（逆透视变换）要求
   平坦地面假设，坡道/颠簸就会失真。
2. **车道线是细长结构**，像素级分割后还需要实例化（聚类/后处理）；
   而 3D 车道还需要给出高度 z。

BEV-LaneDet 的核心创新：
- **FCTransform（虚拟相机 + 全连接视图变换）**：不走显式相机矩阵，而是把
  `s32`（stride 32 图像特征）与 `s64`（降采样更深的特征）分别 **reshape 成一维**，
  用 **两个全连接层 + ReLU** 映射到 BEV 空间尺寸，再 `conv1 + residual`。
  等价于「**可学习**的视图变换」，且是**端到端可微**的；
- **双分支（Key-Point 表示 + 实例嵌入）**：`seg` 出车道线前景，`emb` 出实例嵌入，
  推理时用 **embedding 聚类** 得到不同车道实例；
- **辅助 2D 头**：训练时额外在图像域做分割（`seg2d/emb2d`），提供稠密监督。

### 关键结论

| 数据集 | 指标 | 原文 |
|---|---|---|
| Apollo 3D Lane（合成+真实） | **F-score** | BEV-LaneDet 在 Apollo/OpenLane 上 **F1 ≈ 0.9+**（论文主结果） |
| 相对基线 | 提升 | 相比 3D-LaneNet / Gen-LaneNet 有显著提升 |

> ⚠️ 原文的具体 F1 数字随划分（standard/rare_subset/illus_chg）不同，
> 本仓库**没有**把论文数字记录在案，故此处写作「≈0.9+（以论文为准）」。

---

## 2. 网络结构

### 2.1 整体框图（**输入必须 576×1024**）

```
图像 (B, 3, 576, 1024)
  │
  ▼ bb = ResNet34 (bb.0..bb.7)                    → s32 特征 (B, 512, 18, 32)
  ├─ s32transformer = _FCTransform((512,18,32) → (256,25,5))
  │      全连接展开 18×32=576 → 25×5=125 → 再 125→125 → reshape → conv1×1 + residual
  │      → bev32 (B, 256, 25, 5)
  │
  └─ down = _Residual(Conv512→1024 s2 + Conv1024→1024)   → (B, 1024, 9, 16)
        s64transformer = _FCTransform((1024,9,16) → (256,25,5))
        → bev64 (B, 256, 25, 5)

bev = cat([bev64, bev32], dim=1)                 → (B, 512, 25, 5)
  ▼ lane_head = _LaneHeadWithOffsetZ(bev_shape=(200,48), input_channel=512)
      bev_up_new: Upsample ×2 → ResBlock → Upsample(size=(200,48)) → ResBlock
      head = _InstanceEmbeddingOffsetYZ(64, co=2)
        ├─ ms_new    → seg      (B, 1,   200, 48)   # 车道线前景 logits
        ├─ me_new    → emb      (B, 2,   200, 48)   # 实例嵌入
        ├─ m_offset_new → offset_y (B, 1, 200, 48)   # 横向偏移
        └─ m_z       → z        (B, 1,   200, 48)   # 高度

（训练时额外）lane_head_2d = _LaneHead2D((144,256))  → seg2d, emb2d
输出（训练）: [seg, emb, offset_y, z, seg2d, emb2d]   # 6 项
输出（推理）: [seg, emb, offset_y, z]                 # 4 项
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 |
|---|---|---|---|
| `_resnet34_children` | 标准 ResNet34（`conv1/bn1/relu/maxpool/layer1..4`） | (B,3,576,1024)→(B,512,18,32) | 骨干 |
| `_Residual` | `module(x) + downsample(x)` → ReLU | — | 残差块 |
| **`_FCTransform`** | `reshape(ic,ih·iw)` → `Linear(ih·iw → sh·sw)`→ReLU→`Linear`→ReLU → reshape → `conv1`(1×1) → `_Residual(3×3)` | (B,ic,ih,iw)→(B,sc,sh,sw) | ★ 视图变换 |
| `_LaneHeadWithOffsetZ` | 2 级 up + res（`bev_up_new`）+ `_InstanceEmbeddingOffsetYZ` | (B,512,25,5)→4 张量 | BEV 检测头 |
| `_InstanceEmbeddingOffsetYZ` | `neck_new`(2 级 3×3) 共享 → 4 个独立分支（ms/me/m_offset/m_z） | — | 多任务头 |
| `_LaneHead2D` | 3 级 up + res + `_InstanceEmbedding` | s32 特征→(seg2d, emb2d) | 2D 辅助头 |

### 2.3 与同类的差异

| 对比对象 | 差异点 | 为什么 | 效果 |
|---|---|---|---|
| **3D-LaneNet / Gen-LaneNet** | 它们用 **IPM/相机矩阵** 显式变换 | 平坦地面假设弱 | 本模型可学习变换更鲁棒 |
| **传统 2D 车道（UFLD 等）** | 只出 2D 曲线 | 无 z、无 BEV | 本模型给 3D 车道 |
| **BEVFormer / LSS** | 显式深度分布 + 采样 | 更重、需相机内外参 | FCTransform 极简、无参相机模型 |
| **Paddle3D 实现** | 特征尺寸**硬编码** | 对齐权重 | **输入必须 576×1024** |

---

## 3. 输出与后处理

### 3.1 输出张量

| 键 | 形状 | 含义 |
|---|---|---|
| `seg` | `(B, 1, 200, 48)` | BEV 车道线**前景 logits**（未 sigmoid） |
| `emb` | `(B, 2, 200, 48)` | **实例嵌入**（用于聚类区分车道） |
| `offset_y` | `(B, 1, 200, 48)` | 车道线**横向（y）偏移** |
| `z` | `(B, 1, 200, 48)` | 车道线**高度** |
| `seg2d`/`emb2d` | `(B, 1 / 2, 144, 256)` | 训练时 2D 辅助输出 |

- BEV 网格几何：`x_range=[3,103]`、`y_range=[-12,12]`、`meter_per_pixel=0.5`
  → **`bev_shape=[200,48]`**（官方配置）。
- 2D 输出尺寸 `[144,256]`。

### 3.2 损失（`BEVLaneDetLoss`）

```
loss_seg  = BCEWithLogits(pos_weight=10)(seg, gt_seg) + IoU(seg.sigmoid(), gt_seg)
loss_emb  = PushPull(emb, gt_inst)          # pull: 同类向心; push: 异类推远 (margin 1.0 / 5.0)
loss_off  = BCE(gt_seg * offset.sigmoid(), gt_off)
loss_z    = MSE(gt_seg * z, gt_z)
total     = 3*loss_seg + 0.5*loss_emb + 60*loss_off + 30*loss_z
            + 3*loss_seg2d + 0.5*loss_emb2d        # 2D 辅助
```

- `push_pull=true`（默认）才建 `_NDPushPullLoss`；`ignore_label=200`。
- GT 来自 Paddle3D `ApolloOffsetDataset.get_seg_offset`，键名映射：
  `bev_gt_segment→bev_seg`、`bev_gt_instance→bev_inst`、`bev_gt_offset→bev_off`、
  `bev_gt_z→bev_z`、`image_gt_segment→img_seg`、`image_gt_instance→img_inst`。

### 3.3 后处理（⚠️ 两套口径）

| | 框架 `BEVLaneDetPostProcess` | Paddle3D 官方 `PostProcessDataset` |
|---|---|---|
| 阈值 | `sigmoid(seg) > score_thres`（0.5） | `post_conf = 0.9` |
| 实例化 | **无**（纯像素二值） | **embedding 聚类**（`post_emb_margin=6.0`、`post_min_cluster_size=15`） |
| 拓扑 | 无 | **min-cost-flow** 连通成完整车道 |
| 输出 | 像素掩码 | BEV 车道实例 → 投影回 3D |

> ⚠️ **框架的 `FScore` 是像素级二值分割 F1**，与 Paddle3D `ApolloLaneMetric.f1_score`
> （**实例级 3D 车道 F1**）**不可直接比较**。要与官方同指标，必须把输出导出为
> `(1, 5, 200, 48)` 的 np（`concat([seg(raw), emb(2ch), sigmoid(offset), z])`）
> 再喂给 `ApolloLaneMetric`。

---

## 4. 配置与用法

```yaml
Architecture:
  model_family: lane_bev
  task: lane_bev
  train: true                    # true 才建 2D 辅助头（训练用）
  Head: {bev_shape: [200, 48], output_2d_shape: [144, 256]}
Loss:  {name: BEVLaneDetLoss, ignore_index: 255, push_pull: true}
Metric: {name: BEVLaneDetMetric, main_indicator: FScore}    # 像素级！
PostProcess: {name: BEVLaneDetPostProcess, score_thres: 0.5}
Train:
  dataset:
    name: LaneBEVDataset
    data_dir: datasets/lane_bev_demo
    label_file_list: [datasets/lane_bev_demo/train.txt]
    input_shape: [576, 1024]     # ★ 必须这个尺寸
    gt_dir: bev_gt
    normalize: none              # 真数据用 imagenet/paddle
```

真实 Apollo 配置见 `configs/local/apollo_bev_lanedet.yml`
（batch=4、epochs=50、AdamW+Cosine、`normalize: imagenet`）。

```bash
tkiln check -c configs/lane/bev_lanedet.yml
tkiln train -c configs/local/apollo_bev_lanedet.yml \
    -o Global.pretrained_model=bev_lanedet_apollo_576x1024
tkiln val   -c configs/lane/bev_lanedet.yml --weights output/bev-lanedet/best_accuracy.pth
```

数据目录：

```
<data_dir>/images/xxx.jpg                 # 虚拟相机 warpPerspective 后的 1920×1080
<data_dir>/bev_gt/xxx.npz                 # bev_seg/bev_inst/bev_off/bev_z/img_seg/img_inst
```

---

## 5. 规模与速度

| 项 | 值 | 来源 |
|---|---|---|
| 参数（`train=True`） | **43.9970 M** | 本框架实算 |
| 参数（`train=False`，推理） | **43.1587 M** | 本框架实算 |
| ResNet34 裸骨干 | **21.30 M**（`resnet34-remapped.pdparams` 81.27MB） | AGENTS 记录 |
| 权重体积 | **168 MB**（`weights/bev_lanedet_apollo_576x1024.pth`） | AGENTS 记录 |
| 输入尺寸 | **必须 576×1024** | 代码硬编码 |
| 训练显存 | 2 epoch 冒烟可跑（batch=4）；**未做系统统计** | AGENTS 记录 |
| 推理耗时 | **未统计** | — |

---

---

---

---

---

### 📊 FLOPs（实测）

| 项 | 值 |
|---|---|
| **FLOPs** | **109.04 GFLOPs** |
| **MACs** | **54.52 GMACs** |
| 参数量 | **43.159 M** |
| 输入规格 | `576x1024` |
| 测量工具 | `torch.utils.flop_counter.FlopCounterMode`（PyTorch 内置） |
| 复现脚本 | `_downloads/flops_measure*.py` |

> **口径**：`FLOPs` 是乘加各计 1 次（×2），**与 ultralytics 官方表的 GFLOPs 同口径**
> （已由 yolo11/v8 十个模型逐个吻合验证，见 [`_FLOPS.md`](_FLOPS.md)）；
> `MACs = FLOPs / 2`。
> ⚠️ `FlopCounterMode` **不计自定义算子**（NMS / probiou / iSTFT 等后处理）⇒
> 此处是**网络主干**的 FLOPs。
## 7. 选型建议

| 场景 | 建议 | 理由 |
|---|---|---|
| 单目 3D 车道线 | **本模型** | 无需相机内外参，可学习视图变换 |
| 只需 2D 车道线 | `lane_row` / `lane_seg`（见 `lane_row_seg.md`） | 更轻、更容易训 |
| 需要车道拓扑/实例 | 需补 **embedding 聚类 + min-cost-flow** | 框架后处理只有像素二值 |
| 非 576×1024 输入 | 需改 `FCTransform` 硬编码尺寸并**重训** | 特征尺寸写死 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs 3D-LaneNet / Gen-LaneNet** | 后者 IPM 显式变换；本模型 FCTransform 可学习 |
| **vs UFLD（row）** | UFLD 是「每行分类 x bin」的 2D 方法；本模型是 3D BEV |
| **vs Paddle3D 原版** | 权重/前向/损失对齐（1e-5~1e-6）；f1 框架 0.8378 vs Paddle 0.7776（2ep 噪声） |
| **框架像素 FScore vs 官方实例 f1** | **口径不同，不可比**（见 §3.3）；且 Paddle 用 BGR、框架用 RGB |

---

## 9. 已知问题 / 注意事项

- ⚠️ **输入必须 576×1024**：`_FCTransform` 的特征图尺寸 `(512,18,32)` / `(1024,9,16)`
  是**硬编码**的；换输入尺寸会直接 shape 不符（且权重失效）。
- ⚠️ **权威指标必须用同一个评估器**：框架 `BEVLaneDetMetric` 是像素级 F1，
  **不要**拿它和 Paddle3D 的 `f1_score` 比（口径不同 → 数字不可比）。
- ⚠️ **Paddle 侧训练口径 bug（历史）**：官方 1.0.0 缺 `main.py`，`batch_size` 不生效
  → batch=1 + LR 不衰减；对比时务必 `Trainer(..., dataloader_fn={'batch_size': cfg.batch_size})`。
- ⚠️ **`resnet34-remapped.pdparams` 不在 `data_splits.zip` 里**（那是纯 JSON 划分），
  需从 `https://bj.bcebos.com/paddle3d/models/bev_lanedet/resnet34-remapped.pdparams` 下。
- ⚠️ **预处理信道顺序**：Paddle3D 用 BGR（`A.Normalize`），框架转 RGB；若要进一步对齐需统一。
- ⚠️ **框架用 `best_accuracy.pth`**（按像素 F1 挑）评，Paddle 用 `epoch_2`（最后）；
  严格对比应统一 checkpoint 口径。
- ⚠️ 本框架**未集成 FLOPs 统计**。
