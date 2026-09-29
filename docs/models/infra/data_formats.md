# 数据格式与 YAML 配置

> **定位**：每个任务吃什么数据、配置怎么写
> **分类**：infra

---

## 配置的三段结构

```yaml
Global:        # 运行级（device / epochs / AMP / EMA / best 方向）
Architecture:  # 模型级（task / family / scale / Head 超参）
Optimizer:     # 优化器（含 clip_grad_norm）
Loss / Metric: # 任务级
Train / Eval:  # 数据与 loader
```

**关键**：`Architecture.task` 决定路由到哪个 `TaskAdapter`。
---

## 各任务数据格式

| 任务 | 数据 | 标签 | 说明 |
|---|---|---|---|
| `detect` / `segment` / `pose` | 图片 + `.txt` | YOLO 格式 | `cls cx cy w h`（归一化） |
| `obb` | 图片 + `.txt` | DOTA 四角点 | 8 个归一化数 |
| `classify` | 图片目录 | 目录名 | `ClsDataset` |
| `semantic` / `depth` | 图片 + mask / 深度 | PNG | depth 是 uint16 毫米 |
| `plate_det` / `plate_rec` | 图片 + 标注 | 车牌框 / 文本 | |
| `ocr_e2e` | 图片 + 多边形 | 文本行 | PGNet 格式 |
| `pc_seg` | 点云 `.npy` | 逐点标签 | range-view 投影 |
| `det3d` | 点云 `.bin` + `.npy` | KITTI 标签 | raw_points 模式 |
| `lane_bev` | 图 + npz | BEV GT | 6 个键 |
| **`ts_forecast`** | **CSV** | 滑窗 | target / known / observed / static 四类列 |
| **`ts_anomaly`** | **CSV** | 点级 label | `label` 列可缺 |
| **`ts_classify`** | **CSV / 清单** | 序列级 | 窗内多数投票 |
| **`ts_rul`** | **C-MAPSS / CSV** | 分段 RUL | 按 unit 划分 |
---

## TS 数据集的列角色（重点）

| 列角色 | 配置键 | 说明 |
|---|---|---|
| 目标 | `target_cols` | 要预测的列 |
| **已知协变量** | `known_cols` | **past + future 全窗**（如天气预报、日历） |
| **观测协变量** | `observed_cols` | **仅 past**（如实测气象） |
| 静态 | `static_cols` | 每序列常数（如电站 ID） |

⚠️ **多数 ts 模型（RNN / LSTNet / Transformer / SCINet）会忽略协变量**；
需要协变量必须用 **TFT / DeepAR / NBEATS**。
---

## loader 配置位置（易错）

```yaml
Train:
  dataset: {...}
  loader:            # ← 在这里，不是 dataset.loader
    batch_size_per_card: 4
    num_workers: 4
```

⚠️ 历史上有配置写在 `Train.dataset.loader`，框架已兼容两者，
但**梯度累积 `accumulate = nbs/batch` 依赖它**。
---

## 增广开关（易错）

```yaml
augment: {}          # ← 空 dict 是 falsy！不会建 augmenter
augment:             # ← 必须显式写值
  mosaic: 1.0
  hsv: 0.015
  affine: 1.0
  fliplr: 0.5
  close_mosaic: 10
```
---

## device 写法（易错）

```yaml
device: cuda:0    # ✅ 正确
device: '0'       # ❌ 会按非 gpu/cuda 前缀解析到 CPU
```
