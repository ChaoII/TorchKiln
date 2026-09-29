# AutoEncoder（时序异常检测）

> **定位**：最基础的异常检测：**重建误差**大即为异常
> **任务**：`ts_anomaly` | **权重**：无需预训练（paddlets 无官方权重）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | （AutoEncoder 异常检测为通用方法，非单一论文） |
| arXiv | — |
| 年份 | — |
| 上游实现 | paddlets `models/anomaly/dl/autoencoder.py::AutoEncoder` + `_ed/ed.py::MLP/CNN` |
| 本框架实现 | `torchkiln/nn/s_anomaly.py::AEBlock + torchkiln/tasks/ts_anomaly.py` |
| 对齐方式 | 同起点 dump Paddle 权重 → torch 加载 → 逐层前向 + **fp64 判定** |

### 要解决的问题

工业设备的**正常样本远多于故障样本**，监督学习无从下手。无监督思路：正常数据能被网络很好地重建，异常数据重建得差。

### 核心创新

- **编解码结构**：压缩到隐空间再重建，迫使网络学习「正常模式」的流形
- **重建误差即异常分数**（无需标签）
- `ed_type` 可选 **MLP**（快）或 **CNN**（保留局部时序结构）

---

## 2. 网络结构

```
Input (B, L, C)  →  transpose (B, C, L)
  └─ _encoder:  MLP/CNN: (C,L) → (hidden...) → (h,)
  └─ _decoder:  MLP/CNN: (h,) → (hidden...) → (C,L)
  → transpose → recon (B, L, C)
```

| 模块 | 结构 | 作用 |
|---|---|---|
| `MLP` | Linear → BN1d → ReLU → Dropout 堆叠 | MLP 编解码 |
| `CNN` | Conv1d → BN1d → ReLU → Dropout 堆叠 | CNN 编解码 |
| `AEBlock` | `_encoder` + `_decoder` | 组装 |

### 与 VAE / USAD 的差异
| 对比 | 差异 |
|---|---|
| vs VAE | AE 无隐空间概率建模（无 KL） |
| vs USAD | USAD 用**对抗双解码器**，更鲁棒 |
| vs AnomalyTransformer | AT 用**关联差异**而非重建误差 |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `(recon, x)`，两者均为 `(B, L, C)` |
| 异常分数 | `mean((recon-x)², dim=(1,2))` → `(B,)` |
| 阈值 | `Metric.threshold_percentile`（**按异常占比设**） |
| 指标 | 点级 P/R/F1（**point-adjust**）+ AUC-ROC + AUC-PR |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
Architecture:
  task: ts_anomaly
  algorithm: ae
  in_chunk_len: 64
  num_features: 3
  ed_type: MLP
  hidden_config: [32, 16]
  use_bn: false
  dropout_rate: 0.2
Metric:
  main_indicator: f1
  threshold_percentile: 75.0
  point_adjust: true
```

```bash
tkiln train -c configs/ts/anomaly_ae_demo.yml -o Global.epoch_num=20
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.0053 M（MLP 版）** |
| 对齐 | **maxdiff **1.3e-16**（fp64，逐位级）** |
| 本机速度 | 合成数据 20 epoch < 30 秒 |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0**（8 键） |
| fp64 前向 | recon **1.3e-16** |
| 端到端 AUC-ROC（demo） | **0.769**；F1 ≈1.0 |

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| **快速起步/基线** | ✅ 首选 |
| 时序模式复杂 | 改用 MTAD-GAT / AnomalyTransformer |
| 需要鲁棒性 | 改用 USAD |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| AE vs VAE | VAE 多 KL 正则，隐空间更规整 |
| AE vs USAD | USAD 用两个解码器对抗训练 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **阈值分位必须按实际异常占比设**：demo 异常 26% 用 q=75；工业场景异常少（<1%）应用 q=99
- ⚠️ CNN 版要求每层池化后长度 ≥1，`L=32` 时 k=7 会不足（用 L≥48）
- ⚠️ paddlets `_ed.MLP` 的 `feature_dim` 语义是 dim1 通道
