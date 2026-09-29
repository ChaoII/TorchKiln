# MTAD-GAT（时序异常检测）

> **定位**：**双图注意力**（特征图 + 时间图）+ GRU 的多任务异常检测
> **任务**：`ts_anomaly` | **权重**：无需预训练

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Multivariate Time-series Anomaly Detection via Graph Attention Network (MTAD-GAT) |
| arXiv | 2009.02040 |
| 年份 | 2020 |
| 上游实现 | paddlets `models/anomaly/dl/mtad_gat.py` + `_mtad_gat/{attention,layer,model}.py` |
| 本框架实现 | `torchkiln/nn/s_anomaly_gat.py（ConvLayer/GRULayer/FeatOrTempAttention/MTADGATBlock）` |
| 对齐方式 | 同起点 dump → torch 加载 → fp64 判定 |

### 要解决的问题

多变量时序里，**变量之间**与**时刻之间**都有依赖；普通 RNN 只能隐式建模。

### 核心创新

- **特征图注意力**：把每个传感器当作图的节点，学传感器间关系
- **时间图注意力**：把每个时刻当作节点，学时刻间关系
- **GRU** 融合两条图 + 原始特征（`3C` 通道）
- **双任务**：同时做**预测**+**重建**；异常分数 = `√(pred-true)² + √(recon-true)²`

---

## 2. 网络结构

```
Input (B, L, C)
  └─ ConvLayer(k=7, 零填充)  → 平滑
  ├─ _feature_gat  (特征图注意力) → (B, L, C)
  ├─ _temporal_gat (时间图注意力) → (B, L, C)
  └─ concat([x, h_feat, h_temp]) → (B, L, 3C)
  └─ GRU → 末隐状态 h_end
  ├─ _forec_model(h_end) → preds (B,1,C)
  └─ _recon_model(h_end) → recons (B,L-1,C)
```

| 模块 | 结构 | 作用 |
|---|---|---|
| `ConvLayer` | 零填充 + Conv1d(k) + ReLU | 平滑 |
| `FeatOrTempAttention` | **GATv2**（`_lin` 在 concat 后 + LeakyReLU + `_att`） | 图注意力 |
| `GRULayer` | GRU（**batch_first**） | 时序融合 |
| `Forecasting` | FC 堆叠 | 预测头 |
| `Reconstruction` | GRU 解码 | 重建头 |

⚠️ **两个易错点**：
1. **`name='temporal'` 时先交换 `feature_dim`/`in_chunk_len`**（Paddle 原样，勿擅改）
2. **`_bias` 在 Paddle 是 `Assign(paddle.empty(...))`（未初始化内存）** ⇒ 对拍必须以 dump 值为准

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `(preds, recons)` |
| 异常分数 | 逐通道 `√(pred-y)² + √(recon_last-y)²` 后取均值 |
| 损失 | `√MSE(y,pred) + √MSE(x,recons)` |
| 指标 | 同 AE |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
Architecture:
  task: ts_anomaly
  algorithm: mtad_gat
  kernel_size: 7
  gru_n_layers: 1
  gru_hid_size: 32
  use_gatv2: true
  use_bias: true
  alpha: 0.2
```

```bash
tkiln train -c configs/ts/anomaly_mtad_gat_demo.yml -o Global.epoch_num=20
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.0319 M** |
| 对齐 | **preds **1.2e-10** / recons **3.8e-11**（fp64）** |
| 本机速度 | 比 AE 慢（两个 GAT + GRU） |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0**（24 键） |
| fp64 前向 | preds 1.16e-10 / recons 3.79e-11 |
| 端到端 AUC-ROC | **0.769** |

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| **传感器网络**（多变量有结构） | ✅ 首选 |
| 单变量 | 用 AE（GAT 无意义） |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| vs AE/VAE | MTAD-GAT 用注意力而非纯重建 |
| vs AnomalyTransformer | AT 用关联差异，MTAD-GAT 用图注意力 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`name='temporal'` 会交换 `feature_dim`/`in_chunk_len`**（Paddle 原逻辑）
- ⚠️ **`_bias` 是未初始化内存**（Paddle `Assign(empty)`）
- ⚠️ **`GRULayer` 必须 `batch_first=True`**（Paddle `time_major=False` 默认）
- ⚠️ `in_chunk_len` 必须 ≥2
