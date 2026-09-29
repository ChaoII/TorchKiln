# Anomaly Transformer（时序异常检测）

> **定位**：**关联差异**（Association Discrepancy）：用「注意力分布与高斯先验的 KL」作为异常判据
> **任务**：`ts_anomaly` | **权重**：无需预训练

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy |
| arXiv | 2110.02642 |
| 年份 | 2021 |
| 上游实现 | paddlets `models/anomaly/dl/anomaly_transformer.py` + `_anomaly_transformer/{attention,encoder,embedding}.py` |
| 本框架实现 | `torchkiln/nn/s_anomaly_transformer.py::AnomalyTransformerNet` |
| 对齐方式 | 同起点 dump → torch 加载 → fp64 判定（47 键 missing=0） |

### 要解决的问题

重建类方法把异常当「重建难」，但异常有时也能被重建好；且逐点判据忽略全局关联结构。

### 核心创新

⭐ **Association Discrepancy（关联差异）**：
- **prior-association**：可学习高斯核 `1/(√(2π)σ)·exp(-dist²/(2σ²))`
- **series-association**：从数据学的自注意力
- **关联差异 = 两者 KL**：正常点两者相近，异常点差异大
- **训练 minimax**：`loss1 = rec - k·series_KL`（放大差异）/ `loss2 = rec + k·prior_KL`（收缩正常差异）
- **异常分数** = `softmax(-series_loss - prior_loss) × rec_loss`

---

## 2. 网络结构

```
Input (B,L,C)
  └─ DataEmbedding: TokenEmbedding(circular Conv1d k=3) + PositionalEmbedding
  └─ EncoderLayer × e_layers:
       AnomalyAttention:
         · scores = QKᵀ → series = softmax(scale·scores)
         · sigma = sigmoid(sigma_proj(x)·5)+1e-5 → 3^sigma
         · prior = 高斯核(distances, sigma)
         · V = series @ values
       + Add&Norm + FFN(Conv1d 1×1) + Add&Norm
  → Linear → (B,L,C) + [series, prior, sigma]×e_layers
```

| 模块 | 结构 | 作用 |
|---|---|---|
| `TokenEmbedding` | **Conv1d(k=3, padding_mode=circular)** | 局部编码 |
| `PositionalEmbedding` | sin/cos | 位置 |
| `AnomalyAttention` | QKV 投影 + **Gaussian prior 核** | 核心 |
| `AttentionLayer` | query/key/value/sigma_projection + 多头 reshape | 组装 |
| `Encoder` | n 层 + LayerNorm | 堆叠 |

⚠️ **`distances` 是普通属性**（Paddle 里不进 state_dict），torch 用非持久化张量。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `(enc_out, series_list, prior_list, sigma_list)` |
| 异常分数 | `series_prior_energy`：`softmax(-series-prior)×rec_loss` |
| ⚠️ 口径 | paddlets 对**批内**做 softmax ⇒ **分数随 batch 组成变化**；框架保留该口径并提供 `pointwise` |
| 训练 | **minimax 两阶段**（`train_step` 钩子） |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
Architecture:
  task: ts_anomaly
  algorithm: anomaly_transformer
  in_chunk_len: 64
  d_model: 32
  n_heads: 4
  e_layers: 2
  d_ff: 32
  dropout: 0.0
Loss:
  k: 3
PostProcess:
  at_temperature: 50
  at_score_mode: paddlets
```

```bash
tkiln train -c configs/ts/anomaly_at_demo.yml -o Global.epoch_num=15
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.0138 M** |
| 对齐 | **out **1.5e-13** / series 3.4e-16 / prior 2.1e-16 / sigma 3.0e-16（fp64）** |
| 本机速度 | 最慢的异常检测模型（minimax 两阶段 + 多头注意力） |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0**（47 键） |
| fp64 前向 | out 1.5e-13，其余 1e-16 级 |
| 端到端 AUC-ROC | **0.770**；F1 0.996 |

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| 异常与**上下文关联突变**相关 | ✅ 首选 |
| 需要简单方案 | 用 AE/USAD |
| 批次大小会变 | ⚠️ 注意批内 softmax 口径 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| vs AE/VAE | AT 不靠重建误差，靠关联差异 |
| vs MTAD-GAT | 都显式建模关联，AT 用高斯先验，MTAD-GAT 用图 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **批内 softmax 口径**：`softmax(-series-prior, dim=-1)` 跨 batch 维 ⇒ 同一窗口在不同 batch 下分数不同；部署建议 `at_score_mode='pointwise'`
- ⚠️ **训练是 minimax**，需两次 backward+step（`train_step` 钩子）
- ⚠️ `my_kl_loss` 用 `+1e-4` 而非 `eps`（对齐 paddlets）
- ⚠️ `distances` 不进 state_dict
