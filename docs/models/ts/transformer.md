# Transformer

> **定位**：标准自注意力编码器，建模任意距离依赖
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Attention Is All You Need |
| arXiv | 1706.03762 |
| 年份 | 2017 |
| 机构 | Google Brain / Google Research |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/transformer.py` |
| 本框架实现 | ``torchkiln/nn/s_transformer.py::Transformer`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

RNN 的顺序计算无法并行，且长距离依赖靠隐状态逐级传递会衰减。

### 核心创新

- **Self-attention**：任意两位置直接交互，路径长度 O(1)
- **多头**：不同子空间捕捉不同关系
- **位置编码**：补回顺序信息

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D) → Linear(d_model) + 位置编码
  └─ EncoderLayer × n:
       MultiHeadSelfAttn → Add&Norm → FFN → Add&Norm
  → Linear → (B,H,D)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_encoder` | `nn.TransformerEncoder` 包装 | 主体 |
| `_linear` | `Linear(d_model, H*D)` | 输出头 |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs Informer | 标准 O(L²) vs ProbSparse O(L log L) |
| vs TCN | 全局依赖 vs 局部卷积 |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输入 | `{"past_target": (B, L, D)}`（协变量模型另有 `known_cov_numeric` 等） |
| 输出 | `(B, out_chunk_len, target_dim)` |
| 解码 | **直接输出，无需后处理**（时序回归不同于检测，无 NMS/anchor） |
| 指标 | MSE / MAE |

> ⚠️ 与检测任务不同：**没有 anchor、没有 NMS、没有解码** —— 网络输出即预测值。

---

## 4. 配置与用法

```yaml
Architecture:
  model_family: ts_forecast
  task: ts_forecast
  Head:
    model: transformer
    in_chunk_len: 96
    out_chunk_len: 24
    target_dim: 1
Loss:
  type: mse
Metric:
  main_indicator: MSE
  pred_mode: point
```

```bash
tkiln train  -c configs/ts/transformer_demo.yml
tkiln val    -c configs/ts/transformer_demo.yml --weights output/.../best_accuracy.pth
tkiln predict -c configs/ts/transformer_demo.yml --input data.csv
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.003 M** |
| 本框架对齐 | **maxdiff=0（**逐位一致**）** |
| 训练速度 | 合成数据 30 epoch < 1 分钟（本机 RTX 4060 Ti） |
| FLOPs | ⚠️ **本框架未集成 FLOPs 统计** |

---

## 6. 公开指标

**对齐验证**（同权重同输入，PaddleTS vs 本框架）：

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0** |
| 逐层前向 | **maxdiff=0（**逐位一致**）** |

> ℹ️ PaddleTS **不发布官方预训练权重**，故无"公开指标"可对比；
> 本框架的验收标准是**与 PaddleTS 参考实现数值一致**。

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| 长依赖、序列不长 | ✅ |
| 长序列（>512） | 改用 Informer/SCINet |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| DLinear 论文 | 该论文质疑 Transformer 在时序上的必要性 |
| TFT | TFT 在 Transformer 上加了变量选择与可解释性 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **移植坑**：paddle `nn.Transformer` 用 `q_proj/k_proj/v_proj/out_proj`，
  torch 用 `in_proj_weight/in_proj_bias/out_proj` → **q/k/v 权重需拼接**
- ⚠️ **忽略协变量**
- ⚠️ O(L²) 复杂度，长序列显存吃紧
