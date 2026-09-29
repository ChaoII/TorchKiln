# RNN (LSTM / GRU)

> **定位**：最经典的循环结构，作为对照与轻量方案
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | LSTM: Long Short-Term Memory (1997) / GRU: Learning Phrase Representations (2014) |
| arXiv | — |
| 年份 | 1997 / 2014 |
| 机构 | Hochreiter & Schmidhuber / Cho et al. |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/rnn.py` |
| 本框架实现 | ``torchkiln/nn/s_rnn.py::RNNBlock`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

前馈网络无法处理变长序列与长期依赖。

### 核心创新

- **门控机制**（遗忘/输入/输出门）解决梯度消失
- **GRU** 用重置/更新门简化 LSTM

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D) → RNN(units) → 取最后隐状态 → Linear → (B,H,D)
```

### 2.2 逐模块说明

| 模块 | 结构 |
|---|---|
| `_rnn` | `nn.LSTM/GRU(batch_first=True, num_layers)` |
| `_linear` | `Linear(hidden, H*target_dim)` |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs TCN | RNN 串行、感受野无界；TCN 并行、有限但可调 |
| vs Transformer | RNN O(L) 但难并行 |

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
    model: rnn
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
tkiln train  -c configs/ts/rnn_demo.yml
tkiln val    -c configs/ts/rnn_demo.yml --weights output/.../best_accuracy.pth
tkiln predict -c configs/ts/rnn_demo.yml --input data.csv
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **LSTM 0.070 / GRU 0.053 M** |
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
| 轻量/快速原型 | ✅ |
| RUL（数据少） | 本框架实测 FD001 34.33，**不如 TCN 25.44** |
| 需要协变量 | ❌ |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| TCN | 同期论文对比对象 |
| BiLSTM+Attn | 本框架新增的注意力升级版 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **Paddle `time_major=False`（batch-first），torch 默认 seq-first** ⇒
  必须显式 `batch_first=True`，否则语义完全错（曾报 `mat2 shapes 2x14 vs 4x4`）
- ⚠️ **忽略协变量**
- ⚠️ 层数 >1 时才有 `dropout` 生效
