# DLinear

> **定位**：**质疑 Transformer 的极简模型**：序列分解 + 单层线性，5K 参数打赢一堆 Transformer
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Are Transformers Effective for Time Series Forecasting? |
| arXiv | 2205.13504 |
| 年份 | 2022 |
| 机构 | （DLinear 为论文提出的方法） |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/—（PaddleTS 1.1.0 无此模型）` |
| 本框架实现 | ``torchkiln/nn/ts_models.py::DLinear` / `_MovingAvg` / `_SeriesDecomp`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

2022 年前后大量论文把 Transformer 塞进时序预测，但论文作者质疑：**self-attention 的置换不变性会丢失时序顺序**，而很多基准上简单线性模型效果相当甚至更好。

### 核心创新

- **序列分解**：`trend = MovingAvg(x)`，`season = x - trend`
- **各自单层 Linear 映射到 horizon**，再相加
- **极简**：5K 参数，训练/推理极快

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D)
  ├─ MovingAvg(k=25) → trend (B,L,D) → Linear → (B,H,D)
  └─ x - trend        → season(B,L,D) → Linear → (B,H,D)
  Σ → (B,H,D)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_MovingAvg` | `AvgPool1d(kernel, stride=1, padding)` | 提取趋势 |
| `_SeriesDecomp` | `x - MovingAvg(x)` | 提取季节 |
| 双 Linear | `(B,L,D)→(B,H,D)` | 分别映射 |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs Transformer | DLinear **无注意力**，参数量少 3 个数量级 |
| vs NBEATS | 都做分解，但 DLinear 用移动平均（无学习），NBEATS 用学习的基函数 |

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
    model: dlinear
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
| 参数 | **0.005 M** |
| 本框架对齐 | **⚠️ **不在 PaddleTS 1.1.0 中，无对齐基准**** |
| 训练速度 | 合成数据 30 epoch < 1 分钟（本机 RTX 4060 Ti） |
| FLOPs | ⚠️ **本框架未集成 FLOPs 统计** |

---

## 6. 公开指标

**对齐验证**（同权重同输入，PaddleTS vs 本框架）：

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0** |
| 逐层前向 | **⚠️ **不在 PaddleTS 1.1.0 中，无对齐基准**** |

> ℹ️ PaddleTS **不发布官方预训练权重**，故无"公开指标"可对比；
> 本框架的验收标准是**与 PaddleTS 参考实现数值一致**。

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| **快速基线** | ✅ 秒级训练 |
| 数据量小 | ✅ 参数少不易过拟合 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| Transformer 家族 | DLinear 论文的直接对标对象 |
| MLP | DLinear 多了分解步骤 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **PaddleTS 无此模型**，故本框架**无对齐基准**（属框架自带扩展）
- ⚠️ `kernel_size`（分解核）是关键超参，默认 25；周期长度应能被它覆盖
- ⚠️ 不支持协变量
