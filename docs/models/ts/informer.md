# Informer

> **定位**：长序列预测的**稀疏注意力**方案（ProbSparse），但含算法固有随机性
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting |
| arXiv | 2012.07436 |
| 年份 | 2020 |
| 机构 | （Informer 为该论文提出） |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/informer.py` |
| 本框架实现 | ``torchkiln/nn/s_informer.py::Informer`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

标准 Transformer 在长序列上 O(L²) 注意力太贵，且解码是**自回归**（逐步生成慢）。

### 核心创新

- **ProbSparse attention**：只选 **top-u 个「活跃」query** 算注意力 → **O(L log L)**
- **Self-attention distilling**：逐层减半序列长度
- **生成式解码**：一次输出整段 horizon（**非自回归**）

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D) → ProbSparse Encoder × n（含 distilling）
  └─ Decoder: 已知段 + 占位段 → 注意力 → 一次输出全部 horizon
  → (B,H,D)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_ProbSparseAttention` | top-u 采样 + 稀疏 softmax | 核心 |
| `_EncoderStack` | ConvLayer + distilling | 逐层减半 |
| `_Decoder` | 生成式解码 | 一次出 H |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs Transformer | O(L log L) vs O(L²)；生成式 vs 自回归 |
| vs SCINet | 稀疏注意力 vs 采样卷积 |

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
    model: informer
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
tkiln train  -c configs/ts/informer_demo.yml
tkiln val    -c configs/ts/informer_demo.yml --weights output/.../best_accuracy.pth
tkiln predict -c configs/ts/informer_demo.yml --input data.csv
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **11.301 M** |
| 本框架对齐 | **⚠️ **前向形状/结构正确，但不可逐位复现**** |
| 训练速度 | 合成数据 30 epoch < 1 分钟（本机 RTX 4060 Ti） |
| FLOPs | ⚠️ **本框架未集成 FLOPs 统计** |

---

## 6. 公开指标

**对齐验证**（同权重同输入，PaddleTS vs 本框架）：

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0** |
| 逐层前向 | **⚠️ **前向形状/结构正确，但不可逐位复现**** |

> ℹ️ PaddleTS **不发布官方预训练权重**，故无"公开指标"可对比；
> 本框架的验收标准是**与 PaddleTS 参考实现数值一致**。

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| 长序列（>512） | ✅ 理论上最优 |
| **需要可复现结果** | ⚠️ 慎用（含随机采样） |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| Autoformer/FEDformer | 同期长序列方案 |
| SCINet/NHiTS | 更简单、6~10× 参数更少 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`ProbSparseAttention` 用 `torch.randint` 随机选 key**，
  与 `paddle.randint` **无法跨框架复现** ⇒ **算法固有随机性，非实现 bug**
- ⚠️ 参数 **11.3M**，是其它 ts 模型的 10~3000 倍
- ⚠️ decoder padding 需 `device=src.device`（否则 GPU 训练 device mismatch）
