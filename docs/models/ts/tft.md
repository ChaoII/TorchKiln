# TFT

> **定位**：⭐ **功率预测首选**：可解释 + 多 horizon + 分位数 + 最强协变量建模
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting |
| arXiv | 1912.09363 |
| 年份 | 2019 |
| 机构 | Google Cloud AI |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/tft.py` |
| 本框架实现 | ``torchkiln/nn/s_tft.py::TemporalFusionTransformer`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

多变量多 horizon 预测里：① 哪些变量重要？② 历史哪些时刻重要？③ 如何同时用上「已知未来协变量」和「静态协变量」？

### 核心创新

- **Variable Selection Networks**：自动给每个输入变量**重要性权重**（可解释）
- **Gated Residual Network (GRN)**：自适应地跳过无用层
- **Interpretable Multi-Head Attention**：注意力权重可直接看「关注了哪段历史」
- **Quantile 输出**：直接出 0.1/0.5/0.9 分位数（**PICP/MPIW 可算**）
- **三类协变量**（known/observed/static）分别编码

---

## 2. 网络结构

### 2.1 整体框图

```
Inputs: past_target + known_cov(past+future) + observed_cov(past) + static_cov
  ├─ 变量选择（numeric / categorical 各一次）
  ├─ LSTM 编码历史 → 门控残差 + 静态富化
  ├─ 可解释多头注意力（历史↔未来）
  └─ 位置前馈 + 门控
  → 分位数输出 (B, H, D, Q)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `GatedResidualNetwork` | Dense+ELU+Dense+GLU+残差+LayerNorm | 基础块 |
| `VariableSelectionNetwork` | GRN + softmax 权重 | **变量重要性** |
| `InterpretableMultiHeadAttention` | 共享 V 的多头注意力 | **时序重要性** |
| `StaticCovariateEncoder` | 4 个 GRN → c_h/c_c/c_s/c_e | 静态信息注入 |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs DeepAR | TFT 输出分位数（不用采样）；可解释性更强 |
| vs Transformer | TFT 多了变量选择 + 静态编码 + 门控 |
| vs LSTNet | TFT 显式用协变量，LSTNet 忽略 |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输入 | `{"past_target": (B, L, D)}`（协变量模型另有 `known_cov_numeric` 等） |
| 输出 | ``(B, out_chunk_len, target_dim, num_quantiles)`` |
| 解码 | **直接输出，无需后处理**（时序回归不同于检测，无 NMS/anchor） |
| 指标 | **pinball（分位数损失）** + PICP/MPIW + nRMSE |

> ⚠️ 与检测任务不同：**没有 anchor、没有 NMS、没有解码** —— 网络输出即预测值。

---

## 4. 配置与用法

```yaml
Architecture:
  model_family: ts_forecast
  task: ts_forecast
  Head:
    model: tft
    in_chunk_len: 96
    out_chunk_len: 24
    target_dim: 1
Loss:
  type: quantile
Metric:
  main_indicator: nRMSE
  pred_mode: quantile
```

```bash
tkiln train  -c configs/ts/tft_demo.yml
tkiln val    -c configs/ts/tft_demo.yml --weights output/.../best_accuracy.pth
tkiln predict -c configs/ts/tft_demo.yml --input data.csv
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.251 M** |
| 本框架对齐 | **rel **1e-6**（浮点级）** |
| 训练速度 | 合成数据 30 epoch < 1 分钟（本机 RTX 4060 Ti） |
| FLOPs | ⚠️ **本框架未集成 FLOPs 统计** |

---

## 6. 公开指标

**对齐验证**（同权重同输入，PaddleTS vs 本框架）：

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0** |
| 逐层前向 | **rel **1e-6**（浮点级）** |

> ℹ️ PaddleTS **不发布官方预训练权重**，故无"公开指标"可对比；
> 本框架的验收标准是**与 PaddleTS 参考实现数值一致**。

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| ⭐ **新能源功率预测** | ✅ **首选** |
| 需要**解释模型决策** | ✅ 变量/时序权重可直接看 |
| 有静态协变量（电站/机组 ID） | ✅ 支持最好 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| TFT 论文 | 在多个基准上同时做到高精度与可解释 |
| 实际使用 | AGENTS 记录：**TFT + weather 协变量 nRMSE 0.2014，比不用协变量好 16%** |

---

## 9. 已知问题 / 注意事项

- ⚠️ **需要协变量**：`known_num_dim/observed_num_dim/static_num_dim` 必须正确
  （框架未给时从数据集配置 `known_cols/observed_cols/static_cols` **自动推断**）
- ⚠️ **已知问题：训练与评估正常，但进程退出时偶发原生崩溃 `0xC0000409`**
  （孤立前向+反向退出正常，与训练循环后的 CUDA 状态有关，**不影响结果**）
- ⚠️ `output_quantiles` 必须与 `Metric.quantiles` 长度一致（否则 PICP 索引越界，已修）
- ⚠️ 补 `known/observed/static_cov_categorical` 空张量避免 `None.shape` 崩溃（已修）
