# MLP

> **定位**：最朴素的基线：展平窗口后全连接，**没有时序结构**
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | 无独立论文（PaddleTS 内置基线） |
| arXiv | — |
| 年份 | — |
| 机构 | PaddleTS |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/mlp.py` |
| 本框架实现 | ``torchkiln/nn/ts_models.py::MLP`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

需要一个**下界参照**：如果 MLP 就能做好，说明任务没有强时序依赖，不必上 RNN/Transformer。

### 核心创新

- 无（就是基线）
- 价值在于：**任何复杂模型都应显著优于它**，否则结构是白加的

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D) → flatten (B, L*D) → Linear×n + BN + ReLU → (B, H*D) → reshape (B,H,D)
```

### 2.2 逐模块说明

| 模块 | 结构 |
|---|---|
| `_nn` | `Linear → (BN) → ReLU` 堆叠，末层输出 `H*D` |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs DLinear | DLinear 先做序列分解，MLP 不分解 |
| vs NBEATS | NBEATS 有 basis 结构 |

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
    model: mlp
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
| 参数 | **0.032 M** |
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
| 作为**基线** | ✅ 必跑 |
| 正式建模 | 若复杂模型不显著更好，说明任务简单 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| 所有其它模型 | MLP 是「无结构」参照 |

---

## 9. 已知问题 / 注意事项

- ⚠️ 展平后参数量随 `L*D` 增长，**长窗口会爆参**
- ⚠️ 不支持协变量
