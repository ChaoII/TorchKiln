# TCN

> **定位**：**因果膨胀卷积**：并行度高、感受野指数增长，适合在线流式预测
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling |
| arXiv | 1803.01271 |
| 年份 | 2018 |
| 机构 | （TCN 为该论文提出的架构） |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/tcn.py` |
| 本框架实现 | ``torchkiln/nn/ts_models.py::TCN` / `_TemporalBlock`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

RNN 序列依赖导致**无法并行**、长序列**梯度不稳定**；而普通 CNN 感受野太小。

### 核心创新

- **因果卷积**：`padding` 只加在左侧 → 第 t 步只看 `≤t`，**可在线推理**
- **膨胀卷积**：第 k 层 `dilation=2^k` → 层数与感受野成**指数关系**
- **残差 + weight_norm**：深层可训练

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,C) → transpose (B,C,L)
  └─ TemporalBlock(d=1):  Conv(k=3,d=1)+weight_norm+ReLU+Dropout ×2 + residual
  └─ TemporalBlock(d=2):  同上，d=2
  └─ TemporalBlock(d=4):  同上，d=4
  → transpose → 取最后 H 步 → (B,H,C)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_TemporalBlock` | 2×[Conv1d(dilation) → weight_norm → ReLU → Dropout] + 残差 | 单层 |
| `_downsample` | 1×1 Conv（通道不匹配时） | 残差对齐 |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs RNN | TCN **完全并行**，无序列依赖 |
| vs Transformer | TCN O(L)，Transformer O(L²) |
| vs SCINet | SCINet 用下采样，TCN 用膨胀 |

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
    model: tcn
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
| 参数 | **0.150 M** |
| 本框架对齐 | **rel **6.9e-6**（浮点级）** |
| 训练速度 | 合成数据 30 epoch < 1 分钟（本机 RTX 4060 Ti） |
| FLOPs | ⚠️ **本框架未集成 FLOPs 统计** |

---

---

---

---

---

### 📊 FLOPs（实测）

| 项 | 值 |
|---|---|
| **FLOPs** | **28.509 MFLOPs** |
| **MACs** | **14.255 MMACs** |
| 参数量 | **0.150 M** |
| 输入规格 | `L=96, H=24, D=1` |
| 测量工具 | `torch.utils.flop_counter.FlopCounterMode`（PyTorch 内置） |
| 复现脚本 | `_downloads/flops_measure*.py` |

> **口径**：`FLOPs` 是乘加各计 1 次（×2），**与 ultralytics 官方表的 GFLOPs 同口径**
> （已由 yolo11/v8 十个模型逐个吻合验证，见 [`_FLOPS.md`](_FLOPS.md)）；
> `MACs = FLOPs / 2`。
> ⚠️ `FlopCounterMode` **不计自定义算子**（NMS / probiou / iSTFT 等后处理）⇒
> 此处是**网络主干**的 FLOPs。
## 7. 选型建议

| 场景 | 建议 |
|---|---|
| **在线/流式**（因果） | ✅ 首选 |
| 长序列 | ✅ 比 Transformer 省显存 |
| 作为通用基线 | ✅ |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| RNN 家族 | TCN 是论文中「要替代 RNN」的方案 |
| RUL 场景 | 本框架 RUL 实测 TCN 优于 RNN（FD002 23.52 vs 34.33） |

---

## 9. 已知问题 / 注意事项

- ⚠️ weight_norm 的 `weight_g` 在 paddle 是 `[C]`、torch 是 `[C,1,1]`，加载需 reshape
- ⚠️ 不支持协变量
- ⭐ **RUL 推荐**：C-MAPSS FD002 上 RMSE 23.52（RUL≤30 段 9.00）
