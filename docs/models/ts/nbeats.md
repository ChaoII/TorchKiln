# NBEATS

> **定位**：可解释的纯 MLP 预测器：把预测分解成 trend/season 两条基函数路径
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting |
| arXiv | 1905.10437 |
| 年份 | 2019 |
| 机构 | Element AI（Bengio 组） |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/nbeats.py` |
| 本框架实现 | ``torchkiln/nn/ts_models.py::NBEATS` / `_Block`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

传统时序模型（ARIMA/ETS）需人工指定趋势与季节形式；RNN/CNN 又难解释。能否用一个**纯 MLP** 同时做到「高精度 + 可解释」？

### 核心创新

- **Basis expansion**：每个 block 只输出 trend（多项式基）或 seasonality（傅里叶基）的**系数**，
  再用固定基函数展开 → 预测天然可分解、可视化
- **Residual stacking**：每个 block 先「backcast」减去已解释部分，再「forecast」加回预测 → 逐步逼近
- **纯全连接**，无循环/卷积

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B, L, D)  →  flatten → (B, L*D)
  └─ Stack 1（trend）:  FC→ReLU×4 → trend_coef → 多项式基展开 → backcast+forecast
  └─ Stack 2（trend）:  同上去残差
  └─ Stack 3（season）: FC→ReLU×4 → season_coef → 傅里叶基展开
  最终 forecast = Σ 各 stack forecast  →  (B, H, D)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_Block` | 4 层 FC（`expansion_coefficient_dim`）+ 分支出 `theta_b/theta_f` | 单 block |
| `_TrendGenerator` | `theta` → 多项式基矩阵 → backcast/forecast | trend 展开 |
| `_SeasonalityGenerator` | `theta` → cos/sin 傅里叶基 → 展开 | 季节展开 |
| `_Stack` | n 个 block + 残差连接 | 堆叠 |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs NHiTS | NBEATS 单率、同分辨率；NHiTS 多率采样 → 长周期更好 |
| vs MLP | MLP 直接回归；NBEATS 有 basis 结构 + 残差堆叠 |

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
    model: nbeats
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
| 参数 | **0.663 M** |
| 本框架对齐 | **maxdiff=0（**逐位一致**）** |
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
| **FLOPs** | **1.315 MFLOPs** |
| **MACs** | **0.657 MMACs** |
| 参数量 | **0.663 M** |
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
| 需要**可解释分解** | ✅ 首选 |
| 长周期（>96 步） | 改用 NHiTS |
| 有外生变量 | ❌ 不支持，改用 TFT |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| NHiTS | 同为 basis expansion 家族，NHiTS 多率采样 |
| DLinear | 分解思路相似，但 DLinear 只有 5K 参数 |
| TFT | TFT 可解释性来自注意力权重，NBEATS 来自基函数 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **不支持协变量**（只用 `past_target`）
- ⚠️ `hidden_layer_units`/`num_stacks` 是最关键超参；默认 2 stack 偏浅
- ⚠️ 对比 PaddleTS 时须用同一 `generic_architecture` 设置
