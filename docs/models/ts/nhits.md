# NHiTS

> **定位**：NBEATS 的**多率**升级版：分层插值 + 多分辨率采样，长周期预测更强
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting |
| arXiv | 2201.12886 |
| 年份 | 2022 |
| 机构 | Element AI |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/nhits.py` |
| 本框架实现 | ``torchkiln/nn/ts_models.py::NHiTS` / `_NHiTSBlock` / `_NHiTSStack`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

NBEATS 对**长 horizon** 预测误差大，且所有 block 都看**同一分辨率**，无法同时捕捉「长期趋势」与「短期波动」。

### 核心创新

- **Multi-rate sampling**：不同 stack 先按不同因子 max-pool 下采样 → 各看不同分辨率
- **Hierarchical interpolation**：低分辨率的粗预测**插值上采样**后作为精修信号
- 每 block 输出**多个 horizon 段**（`pooling_kernel_sizes` 控制）

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D)
  └─ Stack 0: MaxPool(k=8) → 粗分辨率 block → 插值放大 → 残差
  └─ Stack 1: MaxPool(k=4) → 中分辨率 block → 插值放大 → 残差
  └─ Stack 2: 原始分辨率 block → 精修
  Σ → (B, H, D)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_NHiTSBlock` | 多段 FC + `MaxPool1d` 下采样 + 插值上采样 | 单 block |
| `_NHiTSStack` | n 个 block + 残差 | 单分辨率 stack |
| `n_freq_downsample` | 各 stack 的下采样因子 | 多率控制 |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs NBEATS | 多率采样 + 分层插值（NBEATS 单率） |
| vs TCN | TCN 靠膨胀卷积扩感受野，NHiTS 靠下采样 |

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
    model: nhits
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
| 参数 | **0.935 M** |
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
| **FLOPs** | **1.864 MFLOPs** |
| **MACs** | **0.932 MMACs** |
| 参数量 | **0.935 M** |
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
| **长周期**（>96 步） | ✅ 首选 |
| 有多尺度周期 | ✅ 多率采样天然适配 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| NBEATS | NHiTS 是其后继，多率是核心增量 |
| Informer/SCINet | 后两者做长序列，但参数量/复杂度更高 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **对拍必须用 `dropout=0.0`**：`_NHiTSBlock` 的 `if dropout > 0` 才插入 Dropout 层，
  否则 Linear 索引错位（0/3 vs 0/2），dump 的参数会对不上
- ⚠️ 不支持协变量
