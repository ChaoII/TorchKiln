# LSTNet

> **定位**：**skip-RNN 建模周期**：CNN 抓短期 + RNN 抓长期 + 跳跃 RNN 抓季节性
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks |
| arXiv | 1703.07015 |
| 年份 | 2017 |
| 机构 |  |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/lstnet.py` |
| 本框架实现 | ``torchkiln/nn/s_lstnet.py::LSTNet`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

真实时序同时有**短期波动**与**长期周期**（电力/流量），普通 RNN 很难同时建模两种尺度。

### 核心创新

- **CNN 分支**：抓短期局部模式
- **RNN 分支**：抓长期依赖
- **⭐ skip-RNN**：按 `skip` 步**跳着连接**（如每 24 步）→ 直接建模**周期**
- **highway**：缓解深层梯度问题

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D)
  ├─ Conv1d(k) → ReLU → GRU            → (B, h)
  ├─ GRU 在「跳采样」序列上（skip=24）  → (B, h)
  ├─ 线性 + highway 组合
  └─ 周期分支：Linear(24 → H) 直接映射
  Σ → (B,H,D)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_conv` | Conv1d + ReLU + Dropout | 短期 |
| `_rnn` | GRU | 长期 |
| `_skip_rnn` | 跳采样的 GRU | **周期** |
| `_highway` | Linear + sigmoid 门 | 融合 |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs RNN | LSTNet 多了 CNN 与 **skip** 分支 |
| vs NHiTS | 都做多尺度，LSTNet 用跳跃，NHiTS 用下采样 |

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
    model: lstnet
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
| 参数 | **0.003 M** |
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
| **FLOPs** | **0.129 MFLOPs** |
| **MACs** | **0.065 MMACs** |
| 参数量 | **0.003 M** |
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
| **强季节性**（日/周周期） | ✅ 首选（`skip` 设成周期长度） |
| 参数受限 | ✅ 仅 3K 参数 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| LSTNet 论文 | 在电力/交通数据集上优于同期 RNN |
| NHiTS | 多尺度思路不同 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`skip` 必须按数据周期设**（如 5 分钟采样、日周期 → skip=288）
- ⚠️ **忽略协变量**
