# USAD（时序异常检测）

> **定位**：**对抗式双解码器**自编码器：用两个 decoder 对抗训练提升鲁棒性
> **任务**：`ts_anomaly` | **权重**：无需预训练

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | USAD: UnSupervised Anomaly Detection on Multivariate Time Series |
| arXiv | —（KDD 2020） |
| 年份 | 2020 |
| 上游实现 | paddlets `models/anomaly/dl/usad.py::USAD` + `_USAModule` |
| 本框架实现 | `torchkiln/nn/s_anomaly.py::USADBlock + TSAnomalyLoss._usad_step（两阶段）` |
| 对齐方式 | 同起点 dump → torch 加载 → fp64 判定 |

### 要解决的问题

单解码器 AE 在**多维时序**上容易被「捷径」欺骗（学恒等映射），异常检测失效。

### 核心创新

- **共享 encoder + 两个 decoder**：decoder1 学「正常重建」、decoder2 学「放大异常」（对抗）
- **两阶段训练**（本框架需 `train_step` 钩子）：
  · 阶段1：`L1 = (1/n)·‖x-w1‖² + (1-1/n)·‖x-w3‖²`
  · 阶段2：`L2 = (1/n)·‖x-w2‖² - (1-1/n)·‖x-w3‖²`
  · 其中 `w1=dec1(enc(x))`、`w2=dec2(enc(x))`、`w3=dec2(enc(w1))`
- **异常分数**：`α·‖x-w1‖² + β·‖x-w3‖²`（默认 α=β=0.5）

---

## 2. 网络结构

```
Input x (B, L*C)  [flatten=True]
  └─ _encoder → z
       ├─ _decoder1(z) = w1       （正常重建）
       └─ _decoder2(z) = w2       （对抗放大）
           └─ _encoder(w1) → _decoder2 → w3
  → (x, w1, w2, w3)
```

| 模块 | 结构 | 作用 |
|---|---|---|
| `_encoder` | MLP/CNN | 共享编码 |
| `_decoder1` | MLP/CNN | 正常重建 |
| `_decoder2` | MLP/CNN | 对抗 |
| 两个 Adam | `opt1=enc+dec1` / `opt2=enc+dec2` | **两阶段优化** |

⚠️ **`flatten=True`（且 `ed_type='MLP'`）时会强制关掉 encoder 的 BN**（对齐 Paddle）。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `(x, w1, w2, w3)` |
| 异常分数 | `0.5·‖x-w1‖² + 0.5·‖x-w3‖²` |
| 训练 | **两阶段**：每 batch 两次 backward + step |
| 指标 | 同 AE |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
Architecture:
  task: ts_anomaly
  algorithm: usad
  ed_type: MLP
  hidden_config: [32, 16]
  flatten: true
Loss:
  alpha: 0.5
  beta: 0.5
```

```bash
tkiln train -c configs/ts/anomaly_usad_demo.yml -o Global.epoch_num=20
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.0205 M** |
| 对齐 | **maxdiff **1e-16 级**（fp64，三种配置）** |
| 本机速度 | 比 AE 慢（两阶段 → 每 batch 两次前向反向） |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0**（12 键） |
| fp64 前向 | x 逐位 0 / w1 3.0e-16 / w2 4.9e-16 / w3 4.3e-16 |
| 端到端 AUC-ROC | **0.769** |

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| 多变量时序异常 | ✅ 优于单解码器 AE |
| 算力受限 | 用 AE（USAD 慢一倍） |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| vs AE | USAD 多一个对抗 decoder |
| vs MTAD-GAT | MTAD-GAT 用图注意力，USAD 用对抗 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **必须用 `train_step` 钩子**（本框架 `BaseTrainer` 支持）：USAD 每 batch 需两次 backward + step
- ⚠️ `flatten=True` 时**自动关闭 encoder BN**
- ⚠️ 两个优化器共享 encoder（`enc+dec1` 与 `enc+dec2`）
