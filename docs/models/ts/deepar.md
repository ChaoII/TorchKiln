# DeepAR

> **定位**：**概率预测**：输出高斯分布参数，可采样得到任意区间
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks |
| arXiv | 1704.04110 |
| 年份 | 2017 |
| 机构 | Amazon |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/deepar.py` |
| 本框架实现 | ``torchkiln/nn/s_deepar.py::DeepAR`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

点预测无法表达**不确定性** —— 业务（库存/风控）需要「90% 区间」而不是单个数。

### 核心创新

- **GaussianLikelihood**：网络输出 `(μ, σ)` 而非点值
- **NLL 训练**：`-log N(y|μ,σ)`
- 可**多次采样**得到任意分位数/预测区间

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D) + covariates
  └─ RNN（自回归，训练用 teacher forcing）
  └─ Linear → (μ, log σ)
  → (B,H,D,2)  或采样 (B,H,D)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_rnn` | LSTM | 编码历史 |
| `_distr` | `GaussianLikelihood` → μ,σ | **概率输出** |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs TFT | DeepAR 只输出高斯；TFT 直接输出分位数且更可解释 |
| vs 点预测模型 | 多了不确定性量化 |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输入 | `{"past_target": (B, L, D)}`（协变量模型另有 `known_cov_numeric` 等） |
| 输出 | ``params` 模式 `(B,H,D,2)` [μ,σ] 或 `quantile` 采样` |
| 解码 | **直接输出，无需后处理**（时序回归不同于检测，无 NMS/anchor） |
| 指标 | **NLL** / MSE（取 μ） |

> ⚠️ 与检测任务不同：**没有 anchor、没有 NMS、没有解码** —— 网络输出即预测值。

---

## 4. 配置与用法

```yaml
Architecture:
  model_family: ts_forecast
  task: ts_forecast
  Head:
    model: deepar
    in_chunk_len: 96
    out_chunk_len: 24
    target_dim: 1
Loss:
  type: nll
Metric:
  main_indicator: MSE
  pred_mode: params
```

```bash
tkiln train  -c configs/ts/deepar_demo.yml
tkiln val    -c configs/ts/deepar_demo.yml --weights output/.../best_accuracy.pth
tkiln predict -c configs/ts/deepar_demo.yml --input data.csv
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.067 M** |
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
| **需要概率分布/采样** | ✅ 首选 |
| 需要区间但不用分布 | TFT 更直接 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| DeepAR 论文 | 在 Amazon 零售数据上验证 |
| TFT | 两者都概率，TFT 多了变量选择 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **需要协变量**（`known/observed/static`）；框架从数据集配置自动推断维度
- ⚠️ `Head.num_samples` 控制采样数（默认 10）
- ⚠️ 评估时 `pred_mode=params` 取 μ（框架 `TSMetric._reduce`）
