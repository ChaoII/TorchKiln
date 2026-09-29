# 时序任务总览与选型

> TorchKiln 时序部分 = **5 个任务 / 21 个模型 / 4 个检测算法**，全部移植或对齐 PaddleTS 1.1.0。

---

## 一、5 个任务

| 任务 | 干什么 | 模型数 | 指标 |
|---|---|---|---|
| **`ts_forecast`** | 未来 H 步预测（点/区间/概率） | **12 + 2** | MSE/MAE/RMSE/**nRMSE**/MAPE/sMAPE/R²/**PICP**/MPIW/pinball |
| **`ts_anomaly`** | 逐点异常检测（无监督） | **5** | P/R/F1(point-adjust)/**AUC-ROC**/**AUC-PR** |
| **`ts_classify`** | 序列分类（工况/故障类型） | **2** | accuracy / macro-F1 / 每类 F1 |
| **`ts_embed`** | 自监督表示学习 | **2** | 对比损失 / `repr_std` |
| **`ts_rul`** | 剩余寿命回归 | 复用预测族 | RMSE / MAE / **NASA Score** |
| `tkiln changepoint` | 变点/漂移检测（纯统计） | **4 算法** | 命中率 / 延迟 |

---

## 二、`ts_forecast` 12 模型的选型矩阵

| 模型 | 结构类型 | 周期建模 | 概率输出 | 协变量 | 参数 | 适合场景 |
|---|---|---|---|---|---|---|
| **NBEATS** | MLP+basis | 中 | ❌ | ❌ | 0.66M | 可解释分解，无协变量 |
| **NHiTS** | 多率+插值 | **强** | ❌ | ❌ | 0.94M | **长周期**预测 |
| **MLP** | 全连接 | ❌ | ❌ | ❌ | 0.03M | **基线/下界** |
| **DLinear** | 分解+Linear | 弱 | ❌ | ❌ | 0.005M | **极快基线**（5K 参数） |
| **TCN** | 因果膨胀卷积 | 中 | ❌ | ❌ | 0.15M | **在线/流式**（因果） |
| **RNN(LSTM/GRU)** | 循环 | 中 | ❌ | ❌ | 0.05~0.07M | 基础对照 |
| **LSTNet** | CNN+RNN+skip | **强** | ❌ | ❌ | 0.003M | **强季节性**（电力/流量） |
| **Transformer** | 自注意力 | 中 | ❌ | ❌ | 0.003M | 长依赖，但 O(L²) |
| **SCINet** | 采样卷积树 | 强 | ❌ | ❌ | 0.003M | 长序列降复杂度 |
| **Informer** | ProbSparse | 强 | ❌ | ❌ | 11.3M | 长序列（⚠️ 含随机性） |
| **DeepAR** | RNN+Gaussian | 中 | ✅**概率** | ✅ | 0.07M | **不确定性量化** |
| **TFT** | GRN+变量选择 | 强 | ✅**分位数** | ✅**最强** | 0.25M | ⭐**功率预测首选** |

> ⚠️ **重要**：`RNN/LSTNet/Transformer/SCINet` 等按 PaddleTS 设计**只用 `past_target`**，
> **完全忽略 known/observed 协变量**（实测加 weather 结果逐位相同）。
> **需要协变量必须用 `TFT` / `DeepAR` / `NBEATS`(部分)**。

---

## 三、选型决策树

```
有没有外生变量（天气/日历/工况）？
├─ 有 → TFT（分位数+可解释）或 DeepAR（概率）
│        ├─ 要区间预测 → TFT（pred_mode=quantile）
│        └─ 要概率分布采样 → DeepAR
└─ 无 → 序列有多长？
         ├─ 长（>512）→ Informer / SCINet / NHiTS
         ├─ 中（96~512）
         │    ├─ 有强周期性 → LSTNet（skip-RNN）
         │    ├─ 要在线推理 → TCN（因果卷积）
         │    └─ 通用 → NBEATS / NHiTS
         └─ 短（<96）→ MLP / DLinear（基线）
```

---

## 四、输出形态（三种 pred_mode）

| 模式 | 输出 | 用途 | 模型 |
|---|---|---|---|
| `point` | `(B,H,D)` | 点预测 | 大部分 |
| `quantile` | `(B,H,D,Q)` | **区间预测**（PICP/MPIW） | TFT |
| `params` | `(B,H,D,2)` | 概率参数（μ,σ） | DeepAR |

---

## 五、数据管线（`TSDataset`）

CSV 列角色（配置在 `<mode>.dataset`）：

| 列角色 | 说明 | 例子 |
|---|---|---|
| `target_cols` | 目标序列 | `power` |
| `known_cols` | **已知协变量**（past+future 全窗） | `hour_sin, hour_cos, weather` |
| `observed_cols` | **观测协变量**（仅 past） | `irradiance, temp` |
| `static_cols` | 静态协变量（每序列常数） | 电站 ID |

---

## 六、评测口径（可复用的方法论）

| 指标 | 定义 | 注意 |
|---|---|---|
| **nRMSE** | `RMSE / RMS(target)`（或 `/装机容量`） | 新能源功率预测标准 |
| **PICP** | 区间覆盖率 | 与 `nominal_coverage` 比，看区间是否可信 |
| **MPIW** | 平均区间宽度 | 越窄越好（PICP 达标前提下） |
| MAPE/sMAPE | 在 target≈0 时会爆 | 仅作参考 |
| **NASA Score** | PHM08 非对称 | RUL 专用，滞后预测罚重 |

---

## 七、已知限制

- ⚠️ `Informer` 含随机 key 采样，**不可逐位复现**（算法固有）。
- ⚠️ `TFT` / `BiLSTM+Attn` / `TransformerReg` 有**退出期原生崩溃**（`0xC0000409`），
  训练与指标正常（已知问题）。
- ⚠️ `ts_rul` 的文献 RMSE 12~18 **对应"筛选易样本"口径**；本框架同口径下为 12.38（RUL≤50）。
- ⚠️ **对比 RUL 必须声明口径**（是否筛 RUL 区间/是否按台聚合/是否取最好 seed）。
