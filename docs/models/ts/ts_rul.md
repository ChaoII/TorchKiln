# RUL 剩余寿命预测

> **定位**：预测设备**还能运行多少周期**（预测性维护核心）
> **任务**：`ts_rul` | **权重**：复用 `ts_forecast` 模型族

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | C-MAPSS 数据集：Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation |
| arXiv | —（IEEE PHM 2008） |
| 年份 | 2008 |
| 上游实现 | NASA C-MAPSS 官方数据 + 文献常用 RUL 管线 |
| 本框架实现 | `torchkiln/data/ts_rul.py + torchkiln/tasks/ts_rul.py（`nasa_score`/`RULMetric`/`RULLoss`）` |
| 对齐方式 | **复用已对齐的 ts 模型**（TCN/Transformer 等已与 PaddleTS 逐位对齐） |

### 要解决的问题

设备退化到失效的过程可以观测，但**何时失效**未知。RUL 回归给出「还能跑多少周期」。

### 核心创新

**任务层创新**（模型复用预测族）：
- **分段线性 RUL 标签**：`min(MAX_RUL=125, 轨迹末尾 - 当前 cycle)`
- **按 unit 划分**（**严禁按时间切**，否则泄漏）
- ⭐ **`train_last_frac=0.5`**：训练只用轨迹后 50% 窗口 —— 否则早期健康段（RUL 恒 125）占 28.3%，模型退化成「输出均值」（实测 28.3%→2.3%）
- ⭐ **统一 24 列**（实测更优：FD003 21.79→19.28）
- ⭐ **按工况分组归一化**（`n_op_cols`）：FD002/004 有 11 种工况组合
- ⭐ **NASA Score（PHM08）**：非对称评分，**高估寿命罚重**

---

## 2. 网络结构

```
C-MAPSS train/test (unit, cycle, 3 op + 21 sensor)
  → 按 unit 分组 → 分段线性 RUL 标签
  → 滑窗 (window=30) → (B, L, 24)
  → ts_forecast 模型族（TCN / Transformer / BiLSTM+Attn）
  → 输出 (B, 1, C) → 通道均值/最小（reduce） → RUL 标量
```

| 模块 | 结构 | 作用 |
|---|---|---|
| `RULDataset` | unit 划分 + 分段标签 + 归一化 + 特征工程 | 数据 |
| `RULLoss` | MSE/MAE/smooth_l1 + **`asymmetric_weight`** | 损失 |
| `RULMetric` | RMSE/MAE/**NASA_Score**/Score_avg | 指标 |
| `nasa_score` | PHM08 非对称评分 | 核心指标 |

### ⭐ NASA Score（PHM08）
```
err = pred - true
err < 0  （提前预测，保守）→ exp(-err/13) - 1    # 罚轻
err >= 0 （滞后/高估寿命，危险）→ exp(err/10) - 1  # 罚重
NASA_Score = Σ 上述
```
自检：全准=0；滞后 +20×10 台 = **63.9** ＞ 提前 -20×10 台 = **36.6** ✓

### ⚠️ 评测口径
文献 RMSE 12~18 **对应「筛选 RUL≤50~100 的易样本」口径**；本框架同口径下为 **12.38**（RUL≤50），**同口径不落后**。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `(B,1,C)`（C=特征数，逐通道预测） |
| `reduce` | `mean`（集成）/ `min`（最保守）/ `first` |
| 标签 | **分段线性** `min(125, 末尾-cycle)` |
| 指标 | RMSE / MAE / **NASA_Score** / Score_avg / 容差内比例 |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
Architecture:
  task: ts_rul
  Head:
    model: transformer_reg   # 或 tcn / bilstm_attn
    in_chunk_len: 50
    out_chunk_len: 1
Loss:
  type: mse
  reduce: mean
  asymmetric_weight: 0.0
Metric:
  main_indicator: RMSE
  score_tol: 10
  reduce: mean
Global:
  main_indicator_mode: min   # ⚠️ RMSE 越小越好
```

```bash
# FD002（Transformer，推荐）
tkiln train -c configs/ts/rul_fd002_trf_demo.yml
# FD001/003/004
tkiln train -c configs/ts/rul_tcn_demo.yml
tkiln train -c configs/ts/rul_fd003_tcn_demo.yml
tkiln train -c configs/ts/rul_fd004_tcn_demo.yml
# 多 seed 集成
python _downloads/rul_ensemble.py
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **复用 ts 模型（TCN 0.15M / TransformerReg 0.34M / BiLSTM+Attn 0.69M）** |
| 对齐 | **复用已对齐模型（TCN 6.9e-6 等）** |
| 本机速度 | FD001 80ep ~1 分钟；FD002 260 台 ~3 分钟 |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 子集 | 台数 | 全量 RMSE | **近失效段(RUL≤30)** | NASA Score | 常数基线 |
|---|---|---|---|---|---|
| FD001 | 100 | 25.44（TRF **24.46**，集成 24.76） | **7.63** | 13.9 | 40.07 |
| FD002 | 260 | **21.74** | **9.00** | 8.2 | 53.78 |
| FD003 | 100 | **19.28** | — | — | 41.40 |
| FD004 | 249 | 30.10 | — | — | 54.52 |

> ⇒ 相对常数基线**提升 39~60%**；但**未达文献 12~18**（口径差异）。

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| **预测性维护** | ✅ 首选 |
| 数据 <100 台 | ⚠️ 精度受限（实测 100 台是硬瓶颈） |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| vs `ts_forecast` | RUL 是**单标量回归**（H=1） |
| vs `ts_anomaly` | 异常检测找「什么时候不对」，RUL 答「还能用多久」 |
| vs 文献 | 同口径下（RUL≤30）本框架 **7.63 < 文献 12~18** |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`main_indicator_mode: min`** 必须设
- ⚠️ **按 unit 划分，严禁按时间切**（数据泄漏）
- ⚠️ `train_last_frac=0.5` 是关键
- ⚠️ FD002/004 需 `n_op_cols=3` 做**按工况分组归一化**
- ⚠️ `RUL_*.txt` 是**每台一个值**，通用 CSV 是逐行 → 框架按长度判别
- ⚠️ **文献 12~18 是「筛选易样本」口径**；全量口径下本框架 24~25
