# VAE（时序异常检测）

> **定位**：在 AE 上加**变分推断**：隐空间建模为高斯，重建更稳定
> **任务**：`ts_anomaly` | **权重**：无需预训练

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Auto-Encoding Variational Bayes (VAE) |
| arXiv | 1312.6114 |
| 年份 | 2013 |
| 上游实现 | paddlets `models/anomaly/dl/vae.py::VAE` + `_VAEBlock` |
| 本框架实现 | `torchkiln/nn/s_anomaly.py::VAEBlock / VAEStack` |
| 对齐方式 | 同起点 dump → torch 加载 → fp64 判定 |

### 要解决的问题

普通 AE 的隐空间**无结构**，易过拟合；且无概率解释。

### 核心创新

- **重参数化**：`z = μ + ε·σ`，使采样可反传
- **KL 正则**：隐空间趋近标准正态 → 更平滑、更抗过拟合
- **损失**：`smooth_l1(recon, x) + β·KL`（`kld_beta=0.2`，对齐 paddlets）

---

## 2. 网络结构

```
Input (B,L,C) → _encoder（MLP/CNN/LSTM）→ h
  ├─ mu = Linear(h)      → (B,L,C)
  ├─ logvar = Linear(h)  → (B,L,C)
  └─ z = mu + ε·exp(0.5·logvar)·stdev   （训练采样；eval 用 mu）
  → _decoder(z) → recon
```

| 模块 | 结构 | 作用 |
|---|---|---|
| `stack` | 按 `base_nn` 选 MLP/CNN/LSTM | 编/解码器 |
| `_VAEBlock` | encoder + mu/logvar/reconstructed 三个 Linear | 核心 |
| `reparameterize` | 训练采样 / eval 用 μ | 变分技巧 |

⚠️ **`activation` 固定为 ReLU6**（对齐 Paddle 实现，非论文原文）。
⚠️ **`stdev=0.1` 缩放噪声**（paddlets 特有，非标准 VAE）。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `[recon, mu, logvar, x]` 四元组 |
| 异常分数 | `smooth_l1(recon, x)` 通道平均 |
| 损失 | `smooth_l1_loss(recon,x,sum) + 0.2·KL` |
| 指标 | 同 AE（P/R/F1 + AUC） |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
Architecture:
  task: ts_anomaly
  algorithm: vae
  hidden_config: [32, 16]
  base_en: MLP
  base_de: MLP
  use_bn: true
  dropout_rate: 0.2
  stdev: 0.1
```

```bash
tkiln train -c configs/ts/anomaly_vae_demo.yml -o Global.epoch_num=20
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.0020 M** |
| 对齐 | **maxdiff **5.2e-16**（fp64）** |
| 本机速度 | 与 AE 相当（<30 秒/20 epoch） |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0**（14 键） |
| fp64 前向 | recon 5.2e-16 / mu 5.0e-17 / logvar 5.9e-17 |
| 端到端 AUC-ROC | **0.776**（本批最好） |

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| 数据有噪声 | ✅ 优于 AE |
| 需要最简方案 | 用 AE |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| vs AE | VAE 多了 KL 与隐空间概率解释 |
| vs β-VAE | paddlets 的 `kld_beta=0.2` 是软约束 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`kld_beta=0.2`**（paddlets 值），非标准 VAE 的 1.0
- ⚠️ eval 模式下 `reparameterize` **返回 μ**（不采样）⇒ 复现有确定性
- ⚠️ `stdev=0.1` 是 paddlets 特有的噪声缩放
