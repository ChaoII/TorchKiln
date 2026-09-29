# 表示学习（TS2Vec / CoST）

> **定位**：**自监督**学通用时序表征，再送下游分类/聚类/预测
> **任务**：`ts_embed` | **权重**：无需预训练

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | TS2Vec / CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations |
| arXiv | 2106.10466 / 2202.01575 |
| 年份 | 2021 / 2022 |
| 上游实现 | paddlets `models/representation/dl/{ts2vec,cost}.py` + `_{ts2vec,cost}/` |
| 本框架实现 | `torchkiln/nn/ts_ts2vec.py / torchkiln/nn/ts_cost.py` |
| 对齐方式 | 同起点 dump → torch 加载 → 前向+损失 fp64 判定 |

### 要解决的问题

标注数据稀缺时，需要**无标签预训练**得到通用表征，再小样本微调。

### 核心创新

**TS2Vec**：
- 两个**独立随机 mask** 视图 → 分层对比损失
- **Hierarchical**：instance-level + temporal-level 逐层 max-pool 聚合
- **SWA**（随机权重平均）提升推理稳定性

**CoST**：
- **TFD**：多尺度移动平均提取 trend
- **SFD**：**rfft → 复权重投影 → irfft** 提取 season
- **时域对比** + **频域对比**

---

## 2. 网络结构

```
**TS2Vec**
  Input (B,L,C) → _in_proj Linear → mask → DilatedConvLayer(膨胀 1,2,4...)
  → 表征 (B,L,D)
  两个视图 r1,r2 → hierarchical_contrastive_loss(r1,r2)

**CoST**
  Input (B,L,C) → _in_proj → DilatedConvLayer → (B,D,L)
  ├─ _tfd: 多尺度 Conv1d(k=1,2,4,...,128) 平均 → trend
  └─ _sfd: rfft → 复权重 → irfft → season
  时域对比(anchor/pos/neg) + 频域对比
```

| 模块 | 结构 | 作用 |
|---|---|---|
| `SamePadConv` | SAME 填充 + Conv1d（`_remove` 处理偶数感受野） | 基础卷积 |
| `ConvLayer` | 残差 + 2×[GELU+Conv] + `_out_proj` | 残差块 |
| `DilatedConvLayer` | dilation=2^k 堆叠 | 扩感受野 |
| `TS2VecModule` | feat_extractor + SWA 平均 | TS2Vec |
| `TFDLayer` | 多尺度因果移动平均 | CoST trend |
| `SFDLayer` | rfft + 复权重 + irfft | CoST season |

⚠️ **TS2Vec 的 `mask` 参数只接受字符串**（`binomial`/`all_true`/`mask_last`），传张量会 `AssertionError` ⇒ 对拍需 monkeypatch。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | TS2Vec `(B,L,D)`；CoST `(trend, season)` |
| 损失 | TS2Vec 分层对比；CoST 时域+频域对比 |
| 指标 | 对比损失 + **`repr_std`**（趋 0 = 表征塌缩） |
| 下游 | 导出表征 → 送 `ts_classify` / 聚类 / `ts_forecast` |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
Architecture:
  task: ts_embed
  algorithm: ts2vec   # 或 cost
  in_chunk_len: 96
  num_features: 3
  hidden_dim: 32
  output_dim: 64
  depth: 4
```

```bash
tkiln train -c configs/ts/embed_ts2vec_demo.yml -o Global.epoch_num=10
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **TS2Vec 0.0456 M / **CoST 0.7720 M**** |
| 对齐 | **TS2Vec encoder **1.5e-16**、loss_inst/temp **逐位 0.0**、loss_hier 6.8e-8；CoST trend **5.4e-16** / season **4.0e-16** / loss_time **1.2e-16** / loss_freq **1.6e-16**** |
| 本机速度 | 自监督训练与普通模型相当 |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0** |
| fp64 前向 | 1e-16 级 |
| 端到端（无标签合成） | TS2Vec loss 3.80→**3.33**；CoST 4.34→**3.68**；`repr_std` 非零（**无塌缩**） |

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| **小样本/无标签** | ✅ 首选 |
| 数据充足有标签 | 直接监督训练 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| TS2Vec vs CoST | TS2Vec 用层次对比；CoST 显式**解耦 trend/season** |
| 用途差异 | 只学表征，**本身不做预测** |

---

## 9. 已知问题 / 注意事项

- ⚠️ **TS2Vec 的 `mask` 只接受字符串**
- ⚠️ **CoST `SFDLayer` 的 rfft 维度需显式对齐**；`bias` 需补 batch 维；Paddle `as_complex` 末维必须为 2
- ⚠️ **表征塌缩检测**：若 `repr_std → 0` 说明对比学习失败
- ⚠️ paddlets 的 `representation/task/` 本框架**未单独实现**，用「导出表征 → ts_classify」替代
