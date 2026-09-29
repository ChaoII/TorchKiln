# 时序分类（CNN / InceptionTime）

> **定位**：把一段序列判成某个**工况/故障类型**
> **任务**：`ts_classify` | **权重**：无需预训练

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | InceptionTime: Finding AlexNet for Time Series Classification |
| arXiv | 1909.04939 |
| 年份 | 2019 |
| 上游实现 | paddlets `models/classify/dl/{cnn,inception_time}.py` |
| 本框架实现 | `torchkiln/nn/ts_classify.py::CNNBlock / InceptionModule / InceptionBlock / InceptionTime` |
| 对齐方式 | 同起点 dump → torch 加载 → fp64 判定 |

### 要解决的问题

故障诊断场景：给定一段传感器序列，判断**属于哪种工况/故障类型**（多分类）。

### 核心创新

**InceptionTime**：
- **Inception 模块**：3 个不同核尺寸的卷积分支 + 1 个 maxpool 分支 → concat
- **残差 shortcut**：每 3 个 module 加跳跃连接
- **GAP + Linear**：全局平均池化后分类
- 是 TSC 领域的「AlexNet」：结构简单但在大量基准上 SOTA

---

## 2. 网络结构

```
**CNN（基线）**
  Input (B,L,C) → transpose (B,C,L)
  └─ [Conv1d(k=7) → (BN) → 激活 → AvgPool(3) → (Dropout)] × n
  └─ Flatten → Linear → Softmax → (B, n_class)

**InceptionTime**
  Input (B,L,C) → transpose (B,C,L)
  └─ InceptionBlock(depth=6):
       InceptionModule × 6:
          ├─ bottleneck Conv1d(k=1)
          ├─ Conv1d(k=40) / Conv1d(k=20) / Conv1d(k=10)   （奇数核）
          ├─ MaxPool1d(3) + Conv1d(k=1)
          └─ concat(4 分支) → BN → ReLU
       （每 3 个 module 加残差）
  └─ GAP → Flatten → Linear → (B, n_class)
```

| 模块 | 结构 | 作用 |
|---|---|---|
| `CNNBlock` | Conv+激活+AvgPool 堆叠 + Linear | 基线 |
| `InceptionModule` | bottleneck + 3 conv + maxpool → concat → BN | 核心 |
| `InceptionBlock` | depth 个 module + 每 3 个加残差 | 堆叠 |
| `InceptionTime` | block + GAP + FC | 入口 |

⚠️ **`CNNBlock` 默认逐层 `Sigmoid`**（paddlets 原样）⇒ **梯度消失，实测 accuracy 仅 0.367**；框架把 `activation` 改为**可配**（demo 用 ReLU 达 0.431）。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `(B, n_class)` logits（CNN 末层 Softmax ⇒ 框架 `_TsCE` 检测概率后走 `log+NLL`） |
| 标签 | 窗口内**多数投票**或末值 |
| 指标 | accuracy / macro-F1 / **每类 F1** |
| ⚠️ 双重 softmax | 输出已 Softmax 再喂 CE 会梯度极小 ⇒ 框架已修 |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
Architecture:
  task: ts_classify
  algorithm: inception_time
  in_chunk_len: 96
  num_features: 3
  num_classes: 3
  kernel_size: 41
  block_out_size: 32
  block_depth: 6
```

```bash
tkiln train -c configs/ts/classify_it_demo.yml -o Global.epoch_num=15
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **CNN 0.0048 M / **InceptionTime 0.0298 M**** |
| 对齐 | **CNNBlock **2.0e-14** / InceptionTime **6.7e-13**（fp64）** |
| 本机速度 | 合成 3 类 15 epoch < 1 分钟 |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0** |
| fp64 前向 | CNN 2.0e-14 / IT 6.7e-13 |
| 端到端（3 类，基线 0.333） | **InceptionTime 0.459** / CNN(ReLU) 0.431 / CNN(Sigmoid) 0.367 |

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| **工况/故障分类** | ✅ InceptionTime 首选 |
| 极简快速 | CNN + ReLU |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| 图像 `classify` | 那个吃图片；本任务吃 1D 序列 |
| `panns_cls` | 那个吃音频特征 |
| `ts_rul` | RUL 是回归；本任务分类 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **paddlets `_CNNBlock` 默认 Sigmoid** ⇒ 梯度消失（实测 0.367）
- ⚠️ **双重 softmax**：框架 `_TsCE` 已修（走 `log`+NLL）
- ⚠️ InceptionTime 核尺寸被强制成**奇数**
- ⚠️ `padding='SAME'` 在 stride=1 时等价 torch `padding=k//2`
