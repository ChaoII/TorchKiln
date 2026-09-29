# 车牌识别（plate_rec，CRNN + CTC）

> **定位**：检测到的车牌框 → 识别字符。**CRNN**（CNN + RNN + CTC），
> **0.75 MB**，单车牌短文本的经典解法。
> **任务**：`plate_rec` | **权重**：`plate_rec_color.pth`（0.75 MB，✅ 已托管 ModelScope）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition**（CRNN，Shi et al.）<br>arXiv:1507.05717 |
| 论文 2 | **Connectionist Temporal Classification**（Graves et al. 2006）—— CTC 的原始论文 |
| 上游 | PaddleOCR 的 rec 头（CRNN 系，CTC 头） |
| 本框架实现 | `torchkiln/ocr/`（rec 链路）+ `torchkiln/ocr/modeling/` |
| 权重 | `plate_rec_color.pth`（0.75 MB）→ ModelScope `pretrained/` ✅ |

### 要解决的问题

车牌识别有 **3 个约束**：

1. **变长文本**：车牌长度 6~7 位，边缘可能截断；
2. **不定长对齐**：图像宽度与字符数**不成比例**，无法直接回归字符序列；
3. **无标注对齐**：只知道整串文本，**不知道每个字符在图中的位置**。

传统做法（字符分类）需要逐字符裁剪标注，**成本高且无法处理变长**。

### 核心创新

**CRNN = CNN + RNN + CTC**，三者各解决一个约束：

| 组件 | 解决什么 | 机制 |
|---|---|---|
| **CNN** | 视觉特征提取 | 转置卷积 / 可分离卷积 → 特征图 `(B, C, H', W')` |
| **序列化** | **把图像变序列** | **宽度维 W' 当时间维** ⇒ `(B, W', C·H')`（高 H' 压成 1，保留字符的竖直结构） |
| **RNN** | 上下文建模 | 双向 LSTM/GRU ⇒ `(B, W', hidden)`，利用左右字符约束 |
| **CTC** | **免对齐** | 直接从序列 logits 训练出文本，**不需要字符级标注**；CTC 引入 **blank 空白符** 解决重复字符问题 |

> ⭐ **CTC 是关键**：它让「图像 → 文本」变成**端到端可训练**且**无需对齐标注**。
> 代价是 CTC 只保证**条件独立**，缺乏语言模型（后处理可加字典约束）。

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B, 3, 48, W)          # 高固定 48，宽按车牌长宽比
 │
 └─ CNN Backbone
      Conv(3→64,  k3, s2)  → BN → ReLU          H/2
      Conv(64→128,k3, s2)  → BN → ReLU          H/4
      Conv(128→256,k3,s2)  → BN → ReLU          H/8
      Conv(256→256,k3,s2)  → BN → ReLU          H/16
      Conv(256→256,k3,s2)  → BN → ReLU          H/32  → (B, 256, 1, W')
      (可选) MaxPool(2,1) 或 AdaptiveAvgPool2d((1, None))
 │
 └─ 序列化（Squeeze）
      (B, 256, 1, W')  →  squeeze(dim=2)  →  transpose  →  (B, W', 256)
      ★ 宽度维 W' 当时间维；高度维 1 保持字符竖直结构
 │
 └─ RNN
      BiLSTM(256 → hidden=256)  或  GRU
      → (B, W', 256)
 │
 └─ Linear(256 → n_char)     # n_char = 字符集大小 + 1(blank)
      → (B, W', n_char)        # ★ logits，**不是概率**
 │
 └─ CTC Loss:  F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
 │
 └─ 解码（推理）
      argmax → 去 blank → 折叠连续重复 → 截断
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| CNN Backbone | 5 层 Conv+BN+ReLU，逐层下采样 H | 提取视觉特征；**H 压到 1** |
| 序列化 | `squeeze(H) + transpose` | **图像 → 序列**（CRNN 的核心技巧） |
| RNN | **BiLSTM**（主流）或 GRU | 上下文；双向能利用**左右字符约束**（如「0/O」「1/I」） |
| 分类头 | `Linear(hidden → n_char)` | 逐时刻字符 logits |
| CTC 头 | `F.ctc_loss` | 免对齐损失 |
| 解码 | 贪心 argmax + 去 blank | 推理 |

### 2.3 为何「高固定 48，宽可变」

| 设计 | 原因 |
|---|---|
| **H 固定 48** | 统一批处理；字符高度在 48 像素足够分辨 |
| **W 按比例** | 车牌长宽比固定（≈4:1），宽度可随输入调整；`W'` 至少需 ≥ 字符数 |
| **W' ≥ 字符数** | CTC 要求 `input_length ≥ target_length + #repeats`；若 W' 太短会**无法解码** |
| **H 压到 1** | 保留字符的**竖直全貌**（一个字符的笔画是竖向的），比压成 1 维更利于区分 |

---

## 3. 输出与后处理

### 3.1 ⭐ 关键坑：CTC 的 log_softmax

这是**跨框架移植最常见的错误**，必须记住：

| 框架 | `ctc_loss` 内部行为 | 要求传入 |
|---|---|---|
| **`paddle.nn.functional.ctc_loss`** | **会做 `log_softmax`** | 传**原始 logits** |
| **`torch.nn.functional.ctc_loss`** | **不做** | 传**对数概率** |

⇒ 移植到 torch 时必须：

```python
# ✅ 正确
log_probs = F.log_softmax(logits, dim=2)      # ← 少了这步 loss 会变成负数
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                  blank=blank_idx, reduction='mean')

# ❌ 错误（直接喂 logits）
loss = F.ctc_loss(logits, ...)                 # 实测 loss = -852（应 +80.47）
```

> **实测记录**：未加 `log_softmax` 时 torch 侧 ctc_loss = **-852**，
> Paddle 侧 = **+80.47**；其余三分量（score/border/direction）完全一致
> ⇒ 精确定位到这一个问题。加上后**逐位一致**。

> ⚠️ **归一化口径**：`reduction='none'` 下 torch 返回**未归一**的值；
> `reduction='mean'` 会**再除以 target 长度**。跨框架对比时注意口径。

### 3.2 解码（CTC 贪心）

```
logits (B, T, C) → argmax(dim=-1) → (B, T) 字符索引序列
  ↓ ① 移除所有 blank（索引 = 字符集大小）
  ↓ ② 折叠**连续重复**（'AA' → 'A'）
  ↓ ③ 截断到实际长度
→ 文本
```

| 项 | 说明 |
|---|---|
| blank 索引 | 最后一个（`n_char - 1`） |
| confidence | 字符概率均值（或取 CTC 路径概率） |
| ⚠️ 局限 | **贪心**（无 beam search）；CTC 无语言模型 ⇒「0/O」「1/I/L」易混 |

### 3.3 完整两阶段（与检测串联）

```
图片
 └─ plate_det（见 plate_det.md）→ 车牌框
     └─ 逐框 crop + 缩放到 (48, W) + 归一化   ← CPU 瓶颈（见下）
         └─ plate_rec（本模型）→ 文本 + 置信度
```

**成本**（实测，v6_tiny 引擎）：

```
ms/图 = 23.9 + 1.669 × 框数
```

⇒ 加大 rec 的 batch **治不了**（1.669→1.555 ms，-7% 后持平），
因为瓶颈是 **CPU 逐框 crop+resize+归一化 的 Python 循环**，不是 GPU 前向
（v6_tiny rec GPU 前向 **<1 ms**）。

---

## 4. 配置与用法

### 4.1 最小配置

```yaml
Global:
  model_name: plate_rec
  pretrained_model: plate_rec_color      # 裸名，自动从 ModelScope 下载
  device: cuda:0
  epoch_num: 100
Architecture:
  task: ocr_rec
  num_classes: 1                         # ★ 车牌字符集大小
  Head:
    hidden_size: 256
    rnn_type: LSTM                       # 或 GRU
  PostProcess:
    img_height: 48                       # ★ 高度固定
    img_width: 192                       # 宽度基准
    batch_size: 6                        # 逐框 batch
    color: true                          # 彩色（vs 灰度）
Loss:
  type: ctc
Optimizer:
  name: Adam
  lr: {name: Linear, learning_rate: 0.001, warmup_epoch: 3}
Train:
  dataset:
    data_dir: datasets/plate_rec
  loader:
    batch_size_per_card: 64              # 一张图可产多个车牌样本
    num_workers: 4
```

### 4.2 四条链路

```bash
# 训练
tkiln train -c configs/plate/plate_rec.yml

# 评估（字符准确率 / 整牌准确率）
tkiln val -c configs/plate/plate_rec.yml --weights output/plate_rec/best_accuracy.pth

# 预测（单图）
tkiln predict -c configs/plate/plate_rec.yml --weights ... --input plate.jpg

# 导出
tkiln export -c configs/plate/plate_rec.yml --weights ... --onnx
```

### 4.3 字符字典

```yaml
# configs/plate/rec_dict.txt（每行一个字符）
京
津
冀
...
0
1
...
9
```

> ⚠️ 字典**最后一项是 blank**（索引 `n_char - 1`），加载时框架自动追加。

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| **权重大小** | **0.75 MB**（`plate_rec_color.pth`） |
| 参数（估） | ≈0.18 M（5 层 CNN + BiLSTM256） |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |
| **GPU 前向** | **<1 ms/框**（★ 不是瓶颈） |
| **CPU 预处理** | **1.555 ms/框**（★ **真正的瓶颈**） |
| 优化方向 | 多进程 crop / 换端到端模型 |

> ⚠️ 参数为按结构推算；精确值以 `tkiln check` 输出为准。

---

## 6. 公开指标

| 项 | 说明 |
|---|---|
| 官方 CRNN 论文 | SVHN 96.8% / MNIST 98.6%（**不是本任务**，仅供参考） |
| 本框架 | 车牌数据集为业务私有（`datasets/` 不入库） |
| 指标 | **整牌准确率**（最重要）/ 字符准确率 / 编辑距离 |
| 口径纪律 | ⚠️ 车牌区分「完整准确率」与「字符准确率」；**评估时必须同时报两个** |

> ℹ️ 本框架**未对 plate_rec 做跨框架逐位对齐**（业务定制 + 字符集私有），
> 故不列对比数字 —— **不编造未验证的数据**。
> 但 **CTC 的 log_softmax 移植坑已在 PGNet 端到端模型上被逐位验证**
> （修后 loss 349.08233642578125 ↔ Paddle 侧同值）。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| **固定车牌识别** | ✅ 本模型 | 0.75 MB，CTC 免对齐，够用 |
| 车牌字符集固定不变 | ✅ 优势最大 | 不需字典更新 |
| 需要区分易混字符（0/O、1/I） | ⚠️ 加后处理约束 | CTC 无语言模型 |
| 长文本 / 通用 OCR | ❌ 用 PP-OCRv6_rec | 字典大得多、SVTR 更强 |
| 与检测串联 | ✅ 本模型 | 但注意 CPU crop 瓶颈 |
| 极高精度要求 | 考虑加字典约束 + 二阶段判别 | 纯 CTC 有天花板 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| **vs 字符分类**（逐字裁剪） | 需逐字符标注；CRNN 只需整串标注，**省 6~7 倍标注成本** |
| **vs Attention-based seq2seq** | 需 attention / 更复杂；CRNN+CTC 更快更稳 |
| **vs PP-OCRv6_rec（SVTR）** | v6 用**单结构 SVTR**（无 RNN），精度更高但更大（4.38 MB vs 0.75 MB） |
| **vs PGNet 端到端** | PGNet 不需裁剪（高框密度快 9~15×）；但需重训（150 epoch） |
| **vs Transformer OCR** | 精度可能更高，但 CTC 的**免对齐**优势对短文本足够 |

---

## 9. 已知问题 / 注意事项

| 项 | 说明 |
|---|---|
| ⭐ **CTC 的 log_softmax** | **torch 必须先 `F.log_softmax` 再进 `F.ctc_loss`**；否则 loss 变负（实测 -852 vs +80.47）。详见 §3.1 |
| ⚠️ **`reduction` 口径** | `'none'` 未归一、`'mean'` 会再除 target 长度；跨框架对比需统一 |
| ⚠️ **CTC 无语言模型** | 「0/O」「1/I/L」易混 ⇒ 建议加字典/规则后处理 |
| ⚠️ **贪心解码** | 未实现 beam search；长文本精度受限 |
| ⚠️ **`W' ≥ 字符数` 是硬约束** | 宽度太小会导致 CTC 无法解码（`input_length < target_length`） |
| ⚠️ **输入高度固定 48** | 换高度需重训（RNN 的 `input_size` 随高度变化） |
| ⚠️ **换字符集要重训** | `Linear` 末层维度 = `n_char`，字典变更 ⇒ 分类头形状变化 |
| ⚠️ **颜色 vs 灰度** | `color: true` 权重不能与灰度权重混用（第一层输入通道不同） |
| ⚠️ **CPU crop 是瓶颈** | 1.555 ms/框；优化用多进程或换端到端 |
| ⚠️ **blank 位置** | 必须是**最后一个索引**；`num_classes` 需含 blank |
