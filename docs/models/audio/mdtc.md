# MDTC (KWS, 关键词唤醒)

> **定位**：小足迹**关键词唤醒**（Keyword Spotting）模型；用**多尺度膨胀时序卷积**
> （Multi-scale Dilated Temporal Convolution）堆叠 12 个残差块，逐帧输出 0/1 命中概率。
> **任务**：`kws`（关键词唤醒）
> **权重**：`weights/mdtc_heysnips.pth`（**240.17 KB**，由 PaddleSpeech `mdtc_heysnips` 转换）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | ⚠️ **不确定**：本仓库**未能核实** MDTC 对应的论文 |
| 说明 | PaddleSpeech 的 `kws/models/mdtc.py` 头注释只写 **"Modified from wekws (wenet-e2e/wekws)"**，版权归 **Jingyong Hou**；仓库与 arXiv 均未标论文 |
| arXiv 检索 | 在 arXiv 检索 `MDTC` / `dilated temporal convolution + keyword spotting` / `multi-scale dilated temporal` **均未找到对应论文** |
| 官方代码 | https://github.com/wenet-e2e/wekws （PaddleSpeech 从其改写） |
| 权重来源 | PaddleSpeech `kws0_mdtc_heysnips_ckpt.tar.gz`（0.1MB） |
| 本框架实现 | `torchkiln/audio/mdtc.py`（`DSDilatedConv1d/TCNBlock/TCNStack/MDTC/KWSModel`）<br>`torchkiln/audio/kws_loss.py`（`max_pooling_loss` 等） |
| 移植方式 | PyTorch 逐命名移植 **PaddleSpeech** + 权重逐键对齐（missing=0/unexpected=0） |

> ⚠️ **按仓库纪律**：查不到论文就写「不确定」，**不编造 arXiv 号**。
> 需要引用时请以 **wekws 仓库**为准。

### 要解决的问题

1. **关键词唤醒必须极轻量**（常驻设备、低功耗）：MDTC 参数量仅 **0.034 M**（34 K）；
2. **关键词在时间轴上位置不定**：用 `max_pooling_loss`（对命中帧做 max 池化）解决
   「不知道关键词在哪一帧」的问题，而**无需**帧级对齐标注；
3. **因果（causal）推理**：流式唤醒要求只用过去帧 → 模型默认 `causal=True`。

### 关键结论（HeySnips）

| 数据集 | 指标 | 原文 |
|---|---|---|
| HeySnips | — | ⚠️ **本仓库未记录**论文/官方数字（也未找到论文），故**不填具体数值** |
| — | 参数量 | **0.034401 M**（本框架实算） |

---

## 2. 网络结构

### 2.1 整体框图

```
kaldi-fbank (N, T, 80)   @16 kHz，25ms 窗 / 10ms 帧移，n_mels=80
  │  ★ 注意布局是「帧在前」：T 在 dim1（详见 §9）
  │  pad：causal → 时间维左侧补 R=184 帧（torch: F.pad(x, (0,0,R,0))）
  ▼  transpose(1,2) → (N, 80, T+R)
preprocessor = TCNBlock(80 → 32, k=5, d=1, causal)        # receptive_fields 4
  ▼  (N, 32, T+180)
blocks = 3× TCNStack(in=32, stack_num=4, stack_size=1, res=32, k=5, causal)
  每个 TCNStack 内 = 4 个 TCNBlock，dilations = [1,2,4,8]
  ▼  (N, 32, T')   逐 stack 输出 → 对齐长度后 **求和**
  ▼  transpose(1,2) → (N, T', 32)；返回 (out, None) 元组（kaldi 风格）
KWSModel: linear(32 → num_keywords=1) → Sigmoid
  ▼  (N, T', 1)   逐帧命中概率
```

- `receptive_fields = 184`；`stack_num=3`、`stack_size=4`、`res_channels=32`、`kernel_size=5`。

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | PaddleSpeech 名 |
|---|---|---|---|---|
| **`DSDilatedConv1d`** | depthwise `Conv1d(groups=C, k, dilation, padding=0)` → BN → **pointwise 1×1** | — | ★ 深度可分离膨胀卷积（**padding=0，由上游手动 pad**） | `DSDilatedConv1d` |
| **`TCNBlock`** | `conv1(DSDilated) → bn1 → relu1 → conv2(1×1) → bn2`；残差：输入做**切片** `inputs[:, :, R:]`（causal）或 `[:, :, h:-h]`（非 causal）；`in==res` 才相加 → `relu2` | — | ★ 残差时序块 | `TCNBlock` |
| `TCNStack` | `dilations = [2**l for l in range(stack_num)]`（本配置 4 个：1,2,4,8）；`nn.Sequential(TCNBlock...)` | — | 多尺度堆叠 | `TCNStack` |
| **`MDTC`** | `preprocessor` + `stack_num=3` 个 `TCNStack` | (N,T,80)→(out,None) | ★ 主干 | `MDTC` |
| `KWSModel` | `MDTC` + `nn.Linear(hidden→num_keywords)` + `Sigmoid`；**解包 `outputs, _ = backbone(x)`** | →(N,T',1) | ★ 整模型 | `KWSModel` |

- ⚠️ **`MDTC.forward` 返回 `(outputs, None)` 元组**；torch 版 `KWSModel.forward`
  已**正确解包**（`outputs, _ = self.backbone(x)`）。
- ⚠️ **实证：PaddleSpeech 官方 `KWSModel.forward` 是坏的** —— 它写
  `outputs = self.backbone(x)` 后直接 `self.linear(outputs)`，
  而 `backbone` 返回元组 → `ValueError: linear(): argument 'X' must be Tensor, but got tuple`。
  （权重里含 `linear.*` 说明它**本应可用**，属官方代码 bug，不是我们改坏的。）

### 2.3 与同类的差异

| 对比对象 | 差异点 | 为什么 | 效果 |
|---|---|---|---|
| **TC-ResNet / TENet** | 用普通/多分支时序卷积 | 感受野受限 | MDTC 用**膨胀**扩大感受野 |
| **DS-CNN（KWS）** | 标准深度可分离 2D 卷积 | 无时序膨胀 | MDTC 是 **1D 因果** |
| **CRNN / RNN(GRU)** | 递归建模 | 序列长、慢 | TCN 可并行 |
| **Paddle vs torch** | Pad 语义 | 见 §9 | 已验证等价 |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `(N, T', num_keywords)` 逐帧 **命中概率**（已 sigmoid） |
| 判定 | 概率 > 0.5 → 该帧命中（④ 的 HIT/filler 口径） |
| 损失 | **`max_pooling_loss`**（`torchkiln/audio/kws_loss.py`）：<br>· **命中关键词** → **max-pooling**（padding 置 0）→ `-log(max)`；<br>· **其它 / filler** → **min-pooling**（padding 置 1）→ `-log(min)`；<br>· 均 `clip(1e-8, 1)`，再除以 `num_utts`；返回 `(loss, num_correct, acc)` |
| 为何用 max/min 池化 | 关键词出现位置未知 → 对**命中帧取最大**、对**非命中帧取最小**，无需帧级对齐 |
| 特征 | kaldi-fbank：`sr=16000, frame_length=25ms, frame_shift=10ms, n_mels=80` |
| torch 侧对拍参数 | `torchaudio.compliance.kaldi.fbank(num_mel_bins=80, frame_length=25.0, frame_shift=10.0, sample_frequency=16000.0, dither=0.0, energy_floor=1.0, snip_edges=True, low_freq=20, high_freq=0, window_type='povey')`；`remove_dc_offset=True`（= Paddle `zero_mean_windows`） |
| 后处理 | 逐帧阈值 0.5（或滑窗投票）；本框架未额外封装 post-process |

---

## 4. 配置与用法

模型配置（来自 PaddleSpeech `conf/mdtc.yaml`，已硬编码到 `torchkiln/audio/mdtc.py`）：

```python
MDTC_CONF = dict(num_keywords=1, stack_num=3, stack_size=4, in_channels=80,
                 res_channels=32, kernel_size=5, causal=True, receptive_fields=184)
KALDI_FBANK_CONF = dict(sample_rate=16000, frame_length_ms=25, frame_shift_ms=10, n_mels=80)
```

```python
from torchkiln.audio import build_mdtc
m = build_mdtc()                 # KWSModel
x = torch.randn(2, 120, 80)      # ★ (N, T, 80) 帧在前！
y = m(x)                         # (2, T', 1) 命中概率
```

```python
# 权重（Paddle .pdparams -> torch .pth 的转换规则，共 3 条）
# 1) 键结构本就一致（KWSModel = backbone.* + linear.*），无需加任何前缀
# 2) BN：._mean/._variance -> .running_mean/.running_var
# 3) linear.weight (32,1) -> .T (1,32)     # Paddle Linear [in,out]
#    depthwise conv (80,1,5) 同形，不转
```

> ⚠️ 本框架**尚无 `kws` 的完整训练配置**（数据集 HeySnips 无公开直链，见 §9）；
> 目前提供的是「模型 + loss + 四条对拍」。

---

## 5. 规模与速度

| 项 | 值 | 来源 |
|---|---|---|
| 参数 | **0.034401 M（34.4 K）** | 本框架实算 |
| 权重体积 | **240.17 KB** | AGENTS 记录 |
| Paddle 原始 | 236 键（可转换 236/275，39 个是 `num_batches_tracked`） | AGENTS 记录 |
| 推理耗时（本机） | **未统计** | — |
| 训练显存 | **未统计** | — |

---

---

---

---

---

### 📊 FLOPs（实测）

| 项 | 值 |
|---|---|
| **FLOPs** | **12.390 MFLOPs** |
| **MACs** | **6.195 MMACs** |
| 参数量 | **0.034 M** |
| 输入规格 | `1s @ 16kHz -> (1,98,80)` |
| 测量工具 | `torch.utils.flop_counter.FlopCounterMode`（PyTorch 内置） |
| 复现脚本 | `_downloads/flops_measure*.py` |

> **口径**：`FLOPs` 是乘加各计 1 次（×2），**与 ultralytics 官方表的 GFLOPs 同口径**
> （已由 yolo11/v8 十个模型逐个吻合验证，见 [`_FLOPS.md`](_FLOPS.md)）；
> `MACs = FLOPs / 2`。
> ⚠️ `FlopCounterMode` **不计自定义算子**（NMS / probiou / iSTFT 等后处理）⇒
> 此处是**网络主干**的 FLOPs。
## 7. 选型建议

| 场景 | 建议 | 理由 |
|---|---|---|
| 极轻量关键词唤醒（MCU/手机） | **本模型（MDTC）** | 34 K 参数、240 KB |
| 流式（低延迟） | `causal=True`（默认） | 只用过去帧 |
| 多关键词 | `num_keywords > 1` + 改 `max_pooling_loss` | 本配置 `num_keywords=1` |
| 需要更高精度 | 换更大 KWS 模型（如 TENet/DS-CNN） | MDTC 主打小 |
| 训练（本框架内） | ⚠️ **暂不可**（HeySnips 无公开直链） | 见 §9 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs wekws** | 本模型**源自 wekws**（PaddleSpeech 从其改写）；命名逐层对齐 |
| **vs PaddleSpeech** | ①②③ 全过、④ 模型端过（logits 4e-09） |
| **vs TC-ResNet/TENet** | 都轻量；MDTC 用膨胀卷积扩感受野 |
| **vs 官方 `KWSModel.forward`** | **官方有 tuple bug**，本框架已修（解包 `outputs, _`） |
| **④ 特征口径** | 两版 kaldi 移植有 1.09% 差异（fp64 已证非舍入） |

---

## 9. 已知问题 / 注意事项

- ⚠️ **输入布局是 `(N, T, 80)`（帧在前），不是 `(N, 80, T)`**！
  实测依据：给 `(2,80,120)` 报 `The channel of input must be divisible by groups, channel is 120`；
  给 `(2,120,80)` 才 OK。原因：`MDTC.forward` 先 pad 再 `transpose(1,2)`，
  若给 `(N,80,T)` 则 T 变通道 → 报错。
- ⚠️ **`F.pad` 的 6 值语义**：Paddle `F.pad(x, (0,0,R,0,0,0))` 的 6 值**从最内层维度开始配对**
  ⇒ C=(0,0)、T=(R,0)（时间维左补 R，causal）、N=(0,0)；等价 torch 写法是
  **`F.pad(x, (0,0,R,0))`**（torch pad 也从最后一维开始）。② 已证 **maxdiff = 0.00e+00**。
- ⚠️ **`MDTC.forward` 返回 `(outputs, None)` 元组**；调用方**必须解包**。
- ⚠️ **官方 `KWSModel.forward` 有 tuple bug**（对元组调 `Linear`）—— 本框架已修。
- ⚠️ **`linear.weight` 布局**：Paddle `(32,1)` vs torch `(1,32)` → 加载/对拍前**必须 `.T`**。
- ⚠️ **④ 特征 1.09% 差异**：两版 kaldi fbank 移植的实现差异（fp64 已证非舍入）；
  **不影响模型结论**（logits 4e-09）。
- ⚠️ **HeySnips 数据集无公开直链**（原需申请，常见镜像已 404）
  ⇒ **无法做数据集级训练对齐**；本框架**无 `kws` 训练配置**。
- ⚠️ `max_pooling_loss` 的细节：`m[:min_duration] = True` 是**原地修改**（与 Paddle 一致）。
- ⚠️ 本框架**未集成 FLOPs 统计**。
