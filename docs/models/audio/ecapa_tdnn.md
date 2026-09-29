# ECAPA-TDNN

> **定位**：说话人验证（Speaker Verification）的 SOTA 级 TDNN 架构；用
> **Res2Net 多尺度 + Squeeze-Excitation + 多层特征聚合（MFA）+ 注意力统计池化（ASP）**
> 把变长 log-fbank 压成固定维 **说话人 embedding**。
> **任务**：`speaker`（说话人验证 / embedding 提取）
> **权重**：`weights/ecapa_tdnn_voxceleb12.pth`（79.5 MB，由 PaddleSpeech `sv0_ecapa_tdnn_voxceleb12` 转换）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification** |
| arXiv | **2005.07143**（2020-05-14，已核实标题/作者） |
| 作者 | Brecht Desplanques、Jenthe Thienpondt、Kris Demuynck（Aalto University / Naver） |
| 会议 | Interspeech 2020 |
| 官方代码 | https://github.com/speechbrain/speechbrain（`speechbrain/spkrec-ecapa-voxceleb`） |
| 权重来源 | PaddleSpeech `sv0_ecapa_tdnn_voxceleb12_ckpt_0_2_0.tar.gz`（266MB，取其中 `model.pdparams` 88.8MB） |
| 本框架实现 | `torchkiln/audio/ecapa_tdnn.py`（`TDNNBlock/Res2NetBlock/SEBlock/AttentiveStatisticsPooling/EcapaTdnn`） |
| 移植方式 | PyTorch 逐命名移植 **PaddleSpeech** `vector/models/ecapa_tdnn.py` + 权重逐键对齐（missing=0/unexpected=0） |

### 论文要解决的问题

1. **x-vector TDNN 的通道信息未被强调**：普通 TDNN 对每个通道一视同仁，
   说话人判别性的通道没有被强调（→ **SE 通道注意力**）；
2. **单尺度时域建模不足**：TDNN 只用一个 kernel 尺寸（→ **Res2Net 多尺度**）；
3. **只用最后一层特征**：浅层细节被丢（→ **MFA 多层特征聚合**）；
4. **统计池化没利用局部上下文**（→ **ASP 注意力统计池化**，含 global context）。

### 关键结论（论文 VoxCeleb1-O EER）

| 系统 | 参数量 | EER |
|---|---|---|
| x-vector（基线） | ~4.1M | **3.10%** |
| **ECAPA-TDNN（C=1024, 4 blocks）** | **~20.8M** | **0.86%** |
| ECAPA-TDNN + 数据增广 | — | **0.80%** |

> ⇒ 参数量 ~5×，EER 降低 ~3.6×。

---

## 2. 网络结构

### 2.1 整体框图

```
log-fbank (N, 80, T)   @16 kHz，25ms 窗 / 10ms 帧移
  ▼
blocks[0] = TDNNBlock(80 → 1024, k=5, d=1)                     # conv→ReLU→BN
blocks[1] = SERes2NetBlock(1024 → 1024, k=3, d=2)
   · tdnn1(k1) → Res2NetBlock(scale=8, d=2) → tdnn2(k1) → SEBlock(se=128) → + residual
blocks[2] = SERes2NetBlock(1024 → 1024, k=3, d=3)
blocks[3] = SERes2NetBlock(1024 → 1024, k=3, d=4)
  ▼  收集每层输出 xl = [x0, x1, x2, x3]
MFA: x = cat(xl[1:], dim=1)        # ★ 排除第一层 → 3×1024 = 3072
     x = TDNNBlock(3072 → 3072, k=1, d=1)
  ▼
ASP: AttentiveStatisticsPooling(channels=3072, attention_channels=128, global_context=True)
     · 用 (x, mean, std) 三联 → TDNN(9216→128) → tanh → Conv1d(128→3072) → softmax
     · 输出 concat(mean, std) → (N, 6144, 1)
  ▼
asp_bn = BatchNorm1d(6144)
fc = Conv1d(6144 → 192, k=1)       # ★ 用 k=1 Conv1d（不是 Linear）
  ▼
说话人 embedding (N, 192)
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | PaddleSpeech 名 |
|---|---|---|---|---|
| **`Conv1d`** | `nn.Conv1d(padding=..., **padding_mode="reflect"**)` | — | ★ **反射填充**（关键坑） | `Conv1d` |
| `BatchNorm1d` | `nn.BatchNorm1d(momentum=1−0.9=0.1)` | — | Paddle `momentum=0.9` ≡ torch `0.1` | `BatchNorm1d` |
| **`TDNNBlock`** | `conv → activation → norm`（**顺序与 Paddle 一致**） | — | 基本块 | `TDNNBlock` |
| **`Res2NetBlock`** | `chunk(scale=8)` + 逐级累加（`i==0` 直通、`i==1` 过 TDNN、`i≥2` 先加 `y_i` 再过） | — | ★ 多尺度 | `Res2NetBlock` |
| `SEBlock` | `conv1(1×1) → ReLU → conv2(1×1) → Sigmoid → ×x`；支持按 lengths 掩码 | — | ★ 通道注意力 | `SEBlock` |
| **`AttentiveStatisticsPooling`** | `(x, mean, std)` → TDNN → tanh → Conv → **softmax（屏蔽 padding 用 `−inf`）** → 加权 mean/std → concat | — | ★ 注意力池化 | `AttentiveStatisticsPooling` |
| `SERes2NetBlock` | `tdnn1 → Res2Net → tdnn2 → SE → + residual` | — | 主干块 | `SERes2NetBlock` |
| `EcapaTdnn` | blocks + mfa + asp + asp_bn + fc | (N,80,T)→(N,192) | ★ 整网 | `EcapaTdnn` |

### 2.3 与同类的差异

| 对比对象 | 差异点 | 为什么 | 效果 |
|---|---|---|---|
| **x-vector TDNN** | ECAPA 加 **SE + Res2Net + MFA + ASP** | 通道/多尺度/多层/注意力四重增强 | EER 3.10% → 0.86% |
| **只做 ASP（无 MFA）** | ECAPA 用 `cat(blocks[1:])` 而非最后一层 | 浅层也可用 | 提升显著 |
| **`fc` 用 Conv1d 而非 Linear** | PaddleSpeech 用 `k=1` Conv1d | 与 Paddle 对齐 | 权重 1:1 |
| **Paddle Conv1d 填充** | **`padding_mode="reflect"`（镜像，非零）** | PaddleSpeech 默认 | 漏掉第一层就差 13% |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | **`(N, 192)` 说话人 embedding**（不是分类 logits！） |
| 损失（本框架/③ 对拍） | 用 `MSE(embedding, 固定 target)` 验证梯度通路；真实训练需加 **AAM-softmax / 分类头** |
| 指标（④ 口径） | **同音频 embedding 的 maxdiff + 余弦相似度**（不是 accuracy） |
| 后处理 | 说话人验证通常：**embedding 余弦相似度** 与阈值比较（或 PLDA 打分） |
| 特征（必须一致） | `paddlespeech.audio.compliance.librosa.melspectrogram`：`sr=16000, window_size=400`（**同时是 FFT size 与窗长 → n_fft=400**）、`hop=160`、`n_mels=80`、`fmin=50`、`fmax=None(=8000)`、hann、`center=True`、`pad_mode="reflect"`、`power=2.0`、`to_db=True(ref=1.0, amin=1e-10, top_db=None)` |

torch 侧复刻（④ 用，rel **4.73e-06**）：

```python
torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160,
    window_fn=torch.hann_window, center=True, pad_mode="reflect",
    power=2.0, n_mels=80, f_min=50, f_max=8000, norm="slaney", mel_scale="slaney")
+ 10 * log10(clamp(mel, 1e-10))
```

---

## 4. 配置与用法

本框架目前**没有 `speaker` 的完整训练配置**（见 §9）：`EcapaTdnn` 作为模型可直接构建/加载/
前向/反传，但**任务适配器与 YAML 尚未落地**（当时因数据集/训练脚本缺失而只做对齐验证）。

```python
from torchkiln.audio import EcapaTdnn, ECAPA_TDNN_CONF
m = EcapaTdnn(input_size=80)
x = torch.randn(2, 80, 200)          # (N, 80, T) log-fbank
emb = m(x)                            # (N, 192)
```

```python
# 权重（Paddle .pdparams -> torch .pth 的转换规则，共 3 条）
# 1) 去掉 backbone. 前缀
# 2) BN 的 ._mean/._variance -> .running_mean/.running_var
# 3) 顶层 weight(192, 7205) 是 wrapper 的说话人分类头 -> 不加载（预期）
```

---

## 5. 规模与速度

| 项 | 值 | 来源 |
|---|---|---|
| 参数（backbone，`input_size=80`） | **20.7676 M** | 本框架实算（论文 ~20.8M） |
| 权重体积 | **79.5 MB**（`.pth`） | AGENTS 记录 |
| Paddle 原始包 | 266MB（含 `model.pdopt` 177MB 优化器状态，**不需要**） | AGENTS 记录 |
| 推理耗时（本机） | **未统计** | — |
| 训练显存 | **未统计** | — |

---

## 6. 公开指标

### 6.1 ⭐ 四条对齐全过

| # | 项 | 结果 |
|---|---|---|
| ① | 权重加载 | **missing=0 / unexpected=0**；200/201 键（第 201 个是 wrapper 分类头 `weight(192,7205)`，**预期不加载**） |
| ② | 逐层前向（**fp64**） | `blocks0` **1.47e-15**（精度极限）；`blocks1~3` 8e-12~1.8e-11；`mfa` **1.38e-11**；`asp` 5.5e-9；`fc` 1.43e-8；**embedding 余弦 = 1.0000000000** |
| ③ | 单步 loss/梯度（**fp64**） | loss rel **1.67e-08**；embedding rel 1.43e-08；可比梯度 **138/138**；非零梯度最差 rel **1.34e-07**（`mfa.norm.norm.weight`） |
| ④ | 端到端（同音频 → fbank → embedding） | 相对 maxdiff **0.001010**；**余弦 0.9999996424**；norm 123.0696 vs 123.0715 → **PASS** |

### 6.2 ⚠️ ② 的 FAIL 根因：**PaddleSpeech Conv1d 默认 reflect 填充**

- **症状**：`blocks0`（仅 conv→ReLU→BN）第一层就 **rel = 1.34e-1**，逐层放大到 4.3e-1；
  embedding 余弦 0.999667。
- **排查顺序（可复用）**：① 先证权重加载无误（`blocks.0.conv.conv.weight` maxdiff **0.00e+00**、
  键数 200 全匹配，脚本只需 paddle 不必装 torch）；② 输入 maxdiff=0.00e+00
  ⇒ 差异只可能在算子 → 读回源码发现第 49 行 **`padding_mode="reflect"`**，
  而 `_manage_padding` 用 `F.pad(x, padding, mode=self.padding_mode)` ⇒ **是镜像填充，不是零填充**。
- **修复**：torch 侧 `nn.Conv1d(..., padding_mode="reflect")`。修后：
  `blocks0` 1.34e-1 → **2.57e-4（↓500×）**、`mfa` 4.29e-1 → 9.99e-4、
  输出 3.90e-2 → **1.08e-3**、余弦 **0.99999988**。
- 残余 ~1e-3 由 **fp64 判定法**确认是 fp32 舍入（② fp64 = 1.47e-15）。

### 6.3 ⚠️ ③ 的判定陷阱：**ASP 的 bias 梯度恒为 0**

`asp.conv.conv.bias` 的 `|grad|max` 在 fp64 是 **8.88e-16**（fp32 是 4.77e-6）
—— **数学上恒为 0**：ASP.conv 的输出进 softmax，**bias 的加性常数被 softmax 平移不变性抵消**。

> 用 `rel = maxdiff / max|pg|` 会因分母过小**虚高到 8.5 / 1.41**（会误判 FAIL）。
> **正确做法**：`denom < 1e-10` 时改用**绝对误差判据**。

---

## 7. 选型建议

| 场景 | 建议 | 理由 |
|---|---|---|
| 说话人验证 / 声纹 | **本模型（ECAPA-TDNN）** | EER 0.86%（VoxCeleb1-O） |
| 说话人 **embedding** 提取 | 直接用（`(N,192)`） | 是通用声纹特征 |
| 极轻量设备 | x-vector / 缩小 `channels` | 本模型 20.8M 偏大 |
| 训练（本框架内） | ⚠️ **暂不可**（无任务适配器/配置） | 见 §9 |
| 需要分类头 | 在 `fc` 后接分类（本框架未内置） | 原始权重含 wrapper 分类头 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs x-vector** | ECAPA EER 0.86% vs 3.10% |
| **vs SpeechBrain 官方** | 本框架移植的是 **PaddleSpeech** 版；命名天然一致（`blocks.0.conv.conv.weight` ↔ Paddle 同名） |
| **vs PaddleSpeech** | ①②③④ 全对齐（fp64 1.47e-15 / loss 1.67e-08 / 余弦 0.9999996） |
| **④ 判据说明** | 输出是 192 维 embedding（norm≈123），**不能**照搬分类模型的「绝对 maxdiff ≤ 0.02」；应用 **余弦 + 相对 maxdiff** |

---

## 9. 已知问题 / 注意事项

- ⚠️ **PaddleSpeech `Conv1d` 默认 `padding_mode="reflect"`**（镜像填充，**不是零填充**）
  —— 这是 ② 的 FAIL 根因。本框架已设 `padding_mode="reflect"`，**不要改成默认 zeros**。
- ⚠️ **残差集中在 ASP**：含 `sqrt(clamp(var, 1e-12))`（`d(sqrt)/dvar` 对小方差可达 1e5 倍）
  + softmax → **数值敏感点，非逻辑 bug**；前半段（conv/BN/Res2Net/SE/MFA）全 1e-11 级已证正确。
- ⚠️ **零梯度参数要用绝对判据**：`asp.conv.conv.bias` 梯度**数学上恒为 0**
  （softmax 平移不变性），相对误差会虚高。
- ⚠️ **BatchNorm1d(momentum=0.9) ≡ torch momentum=0.1**（都保留 90% 旧统计量）；
  写错会改变训练动力学（eval 下用 running stats，不影响对拍）。
- ⚠️ **PaddleSpeech 的 `TDNNBlock.forward(x)` 不接受 `lengths`**；
  `EcapaTdnn.forward` 的 `try/except` 就是为此而设。
- ⚠️ **`EcapaTdnn.forward` 的 MFA 是 `cat(xl[1:])` —— 排除第一层**
  （concat blocks[1..3] = 3×1024 = 3072）；写成 `xl` 全量会维度不符。
- ⚠️ **本任务只有「模型 + ④ 对拍」，没有训练**：
  - PaddleSpeech wheel **不带 `vector/exps`**（训练脚本缺失）；
  - VoxCeleb 数据集 **~300GB 且无可用直链**；
  ⇒ **无法做数据集级训练对齐**（AGENTS 已记录为阻塞）。
- ⚠️ 本框架**未集成 FLOPs 统计**。
