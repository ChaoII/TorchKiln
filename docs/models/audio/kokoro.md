# kokoro-82M

> **定位**：**82M 参数的高质量 TTS**（Text-to-Speech），架构 = **PL-BERT 文本编码器 +
> Prosody Predictor（时长/F0/能量）+ iSTFTNet 声码器**，用 **style 向量**控制音色与韵律。
> 属于 **StyleTTS2 风格**的非自回归 TTS。
> **任务**：`kokoro_tts`（`Architecture.task: kokoro_tts`）
> **权重**：`kokoro-v1_0.pth`（英文/多语言，327.21 MB）、`kokoro-v1_1-zh.pth`（中文，327.25 MB）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | ⚠️ **无官方论文**：kokoro-82M 是 **HuggingFace 上的模型发布**（hexgrad），不是论文 |
| arXiv | ⚠️ **不确定 / 不适用**：在 arXiv 检索未找到 kokoro 的 TTS 论文（仓库**未记录**编号） |
| 官方仓库 | https://github.com/hexgrad/kokoro （**仅推理代码，MIT**）；G2P 见 https://github.com/hexgrad/misaki |
| 权重来源 | **ModelScope** `hexgrad/Kokoro-82M`（`kokoro-v1_0.pth`）与 `hexgrad/Kokoro-82M-v1.1-zh`（`kokoro-v1_1-zh.pth`） |
| 本框架实现 | `torchkiln/audio/kokoro/`（`model.py` 149 行 / `istftnet.py` 422 / `modules.py` 184 / `custom_stft.py` 198，**保留原版权头**）+ `kokoro_loss.py`、`kokoro_dataset.py`、`torchkiln/tasks/kokoro_tts.py` |
| 移植方式 | **官方就是 PyTorch** ⇒ 只做「归位相对导入 + 去 HF 回退 + `loguru`→`logging`」；对齐基准 = **官方 pip 包** `kokoro==0.9.4` |

> ⚠️ 与仓库其它音频模型（PANNs/ECAPA/MDTC 是 Paddle→torch 跨框架）不同：
> **kokoro 是纯 torch** ⇒ ① 的基准改为「与官方 pip 包逐层对拍」。

### 要解决的问题

① 自回归 TTS 慢（逐 token 生成）→ 用**非自回归 + 时长预测 + iSTFTNet**，一次前向出整条波形；
② 音色/风格可控 → **style 向量**（`ref_s`，256 维）注入 predictor 与 decoder；
③ 模型要小 → 82M / 327MB fp32，可端侧部署。

### 关键结论

| 项 | 值 |
|---|---|
| 参数量 | **81.7634 M**（与权重文件大小吻合） |
| 采样率 | **24 kHz** |
| 词表 | v1.0 = **114** / v1.1-zh = **171**（两版架构**完全相同**，仅 vocab 不同） |
| 质量 | ⚠️ **本仓库未记录**官方 MOS/评测数字，**不编造** |

---

## 2. 网络结构

### 2.1 整体框图（推理链）

```
input_ids (B, T)  （音素 id；首尾各加 0=pad）
  ▼ bert = CustomAlbert（PL-BERT，ALBERT，hidden 768，12 层，max_pos 512）→ d_en = Linear(768→512)
  ├─ s = ref_s[:, 128:]                     ← 给 predictor 的 style
  ▼ predictor.text_encoder = DurationEncoder(d_model=512)；predictor.lstm = BiLSTM(512+128→256×2)
  ▼ duration_proj = Linear(512→max_dur=50) → duration = sigmoid(...).sum(-1)/speed
      pred_dur = round(duration).clamp(min=1)      # ★ 每个音素持续几帧
  ▼ repeat_interleave 建对齐矩阵；en = d_en.T @ pred_aln_trg
  ▼ predictor.F0Ntrain(en, s) → F0_pred, N_pred     # F0/能量曲线
  ▼ text_encoder = 5×[Conv1d(k=5)+LayerNorm+LeakyReLU+Dropout] + BiLSTM → asr = t_en @ pred_aln_trg
  ▼ decoder = iSTFTNet(asr, F0_pred, N_pred, ref_s[:, :128]) → audio (B,L) @24kHz
```

### 2.2 iSTFTNet Decoder / Generator

```
Decoder(dim_in=512, style_dim=128, dim_out=80)：encode = AdainResBlk1d(512+2→1024)；
  decode[0..2] = AdainResBlk1d(1024+2+64→1024)，decode[3] = (…→512, upsample)；
  F0_conv / N_conv = weight_norm(Conv1d(1,1,k3,s2))    # 2× 下采样；asr_res = Conv1d(512→64,k1)
  generator = SourceModuleHnNSF(谐波源, harmonic_num=8) → 2× weight_norm(ConvTranspose1d)
      → AdaINResBlock1(kernels[3,7,11], dilations[1,3,5]×3) + noise 分支
      → iSTFT(n_fft=20, hop=5) → 波形
```

### 2.3 逐模块说明（本框架实现文件）

| 模块 | 结构 | 作用 | 关键点 |
|---|---|---|---|
| **`KModel`** | bert + bert_encoder + predictor + text_encoder + decoder | ★ 整模型 | `forward_with_tokens(input_ids, ref_s, speed)` |
| `CustomAlbert` | 继承 `transformers.AlbertModel` | PL-BERT 文本编码 | **本体**（调 `km.bert(...)`，非 `km.bert.bert(...)`） |
| `ProsodyPredictor` / `TextEncoder` | DurationEncoder + BiLSTM + duration_proj + shared LSTM + F0/N 头 / Embedding + 5×(Conv1d+LayerNorm+LeakyReLU+Dropout) + BiLSTM | 时长/F0/能量、音素特征 | `style_dim=128`；`n_symbols=n_token` |
| **`Decoder`** / **`AdaIN1d`** | AdainResBlk1d 栈 + F0/N conv + `Generator` / `InstanceNorm1d(affine=False)` + `Linear` | ★ 声码器、风格调制 | `dim_out=n_mels=80`；⚠️ **必须 `affine=False`** |
| `SineGen` / `SourceModuleHnNSF` | 谐波激励源 | 有声/无声建模 | ⚠️ **含随机数** |

### 2.4 与同类的差异

**vs StyleTTS2**：kokoro 是同一路线的**精简版**（82M），端侧可用。
**vs 自回归 TTS（Tacotron/VITS）**：kokoro **非自回归**，一次前向出整条波形。
**vs HiFi-GAN 声码器**：kokoro 用 **iSTFTNet**（频域），质量/速度平衡。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出 | `audio`：波形 @24 kHz；`pred_dur`：每音素持续帧数（`(T,)` / `(B,T)`） |
| 时长解码 | `duration = sigmoid(duration_logits).sum(-1) / speed`；`pred_dur = round(duration).clamp(min=1)` |
| 对齐矩阵 | `pred_aln_trg[T, Σdur]` 由 `repeat_interleave` + 赋值构造（**不可微**，监督须作用在 sigmoid 分布上） |
| 声码器 | `Decoder(asr, F0_curve, N, s)` → iSTFT → 波形 |
| style / speed | `ref_s` 必须是 `(1,256)` 或 `(B,256)`：`[:,128:]` → predictor、`[:,:128]` → decoder；`speed` 缩放时长 |
| G2P | **不在模型内**：文本→音素需外部（`misaki` 等）；`KokoroDataset` **只做 vocab 映射** |

### 3.1 训练损失（⚠️ 自写，官方未公开）

`KokoroLoss` = **mel 重建 L1 + F0 L1 + 能量 L1 + 时长 L1** 四项加权（`mel`：生成 vs 目标波形的
log-mel L1，可微、按设备缓存；`f0`：`F0_pred` vs 目标波形自相关估的 F0；`energy`：`N_pred` vs
目标波形 RMS 包络；`duration`：`sigmoid(duration_logits).sum(-1)` 的总时长 vs `pred_dur` 总和，
**避开 `round` 不可微**）。

> ⚠️ **官方从未公开训练代码** ⇒ 训练 loss 为**自定**，**不能声称与官方对齐**。
> ③ 的验收改用「**模块级梯度通路 vs 官方推理包**」。

---

## 4. 配置与用法

```yaml
Architecture: {task: kokoro_tts, algorithm: kokoro,
               config_path: torchkiln/audio/kokoro/configs/kokoro-v1_0.json, weights_path: null}
Loss: {name: KokoroLoss, sr: 24000, n_fft: 1024, hop_length: 256, n_mels: 80,
       w_mel: 1.0, w_f0: 1.0, w_energy: 1.0, w_duration: 1.0}
Optimizer: {name: Adam, beta1: 0.9, beta2: 0.999, lr: {name: Cosine, learning_rate: 0.0001}}
Train: {dataset: {name: KokoroDataset, data_dir: datasets/kokoro_demo,
        label_file_list: [datasets/kokoro_demo/train.txt], sr: 24000, max_len: 48000}}
```

数据清单格式（每行 `<音频相对路径>\t<音素串>`）：`wav/0001.wav<TAB>həlˈoʊ wˈɜːld`。
`weights_path` 正式训练填本地 `.pth`，冒烟用 `null`。

```bash
tkiln train -c configs/audio/kokoro_demo.yml
tkiln val   -c configs/audio/kokoro_demo.yml --weights output/kokoro_demo/best_accuracy.pth
```

推理：`KModel(config=<本地 config.json>, model=<本地 .pth>)` 后 `km(phonemes, ref_s)`，
`ref_s` 需 voice pack 或参考音频（`FloatTensor(1,256)`）。

---

## 5. 规模与速度

| 子模块 | 参数 |
|---|---|
| `bert` 6.2925 M / `bert_encoder` 0.3937 M / `predictor` 16.1946 M / `text_encoder` 5.6064 M / `decoder` 53.2762 M | **合计 81.7634 M** |

- 权重体积 **327.21 MB**（v1.0）/ **327.25 MB**（v1.1-zh）；采样率 **24 kHz**。
- CLI 冒烟速度 **13.9 fps**（demo，1 epoch，batch 2）；正式推理耗时**未统计**。

---

## 6. 公开指标

### 6.1 ⭐ 四条对齐全过（英文 v1.0 + 中文 v1.1-zh 双版）

| # | 项 | 结果 |
|---|---|---|
| ① | 权重加载 | **missing=0 / unexpected=0**（两版权重）；**81.763 M** 与权重总量一致 |
| ② | 全链逐层前向 | `bert_dur / d_en / d / lstm_x / duration_raw / pred_dur / pred_aln_trg / en / F0_pred / N_pred / t_en / asr` **全部 maxdiff = 0.000e+00**（13+ 层逐位）；Generator 固定种子（1234/2026）后 **0.000000e+00**，确定性部分（`har_source/uv/hs/har_spec/har_phase/har/noi_source`）**0.000000e+00** |
| ③ | 模块级 loss + 梯度 | `bert` 0.28293204 / `bert_encoder` 0.79446465 / `text_encoder` 0.15265706 / `predictor` 19.98767853 / `decoder` 29226.50976562 —— **两侧完全相同**；**510 个参数梯度全部 rel = 0.000e+00** |
| ④ | 端到端（波形/频谱） | **英文 v1.0**：`pred_dur` 逐位 `[3,2,2,2,2,2,2,2]`；**波形 maxdiff 0.000000e+00**（余弦 1.0000007153）；mel 0.000000e+00<br>**中文 v1.1-zh**：`pred_dur` 逐位 `[1,1,1,1,2,2,2,2]`；波形 **0.000000e+00**（余弦 1.0000019073）；mel 0.000000e+00 |

### 6.2 ⚠️ ① 的根因：`AdaIN1d` 的 `InstanceNorm1d` 必须 `affine=False`

- **证据（键差集）**：权重键（去 `module.`）vs 模型期望键 → **交集 10 / unexpected 0 /
  missing 4**（以 `F0.0.*` 为例），缺失全是 `F0.0.norm{1,2}.norm.{weight,bias}`。
  **算术吻合**：6 blocks × 2 norms × (weight+bias) = 24 = predictor 的 missing 数
  （decoder 的 116 同理）⇒ 两处同源。
- **根因**：`AdaIN1d = { norm: InstanceNorm1d, fc: Linear }`；权重有 `.fc.*`（交集）
  但**没有** `.norm.*` ⇒ 旧版权重的 `InstanceNorm1d` 是 **`affine=False`（无参数）**，
  而当前代码是 `affine=True`。**修法**：改模型 —— 构造成 **`affine=False`**（重建 70 个）。
- **官方源码自认**（`istftnet.py::AdaIN1d` 注释）：`affine should be False … but there's a bug
  in the old torch.onnx.export … This shouldn't really matter setting it to True, since we're in
  inference mode` ⇒ 是 **ONNX workaround 的副产物**；默认 `weight=1/bias=0` ⇒ **与官方前向等价**。
  反过来，用默认 `affine=True` 会 **missing 140 键**（24 + 116）。

### 6.3 ⚠️ `Generator` 含 3 处随机噪声（对拍必须固定种子）

`SineGen._f02sine` 的 `rand_ini = torch.rand(...)`（正弦初相）、`SineGen.forward` 与
`SourceModuleHnNSF.forward` 各一处 `torch.randn_like(...)` ⇒ 谐波源激励**本质带随机性**。
**对拍方法**：两侧**同步 `torch.manual_seed`**（seed 1234/2026 → maxdiff 0.000e+00）
或**只比确定性部分**；此前看到的 `audio rel 0.517` 纯粹来自这些随机噪声。

---

## 7. 选型建议

| 场景 | 建议 | 理由 |
|---|---|---|
| 高质量离线 TTS | **kokoro-82M** | 82M / 327MB，质量高 |
| **中文** TTS | **v1.1-zh**（`.pth` + `-zh.json`） | 词表 171 含中文音素（英文用 v1.0，词表 114） |
| 音色克隆 / 多说话人 | 需 **voice pack**（`ref_s`） | 本框架训练用**确定性随机 ref_s**（占位） |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs 官方 `kokoro` pip 包** | ①~④ 全对齐，**端到端波形逐位一致（0.000e+00）** |
| **vs StyleTTS2** | kokoro 是同一路线的精简版；官方**未开源训练代码** |
| **vs VITS/Paraformer（PaddleSpeech）** | 那两个**权重未发布 / 代码不在 wheel**（做不了，AGENTS 记录） |
| **vs `jonirajala/kokoro_training`（第三方 ★46）** | 有 LJSpeech 训练代码，但**无许可证 + 自实现模型** ⇒ 只作**训练管线参考**，**不作对齐基准** |

---

## 9. 已知问题 / 注意事项

- ⚠️ **权重键带 `module.` 前缀** → 加载时 `k[7:]`；**不需要** `.fc.→.norm.` 改名（早期误判）。
- ⚠️ **整链一次反传会触发原生段错误**：退出码 **`0xC0000005 = ACCESS_VIOLATION`**、
  **无 Python traceback**（原生崩溃，不是异常）；逐个模块单独 forward+backward **全部正常**。
  疑似 `Generator.forward` 内的 `with torch.no_grad()`（`istftnet.py` L306）与 `custom_stft`
  的**复数 iSTFT** 在整链 autograd 图下触发原生层 bug。**规避**：训练时**按模块分段反传**。
- ⚠️ **`Generator` 含 3 处随机噪声** → 对拍/复现**必须固定种子**；
  **HF 被墙** ⇒ `KModel` 必须显式传本地 `config.json` + `.pth`（权重走 ModelScope，25.5 MB/s）。
- ⚠️ **官方无训练代码** ⇒ `KokoroLoss` 为**自定**，**不能声称与官方训练对齐**。
- ⚠️ **训练期修的 7 个真问题**（官方代码在 CUDA 场景下的缺陷，均已注释留痕）：
  ① `build_trainer` 的 `family` 默认 `"ocr"` 吞掉新任务 → **task 优先**；② `kokoro_collate` 返回 **list**；
  ③ `KokoroLoss.forward` 返回 **dict**；④ `batch>1` 时 `pred_dur.squeeze()` 变多维 → 保留 `(B,T)`；
  ⑤ `TorchSTFT.window` 改 **`register_buffer`**；⑥ `custom_stft` 的 `torch.from_numpy` → `as_tensor`；
  ⑦ 损失里 `MelSpectrogram` **按 `str(device)` 缓存**并 `.to(dev)`。
- ⚠️ **`ref_s` 用确定性随机**（按 idx 播种）—— 真实训练需 **voice pack 或参考音频编码器**；
  **评估指标很简**（`wav_l1` + 余弦），正式评测应加 **mel/F0 指标**。
- ⚠️ 两版 config **不可混用**（v1.0 词表 114 / v1.1-zh 词表 171，**vocab 必须配对**）；
  本框架**未集成 FLOPs 统计**。
