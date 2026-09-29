# PANNs CNN14

> **定位**：音频模式识别（AudioSet 527 类）的经典 SOTA 基线；**CNN14** 是 PANNs 家族
> 里最常用的骨干，输出 2048 维音频 embedding，也常作其他音频任务的通用特征提取器。
> **任务**：`panns_cls`（音频分类）
> **权重**：`weights/panns_cnn14.pth`（308 MB，由 PaddleSpeech `panns_cnn14.pdparams` 491.3 MB 转换）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition** |
| arXiv | **1912.10211**（2019-12-21，已核实标题/作者） |
| 作者 | Qiuqiang Kong、Yin Cao、Turab Iqbal、Yuxuan Wang、Wenwu Wang、Mark D. Plumbley |
| 期刊 | IEEE/ACM TASLP 2020 |
| 官方代码 | https://github.com/qiuqiangkong/audioset_tagging_cnn |
| 权重来源 | PaddleSpeech：`https://bj.bcebos.com/paddleaudio/models/panns_cnn14.pdparams` |
| 本框架实现 | `torchkiln/audio/panns.py`（`ConvBlock/ConvBlock5x5/CNN14/CNN10/CNN6`）<br>`torchkiln/tasks/panns_cls.py`（任务 + `PannsFEATURE`）、`torchkiln/audio/esc50_dataset.py`（ESC-50） |
| 移植方式 | PyTorch 逐命名移植 **PaddleSpeech** `cls/models/panns/panns.py` + 权重逐键对齐（missing=0/unexpected=0） |

### 论文要解决的问题

1. **音频模式识别缺少大规模预训练**：此前多数方法在小数据上从零训，泛化差；
2. **不同 CNN 结构与规模的影响未被系统研究**；
3. **迁移能力**：一个 AudioSet 上预训练的音频 embedding，应能迁移到
   声事件检测、音乐分类、语音等下游任务。

PANNs 的贡献：
- 在 **AudioSet（约 200 万段，527 类）** 上系统性比较 6 种 CNN（CNN6/10/14 + ResNet 系列）；
- 提出 **wavegram-Logmel-CNN**（时域+频域双流）；
- 用 **平均池化（mean）替代 max** 做时间聚合，泛化更好。

### 关键结论（论文 Table）

| 模型 | 参数 | AudioSet mAP | 备注 |
|---|---|---|---|
| CNN6 | 4.5M | **0.343** | 最轻 |
| CNN10 | 5.2M | **0.380** | — |
| **CNN14** | **80.8M** | **0.431** | ★ 本框架实现 |
| ResNet38 | 76.6M | 0.434 | — |
| Wavegram-Logmel-CNN | 81.1M | **0.439** | 最高 |

---

## 2. 网络结构

### 2.1 整体框图（CNN14）

```
波形 (N, L)  @32 kHz
  │  特征（PannsFEATURE）：LogMelSpectrogram(sr=32000, n_fft=1024, hop=320, win=1024,
  │        hann, f_min=50, f_max=14000, n_mels=64) → 10*log10(clamp(x,1e-10))
  ▼ (N, 64, T)  --transpose-->  (N, T, 64)  --unsqueeze-->  (N, 1, T, 64)
bn0: BatchNorm2d(64)   ← ★ 先把 (N,1,T,64) transpose 成 (N,64,T,1) 再 BN，再转回
  ▼
conv_block1: Conv3×3(1→64,   bias=False)×2 + BN + ReLU → avg_pool(2,2) → Dropout(0.2)
conv_block2: Conv3×3(64→128) ×2 同上
conv_block3: Conv3×3(128→256)
conv_block4: Conv3×3(256→512)
conv_block5: Conv3×3(512→1024)
conv_block6: Conv3×3(1024→2048)   ← ★ 此块 pool_size=(1,1)（前 5 块是 (2,2)）
  ▼ (N, 2048, T', 1) 左右
mean(dim=3) → (N,2048,T')  →  amax(dim=2) + mean(dim=2)  （max + mean 拼接式聚合）
  ▼ Dropout(0.5) → fc1: Linear(2048→2048) → ReLU
  ├─ extract_embedding=True  → Dropout(0.5) → (N, 2048)   # ★ 通用音频 embedding
  └─ extract_embedding=False → fc_audioset: Linear(2048→527) → Sigmoid  # AudioSet 527 类
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | PaddleSpeech 名 |
|---|---|---|---|---|
| `ConvBlock` | 2× [`Conv2d(k3,p1,bias=False)` → BN → ReLU] + 池化 | 4D→4D | 主体 | `ConvBlock` |
| `ConvBlock5x5` | 1× [`Conv2d(k5,p2)` → BN → ReLU] | 4D→4D | CNN6 用 | `ConvBlock5x5` |
| `_pool` | `max` / `avg` / **`avg+max`** | — | 池化 | `_pool` |
| **`bn0` + `_bn0_nhwc`** | `transpose(1,3)` → `BatchNorm2d(64)` → `transpose(1,3)` | (N,1,T,64) | ★ **把 mel 维(64)当通道** | `BatchNorm2D(64)` + NHWC 转置 |
| `CNN14` | 6×ConvBlock + fc1 + fc_audioset | 波形特征→embedding/概率 | 模型本体 | `CNN14` |
| `PannsFEATURE` | `torchaudio.MelSpectrogram` + `mag2db` | (N,L)→(N,64,T) | 特征（可微、按设备） | `LogMelSpectrogram` |

### 2.3 与同类的差异

| 对比对象 | 差异点 | 为什么 | 效果 |
|---|---|---|---|
| **CNN6 / CNN10** | 层数 6/10 vs 14 | 规模权衡 | CNN14 mAP 最高（0.431） |
| **ResNet38** | 残差 vs 直筒 | ResNet 更深 | 0.434，与 CNN14 接近 |
| **Wavegram-Logmel-CNN** | 双流（时域+频域） | 补时域信息 | 0.439（最高） |
| **PaddleSpeech vs 原版** | Paddle 版把 `bn0` 用 **NHWC 转置** 实现 | 对齐 Paddle 的 `BatchNorm2D(64)` 语义 | 权重可 1:1 加载 |

> ⚠️ **输入必须是 4D `(N, 1, T, n_mels)`**（官方 `SoundClassifier` 里 `unsqueeze(1)`）。
> 传 `(N, T, 64)` 会 shape 报错 —— 这是移植时最容易漏的一步。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输出（`extract_embedding=True`，默认） | `(N, 2048)` 音频 embedding |
| 输出（`extract_embedding=False`） | `(N, 527)` AudioSet 概率（已 `sigmoid`） |
| 分类头（微调时） | `PannsClassifier = backbone + Dropout(0.1) + Linear(2048, num_class)` |
| 损失 | `CrossEntropyLoss`（ESC-50 单标签）；返回 **dict** `{"loss", "cls_loss"}`（trainer 约定） |
| 指标 | `acc`（top-1）+ 平均 `cls_loss` |
| 后处理 | **无**（音频分类直接 argmax/概率） |
| 帧无关性 | 时间维通过 `amax + mean` 聚合 → **输入长度 T 可变**（同一模型可跑不同时长） |

---

## 4. 配置与用法

```yaml
# ESC-50 训练对齐配置（节选）
Architecture:
  task: panns_cls
  algorithm: panns
  num_class: 50
  weights_path: _downloads/speech/panns_train/init_cnn14.npz   # 与 Paddle 同起点
  feature: {sr: 32000, n_fft: 1024, hop_length: 320, win_length: 1024,
            window: hann, f_min: 50.0, f_max: 14000.0, n_mels: 64}
Loss: {name: CrossEntropyLoss}
Optimizer:
  name: Adam
  clip_grad_norm: null        # ★ PaddleSpeech 的 PANNs 训练循环不裁剪（见 §9）
  lr: {name: Const, learning_rate: 0.001, warmup_epoch: 0}
Metric: {name: PannsClsMetric}
Train:
  dataset: {name: ESC50Dataset, data_dir: datasets/esc50, split: 1, sample_rate: 32000}
  loader:  {batch_size_per_card: 32, num_workers: 4, prefetch_factor: 4}
```

数据目录（ESC-50）：`<data_dir>/ESC-50-master/{meta/esc50.csv, audio/*.wav}`；
划分 `mode='train'` → `fold != split`（1600 条），`mode='dev'` → `fold == split`（400 条）。

```bash
tkiln train -c configs/audio/panns_esc50_align.yml
tkiln val   -c configs/audio/panns_esc50_align.yml --weights output/panns_esc50_align/best_accuracy.pth
```

> 另有两个对照配置：`panns_esc50_ctrl30.yml`（共享 CSV 原序）
> 与 `panns_esc50_order30.yml`（共享随机置换，`order_seed`）。

---

## 5. 规模与速度

| 项 | 值 | 来源 |
|---|---|---|
| 参数（CNN14） | **80.7536 M** | 本框架实算（与论文 80.8M 一致） |
| 权重体积 | **308 MB**（`.pth`）；Paddle `.pdparams` **491.3 MB** | AGENTS 记录 |
| 下载速度 | 105 MB/s（bcebos） | AGENTS 记录 |
| 推理耗时（本机） | **未统计** | — |
| 训练显存 | **未统计**（batch=32 可跑，ESC-50 波形短） | — |

---

---

---

---

---

### 📊 FLOPs（实测）

| 项 | 值 |
|---|---|
| **FLOPs** | **41.15 GFLOPs** |
| **MACs** | **20.58 GMACs** |
| 参数量 | **80.754 M** |
| 输入规格 | `10s @ 32kHz -> (1,1,1024,64)` |
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
| 音频分类（声事件/环境音） | **CNN14 + AudioSet 预训练** | 80.8M 虽大但 mAP 0.431 |
| 需要**音频 embedding** 做下游 | `extract_embedding=True` → 2048 维 | 通用特征 |
| 极轻量边缘设备 | 换 **CNN6**（4.5M） | mAP 0.343，差 ~0.09 |
| 需要最高 mAP | **Wavegram-Logmel-CNN** | 0.439；本框架**未实现**该双流 |
| 训练对齐参考 | 用 `panns_esc50_{align,ctrl30,order30}.yml` | 与 Paddle 共享随机序 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs 原版 audioset_tagging_cnn** | PaddleSpeech 版把 `bn0` 用 NHWC 转置；本框架照搬 Paddle 口径 |
| **vs PaddleSpeech** | ①②③④ 全对齐（fp64 3.55e-11 / loss 6.46e-12 / probs 3.4e-05） |
| **vs 一般 VGGish** | CNN14 mAP 更高（0.431 vs VGGish ~0.314） |
| **vs AST（音频 Transformer）** | AST 更强但更重；CNN14 是轻量基线 |
| **训练 vs 官方** | 官方 `panns.yaml` **未随 wheel 发布** ⇒ 超参为自建（batch 32 / Adam 1e-3 / fold1） |

---

## 9. 已知问题 / 注意事项

- ⚠️ **输入必须 4D `(N, 1, T, n_mels)`**：`CNN14` 的 `bn0` 是 `BatchNorm2d(64)`，
  传 3D 或把 mel 放错维度都会报错。
- ⚠️ **`bn0` 是 NHWC 转置**：`(N,1,T,64) → transpose(1,3) → (N,64,T,1) → BN → 转回`。
  这段在 Paddle 原版就有，漏掉会数值不符。
- ⚠️ **`conv_block6` 的 `pool_size=(1,1)`**（前 5 块才是 `(2,2)`）—— 容易看错。
- ⚠️ **梯度裁剪差异（关键）**：PaddleSpeech 的 PANNs 训练**完全不裁剪**
  （实测全局梯度 L2 范数 **81~94**），而框架默认 `clip_grad_norm=10.0`
  → 每步梯度被缩到约 1/9。已把阈值做成**可配置** `Optimizer.clip_grad_norm`
  （**默认仍 10.0，行为不变**），设 `null` 关闭以对齐。
- ⚠️ **`_PannsCE.forward` 必须返回 dict**（trainer 取 `loss_dict["loss"]`）；
  且 trainer 传的 `labels` 是 **list** `[wav, label]`。
- ⚠️ **ESC-50 波形读取口径**：必须 `soundfile(float32)` + `resampy(kaiser_fast)` +
  `normalize(linear, 1e-8)`。早期用 `wave` + `np.interp` 线性插值 → 特征有系统差异。
- ⚠️ **`esc50_collate` 不做 padding**，按 batch 内**最短长度截断**后 stack（与官方一致）。
- ⚠️ Windows 上 **eval 的 DataLoader `num_workers>0` 会 "worker exited unexpectedly"** → 评估用 0。
- ⚠️ **官方 `panns.yaml` 未发布** ⇒ 训练超参是自建，**不能声称与官方训练对齐**；
  已对齐的是「实现级」（数据/特征/前向/loss/优化器）。
- ⚠️ 异常退出会留孤儿 worker 占数 GB 内存 → `Stop-Process -Name python -Force` 后重跑。
- ⚠️ 本框架**未集成 FLOPs 统计**。
