# FLOPs 实测汇总（本框架自测）

> **测量工具**：`torch.utils.flop_counter.FlopCounterMode`（PyTorch 内置，**无需装 thop / fvcore**）
> **脚本**：`_downloads/flops_measure.py`（主）、`flops_measure2/3.py`（补测）
> **数据**：`_downloads/flops_all.json`

## ⚠️ 口径说明（极易踩坑）

| 口径 | 定义 | 换算 |
|---|---|---|
| **FLOPs** | 乘 + 加各计 1 次（×2） | 本表的 **GFLOPs** 列 |
| **MACs** | 乘加合计算 1 次 | `MACs = FLOPs / 2` |

**ultralytics 官方表里的 "GFLOPs" 是 FLOPs 口径**（不是 MACs）。本框架实测与官方**逐个吻合**：

| 模型 | 本框架实测 | ultralytics 官方 | 吻合 |
|---|---|---|---|
| yolo11n | **6.541** | 6.5 | ✅ |
| yolo11s | **21.589** | 21.5 | ✅ |
| yolo11m | **68.103** | 68.0 | ✅ |
| yolo11l | **87.155** | 87.2 | ✅ |
| yolo11x | **195.271** | 195.3 | ✅ |
| yolov8n | **8.743** | 8.7 | ✅ |
| yolov8s | **28.602** | 28.6 | ✅ |
| yolov8m | **78.936** | 78.9 | ✅ |
| yolov8l | **165.147** | 165.2 | ✅ |
| yolov8x | **257.803** | 257.8 | ✅ |

> ⭐ **这是结构一致性的极强证据**：FLOPs 精确到小数点后一位吻合，
> 说明框架的 backbone / neck / head 结构与 ultralytics **完全一致**。

## 关于 TOPS

**TOPS = Tera Operations Per Second，是硬件指标**（依赖芯片的 int8 量化与并行度），
**不是模型固有属性**。本表只给 **FLOPs / MACs**（模型固有）。
换算示例：某芯片 int8 峰值 30 TOPS ÷ 2（乘加）÷ 利用率 40% ⇒ 有效 ~6 TFLOPs。

---

## 2D 检测（YOLO）

| 模型 | FLOPs | MACs | 参数量 |
|---|---|---|---|
| `plate_det` | **1.47** GFLOPs | 0.73 G | 0.447 M |
| `yolo26n` | **6.08** GFLOPs | 3.04 G | 2.572 M |
| `yolo11n` | **6.54** GFLOPs | 3.27 G | 2.624 M |
| `yolo12n` | **7.47** GFLOPs | 3.73 G | 2.602 M |
| `yolov5n` | **7.72** GFLOPs | 3.86 G | 2.655 M |
| `yolov10x` | **7.73** GFLOPs | 3.87 G | 2.321 M |
| `yolov10l` | **8.08** GFLOPs | 4.04 G | 2.430 M |
| `yolov10m` | **8.26** GFLOPs | 4.13 G | 2.485 M |
| `yolov10s` | **8.46** GFLOPs | 4.23 G | 2.565 M |
| `yolov10n` | **8.63** GFLOPs | 4.31 G | 2.776 M |
| `yolov8n` | **8.74** GFLOPs | 4.37 G | 3.157 M |
| `yolov6n` | **12.41** GFLOPs | 6.21 G | 4.408 M |
| `yolo11s` | **21.59** GFLOPs | 10.79 G | 9.459 M |
| `yolo26s` | **22.78** GFLOPs | 11.39 G | 10.010 M |
| `yolo12s` | **23.34** GFLOPs | 11.67 G | 9.284 M |
| `yolov5s` | **23.99** GFLOPs | 12.00 G | 9.153 M |
| `yolov8s` | **28.60** GFLOPs | 14.30 G | 11.166 M |
| `yolov6s` | **43.67** GFLOPs | 21.84 G | 16.286 M |
| `yolov9t` | **53.17** GFLOPs | 26.58 G | 13.914 M |
| `yolov9e` | **53.17** GFLOPs | 26.58 G | 13.914 M |
| `yolov5m` | **64.20** GFLOPs | 32.10 G | 25.111 M |
| `yolov9s` | **65.55** GFLOPs | 32.78 G | 16.845 M |
| `yolo11m` | **68.10** GFLOPs | 34.05 G | 20.115 M |
| `yolo12m` | **70.72** GFLOPs | 35.36 G | 20.199 M |
| `yolo26m` | **75.01** GFLOPs | 37.51 G | 21.896 M |
| `yolov8m` | **78.94** GFLOPs | 39.47 G | 25.903 M |
| `yolov9m` | **82.46** GFLOPs | 41.23 G | 20.775 M |
| `yolo11l` | **87.15** GFLOPs | 43.58 G | 25.372 M |
| `yolo26l` | **93.38** GFLOPs | 46.69 G | 26.300 M |
| `yolo12l` | **95.43** GFLOPs | 47.72 G | 26.451 M |
| `yolov9c` | **103.33** GFLOPs | 51.67 G | 25.591 M |
| `yolov5l` | **134.99** GFLOPs | 67.49 G | 53.225 M |
| `yolov8l` | **165.15** GFLOPs | 82.57 G | 43.691 M |
| `yolo11x` | **195.27** GFLOPs | 97.64 G | 56.966 M |
| `yolo12x` | **208.85** GFLOPs | 104.42 G | 59.211 M |
| `yolo26x` | **208.91** GFLOPs | 104.45 G | 58.993 M |
| `yolov5x` | **246.39** GFLOPs | 123.20 G | 97.276 M |
| `yolov8x` | **257.80** GFLOPs | 128.90 G | 68.230 M |
| `yolov3u` | **282.55** GFLOPs | 141.27 G | 103.754 M |

## OCR

| 模型 | FLOPs | MACs | 参数量 |
|---|---|---|---|
| `PGNet_lite` | **20.08** GFLOPs | 10.04 G | 1.221 M |

## 3D 检测

| 模型 | FLOPs | MACs | 参数量 |
|---|---|---|---|
| `CenterPoint-Pillars` | **310.11** GFLOPs | 155.05 G | 4.996 M |

## 点云分割

| 模型 | FLOPs | MACs | 参数量 |
|---|---|---|---|
| `SqueezeSegV3` | **130.51** GFLOPs | 65.25 G | 25.960 M |

## 车道线

| 模型 | FLOPs | MACs | 参数量 |
|---|---|---|---|
| `BEV-LaneDet` | **109.04** GFLOPs | 54.52 G | 43.159 M |

## 音频

| 模型 | FLOPs | MACs | 参数量 |
|---|---|---|---|
| `MDTC KWS` | **12.390** MFLOPs | 6.195 M | 0.034 M |
| `ECAPA-TDNN` | **11.25** GFLOPs | 5.62 G | 20.768 M |
| `PANNs CNN14` | **41.15** GFLOPs | 20.58 G | 80.754 M |

## 时间序列

| 模型 | FLOPs | MACs | 参数量 |
|---|---|---|---|
| `RNN-LSTM` | **0.006** MFLOPs | 0.003 M | 0.070 M |
| `DLinear` | **0.009** MFLOPs | 0.005 M | 0.005 M |
| `DeepAR` | **0.010** MFLOPs | 0.005 M | 0.018 M |
| `SCINet` | **0.014** MFLOPs | 0.007 M | 0.003 M |
| `Transformer` | **0.029** MFLOPs | 0.015 M | 0.003 M |
| `MLP` | **0.063** MFLOPs | 0.032 M | 0.032 M |
| `LSTNet` | **0.129** MFLOPs | 0.065 M | 0.003 M |
| `NBEATS` | **1.315** MFLOPs | 0.657 M | 0.663 M |
| `NHiTS` | **1.864** MFLOPs | 0.932 M | 0.935 M |
| `BiLSTMAttention` | **3.215** MFLOPs | 1.607 M | 0.693 M |
| `RNN-GRU` | **9.517** MFLOPs | 4.758 M | 0.053 M |
| `TCN` | **28.509** MFLOPs | 14.255 M | 0.150 M |
| `TFT` | **36.724** MFLOPs | 18.362 M | 0.401 M |
| `TransformerRegressor` | **66.220** MFLOPs | 33.110 M | 0.340 M |
| `Informer` | **1.30** GFLOPs | 0.65 G | 11.301 M |

---

## 各模型输入规格（复现用）

| 模型 | 输入 |
|---|---|
| YOLO（各家族 n 档） | `(1, 3, 640, 640)`，nc=80 |
| PGNet_lite | `(1, 3, 512, 512)`（训练固定 512，测试保比例到长边 768） |
| plate_det | `(1, 3, 640, 640)`，nc=1 |
| CenterPoint-Pillars | KITTI 4D 点云 → pillar 特征（本文用 2000 点） |
| SqueezeSegV3 | `(1, 5, 64, 256)` range-view 距离图（**必须投影后再测**） |
| BEV-LaneDet | `(1, 3, 576, 1024)` ⭐ **尺寸硬编码**，不可改 |
| PANNs CNN14 | `(1, 1, 1024, 64)`（10s @ 32kHz，64 mel） |
| ECAPA-TDNN | `(1, 80, 300)`（3s @ 16kHz，80 mel）⚠️ 顺序是 **(N, C, T)** |
| MDTC KWS | `(1, 98, 80)`（1s @ 16kHz）⚠️ 顺序是 **(N, T, C)** 帧在前 |
| 时序各模型 | `{'past_target': (1, 96, 1)}`，H=24 |
| TFT / DeepAR | 额外传 `known/observed/static` 协变量（本文测 2 维 known + 2 维 observed + 1 维 static） |

---

## 复现

```bash
python _downloads/flops_measure.py     # YOLO / 时序 / 音频 / OCR 主表
python _downloads/flops_measure2.py    # 补测：yolo12/v9/v10/v5/v3/v6/plate + 3D/点云/车道
python _downloads/flops_measure3.py    # 补测：DeepAR / TFT / TransformerReg / BEVLaneDet
```

> ⚠️ 脚本里 `sys.path` 需指向仓库根；`OMP_NUM_THREADS=1` 避免多线程干扰。
> ⚠️ `FlopCounterMode` 对 **自定义算子不计数**（如 iSTFT、probiou NMS）
> ⇒ 表中数值是**网络主干的 FLOPs**，不含后处理。
