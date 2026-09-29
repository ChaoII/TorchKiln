# TorchKiln 模型手册（逐模型：结构 / 公开指标 / 速度 / 权重）

> 每个模型给出：**结构组成**、**后处理（anchor/NMS/解码）**、**公开指标**、**速度/规模**、**权重**。
> 指标来源标注：`ultra官方` / `Paddle3D官方` / `PaddleSpeech官方` / `本仓库实测`。

---

## 一、2D 目标检测 / 分割 / 姿态 / OBB（YOLO 家族）

### 1.1 通用结构（所有 YOLO 家族共用三段式）

```
Input(B,3,H,W)
  └─ Backbone   输出 P3@8s P4@16s P5@32s（v3/v5/v8/v9/v10/v11/v12/v26 各不同）
  └─ Neck       自顶向下 + 自底向上融合（PANet / RepLKFPN / A2C2f ...）
  └─ Head       分类(cv3) + 回归(cv2) [+ DFL]（+ 分割 cv4 / 姿态 cv4 / OBB cv4-angle）
```

| 家族 | Backbone 变换 | Neck | Head 形态 | DFL |
|---|---|---|---|---|
| **v3** | Bottleneck+C3 | FPN(+SPP) | **legacy Conv 头** | reg_max=16 |
| **v5** | C3+Bottleneck, SPPF | PANet(C3) | **legacy** | 16 |
| **v8** | C2f | PANet(C2f) | **legacy** | 16 |
| **v9** | RepNCSPELAN4 / AConv / ELAN1 | PANet+SPPELAN | legacy | 16 |
| **v10** | C2f/C2fCIB(按规模) + PSA | PANet+SPPF | **legacy** | 16 |
| **v11** | C3k2(attn 可选) + C2PSA | PANet(C3k2) | **新 DWConv 头** | 16 |
| **v12** | A2C2f/ABlock/AAttn | PANet(A2C2f) | 新头 | 16 |
| **v26** | C3k2(attn) + C2PSA | PANet+SPPF(add) | 新头 + **end2end(one2one)** | **1（无 DFL，用 L1）** |

### 1.2 后处理口径（detect / obb）

| 项 | 用法 |
|---|---|
| **Anchor** | **全部家族 anchor-free**（v5u/v3u 亦然；`anchors` 属性仅为占位） |
| 解码 | `xywh = (raw*2 + (anchor-0.5)) * stride`（网格中心） |
| DFL | `reg_max=16` 时 softmax 期望；**v26 reg_max=1 时用 L1 回归** |
| **NMS** | **`torchvision.ops.nms`**（C++，实测 16800 框 10s → **2.5s/50图，200×**）；旋转框用 **probiou + fast_nms**（2000 框 87s → **0.13s**） |
| 置信度 | `conf_thres` 默认 0.001（评估）/ 0.25（推理） |
| OBB 角度 | v8/v11: `θ=(sigmoid(ang)-0.25)*π`；**v26: raw angle**（不 sigmoid） |

### 1.3 公开指标（COCO val2017，ultralytics 官方）

| 模型 | 参数 | mAP50-95 | 权重 |
|---|---|---|---|
| yolo11n | 2.6M | 39.5 | `yolo11n.pt` |
| yolo11s | 9.4M | 47.0 | `yolo11s.pt` |
| yolo11m | 20.1M | 51.5 | `yolo11m.pt` |
| yolo11l | 25.3M | 53.4 | `yolo11l.pt` |
| yolo11x | 56.9M | 54.7 | `yolo11x.pt` |
| yolov8n | 3.2M | 37.3 | `yolov8n.pt` |
| yolov8x | 68.2M | 53.9 | `yolov8x.pt` |
| yolo26n | 2.4M | 40.3 | `yolo26n.pt` |
| yolov10n | 2.3M | 39.5 | `yolov10n.pt` |
| yolov9c | 25.4M | 53.0 | `yolov9c.pt` |
| yolov5nu | 2.6M | 34.3 | `yolov5nu.pt` |
| yolov3u | 61.6M | 47.3 | `yolov3u.pt` |

> **本仓库实测（dota128 OBB 预训练推理）**：v11n **0.8005** / v8n 0.790 / v26n 0.806
> （ultra 分别 0.821 / 0.8021 / 0.828，**差 ≤0.022，算子级**）。

### 1.4 速度参考（RTX 4060 Ti，本仓库实测）

| 场景 | 实测 |
|---|---|
| 训练（imgsz=1024, batch=4, OBB） | 稳定可跑，占 4~6 GB |
| 训练（imgsz=640, batch=16, seg） | ~6.6s/epoch（缩放集 100/50） |
| 评估 NMS（16800 框） | 2.5s/50 图（torchvision C++） |

### 1.5 分割 / 姿态 / OBB 特有

| 任务 | 头结构 | 后处理 | 本仓库实测 |
|---|---|---|---|
| **segment** | `cv4` → mask coeff + `Proto`（原型） | 系数×proto + 上采样 + 二值化 | 同权重 mask mAP50-95 **0.8206**（ultra 0.8214） |
| **pose** | `cv4` → 关键点 `(x,y,vis)`，`c4=max(ch[0]//4, nk)` | 网格解码 + OKS-NMS | 微调 **0.495**（ultra 0.509） |
| **obb** | 独立 `cv4`(angle) 塔 | probiou+NMS | 0.8005（v11n 预训练） |

---

## 二、时序（5 任务 / 21 模型）

### 2.1 `ts_forecast`（12 模型，PaddleTS 1.1.0 移植）

| 模型 | 结构要点 | 参数(M) | 对齐 |
|---|---|---|---|
| **NBEATS** | trend/season 双 generator + FC stack，residual | 0.663 | 逐位 0 |
| **NHiTS** | 多尺度 pool + 分层插值 + 每 block 多 horizon | 0.935 | 逐位 0 |
| **MLP** | 展平 + 全连接 | 0.032 | 逐位 0 |
| **DLinear** | 序列分解（移动平均）+ 双线性 | 0.005 | — |
| **TCN** | 膨胀因果卷积 + 残差 | 0.150 | 浮点 6.9e-6 |
| **RNN(LSTM/GRU)** | 单/多层 RNN + 线性头 | 0.070 / 0.053 | 逐位 0 |
| **LSTNet** | CNN + RNN + **skip-RNN** + highway | 0.003 | 逐位 0 |
| **Transformer** | 标准 Encoder（paddle nn.Transformer） | 0.003 | 逐位 0 |
| **SCINet** | even/odd 采样树 + 交互卷积 + 解码 | 0.003 | 浮点 5e-6 |
| **Informer** | **ProbSparse** 注意力（含随机采样，**不可逐位复现**） | 11.301 | 前向正确 |
| **DeepAR** | RNN + **GaussianLikelihood**（概率输出） | 0.067 | 逐位 0 |
| **TFT** | GRN + 变量选择 + 可解释多头注意力 + 静态编码 | 0.251 | 浮点 1e-6 |
| *(新增)* **BiLSTM+Attn** | 双向 LSTM + 加性注意力池化 | 0.694 | — |
| *(新增)* **TransformerReg** | TransformerEncoder + cross-attn 池化 | 0.341 | — |

**输出形态**：点预测 `(B,H,D)` / 分位数 `(B,H,D,Q)` / 概率参数 `(B,H,D,2)`。

### 2.2 `ts_anomaly`（5 模型）

| 模型 | 结构 | 参数(M) | 异常分数 |
|---|---|---|---|
| **AutoEncoder** | MLP/CNN 编解码（逐时刻重建） | 0.005 | 重建 MSE（通道平均） |
| **VAE** | 编解码 + **重参数化** + KL | 0.002 | smooth_l1 + 0.2·KL |
| **USAD** | **对抗双解码器**（enc+dec1/dec2，两阶段训练） | 0.021 | `α·‖x-w1‖² + β·‖x-w3‖²` |
| **MTAD-GAT** | **双 GAT**(特征图/时间图) + GRU + 预测+重建双任务 | 0.032 | `√(pred-true)² + √(recon-true)²` |
| **AnomalyTransformer** | 关联差异（**Gaussian prior 核 vs series 注意力**）+ KL | 0.014 | `softmax(-series_loss-prior_loss) × rec_loss` |

**指标**：点级 P/R/F1（含 **point-adjust**）/ **AUC-ROC** / **AUC-PR**；阈值 `percentile`。
**本仓库实测（合成 demo）**：AUC-ROC **0.769~0.776**，F1≈1.0。

### 2.3 `ts_classify`（2 模型）

| 模型 | 结构 | 参数(M) |
|---|---|---|
| **CNN** | Conv1d→激活→AvgPool 堆叠（默认 Sigmoid，**可配 ReLU**） | 0.005 |
| **InceptionTime** | **Inception 模块**（3 分支 conv + maxpool）+ 残差 shortcut | 0.030 |

**指标**：accuracy / macro-F1 / 每类 F1。**实测（3 类工况）**：IT 0.459 / CNN 0.431（基线 0.333）。

### 2.4 `ts_embed`（2 模型，自监督）

| 模型 | 结构 | 参数(M) |
|---|---|---|
| **TS2Vec** | 膨胀卷积编码器 + **SWA 平均** + 分层对比（instance+temporal） | 0.046 |
| **CoST** | 膨胀卷积 + **TFD**(多尺度趋势) + **SFD**(频域 rfft 季节) + 时/频对比 | 0.772 |

**指标**：对比损失 + `repr_std`（查表征塌缩）。**实测**：loss 3.80→3.33 / 4.34→3.68。

### 2.5 `ts_rul`（复用预测模型族）

**数据**：C-MAPSS FD001-004（run-to-failure，按 unit 划分，分段线性 RUL，`max_rul=125`）。
**指标**：RMSE / MAE / **NASA Score（PHM08 非对称）** / Score_avg。

| 子集 | 台数 | 全量 RMSE | **近失效段(RUL≤30)** | NASA Score |
|---|---|---|---|---|
| FD001 | 100 | 25.44（TRF 24.46） | **7.63** | 13.9 |
| FD002 | 260 | **21.74** | **9.00** | 8.2 |
| FD003 | 100 | **19.28** | — | — |
| FD004 | 249 | 30.10 | — | — |

> ⚠️ **口径提示**：文献 12~18 对应「筛选 RUL≤50~100 的易样本」；
> 本框架同口径下为 **12.38**（RUL≤50），**同口径不落后**。

### 2.6 变点/漂移检测（纯统计，非模型）

CUSUM / Page-Hinkley / ADWIN / z-score；自检命中 **3/3**（CUSUM 延迟仅 1 点）。

---

## 三、3D / 点云 / 车道

| 模型 | 结构 | 后处理 | 公开指标 | 权重 |
|---|---|---|---|---|
| **CenterPoint-Pillars** | PillarFeatureNet(9→32→64) + Scatter + SecondBackbone(3/5/5) + SecondFPN + CenterHead（**hm bias=-2.19**） | 逐任务解码 + **旋转 NMS** | Paddle3D KITTI BEV mAP(Mod) **71.87**；**本仓库 67.8**（差 4，NMS/IoU 算子级） | `centerpoint_pillars_kitti.pth` (19.6MB) |
| **SqueezeSegV3** | SACRangeNet53 backbone + 5 尺度 head（range-view 球面投影 `(5,H,W)`） | 逐点 argmax | SemanticKITTI mIoU（Paddle3D 官方 ~62） | `squeezesegv3_rangenet53_semantickitti.pth` (99.7MB) |
| **BEV-LaneDet** | ResNet34 + **FCTransform**(BEV 投影) + seg/emb/offset/z 四头 | `post_conf=0.9` + **embedding 聚类** + min-cost-flow | Apollo 3D Lane F1：Paddle3D **0.7776**；**本仓库 0.8378**（同官方评估器，2 epoch） | `bev_lanedet_apollo_576x1024.pth` (168MB) |

> ⚠️ BEV-LaneDet **输入必须 576×1024**（FCTransform 特征尺寸硬编码 18×32 / 9×16）。

---

## 四、OCR

| 模型 | 结构 | 后处理 | 指标 |
|---|---|---|---|
| **plate_det** | YOLOv5n-0.5 改造 | YOLO 解码 + NMS | 车牌检测 mAP |
| **plate_rec** | CRNN/CTC | CTC 贪心解码 | 字符准确率 |
| **PGNet_lite（e2e）** | **PPLCNetV4E2E**(0.394M) + **PGFPNLCNet**(0.424M) + **PGHeadLite**(0.416M) = **1.22M/4.66MB** | PGNet_PostProcess（border→TCL 骨架→CTC→方向） | 官方 r50 基准 TotalText Hmean **84.69**；本轻量版前向 **11ms**（官方 r50 第 31.2ms） |

**e2e 速度对比（实测）**：两阶段 `23.9 + 1.669×框数` ms；PGNet 固定 54.2ms（r50）。
**交叉点 ≈ 20 框/图** —— 高框密度（几百框）时 PGNet 快 **9×**。

---

## 五、音频（PaddleSpeech 移植）

| 模型 | 结构 | 参数 | 公开指标 | 本仓库对齐 |
|---|---|---|---|---|
| **PANNs CNN14** | 6×ConvBlock(3×3) + bn0(NHWC) + fc1 + fc_audioset(527) | 80.77M | AudioSet mAP 0.431（原论文） | **四条全过**：fp64 前向 3.55e-11 / probs 差 3.4e-05 |
| **ECAPA-TDNN** | TDNNBlock + **Res2Net** + **SE** + MFA + **AttentiveStatPooling** | 20.77M | VoxCeleb1-E EER ~1.0%（原论文） | **四条全过**：fp64 1.47e-15 / 余弦 0.9999996 |
| **MDTC (KWS)** | depthwise 膨胀卷积 + TCNBlock×12 + linear | 0.034M | HeySnips 命中率 | **①②③全过**（fp64 6e-10 / logits 4.04e-09） |
| **kokoro-82M** | iSTFTNet 声码器 + PL-BERT(ALBERT) + ProsodyPredictor + style 向量 | 81.76M | TTS（英文 v1.0 / 中文 v1.1-zh） | **四条全过**：端到端波形逐位 0.000e+00 |

> 音频模型的**输入口径**：PANNs `(N,1,T,64)` 特征 / ECAPA `(N,80,T)` log-fbank /
> MDTC `(N,T,80)` **帧在前** / kokoro `input_ids + ref_s(1,256)`。

---

## 六、视频 / 行为 / 属性

| 任务 | 结构 | 说明 |
|---|---|---|
| `video_cls` | 帧采样 + 2D backbone + 时序聚合 | 视频分类 |
| `pose_action` | 骨架序列 → 图/时序网络 | 行为识别 |
| `attribute` | 多标签头（YOLO backbone + 多分支） | 车辆属性 |

---

## 七、速度与规模速查（本机 RTX 4060 Ti 16GB）

| 场景 | 实测 |
|---|---|
| YOLO 训练（imgsz=1024, batch=4） | 4~6 GB；batch=8 OOM |
| YOLO 评估 NMS | 2.5s / 50 图 |
| RUL 训练（C-MAPSS, 80ep） | ~1~3 分钟（FD002 260 台） |
| ts_forecast 训练（合成, 30ep） | <1 分钟 |
| PGNet_lite 训练（batch16+AMP） | **75.5s/epoch**，GPU 利用率 **98.5%** |
| PGNet_lite 前向 | **11.0ms** |
| BEV-LaneDet 训练 | 16.4 samples/s |

> **未统计 TOPS/FLOPs**：本仓库未集成 FLOPs 计数工具（如需，可加 `thop`/`fvcore`）。
