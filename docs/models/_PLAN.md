# docs/models 文档体系（规划与完成状态）

> 目标：**一个模型一个文档，细到能照做**（对标 ultralytics / PaddleX 的文档详细度）。
> 写作纪律：**指标只能来自实测或官方公开表，查不到写「未统计/未验证」，绝不编造。**

---

## 统一结构（9 大节）

```
1. 论文与出处        （论文/arXiv/年份/机构/上游代码/本框架实现/移植方式）
2. 网络结构          （ASCII 框图 + 逐模块表 + 与同代/上代的差异表）
3. 输出与后处理      （anchor/解码/DFL/NMS 或 CTC/旋转 NMS/阈值）
4. 配置与用法        （YAML + 训练/评估/预测/导出四条链路命令）
5. 规模与速度        （参数/体积/推理/训练显存/FLOPs）
6. 公开指标          （官方值 + 本框架实测 + 差异原因）
7. 选型建议          （按场景推荐）
8. 参考对比          （与同族/同代/替代方案）
9. 已知问题          （踩过的坑、易错配置、限制）
```

模板见 [`_TEMPLATE.md`](_TEMPLATE.md)；索引见 [`README.md`](README.md)。

---

## 完成状态

### 2D 检测家族（9）

- [x] [`yolov3.md`](detect/yolov3.md) — YOLOv3（u/u-spp/u-tiny） <sub>`15.5 KB`</sub>
- [x] [`yolov5.md`](detect/yolov5.md) — YOLOv5（nu/n6u 等 u 系列） <sub>`13.7 KB`</sub>
- [x] [`yolov6.md`](detect/yolov6.md) — YOLOv6（仅 YAML，未对齐） <sub>`12.1 KB`</sub>
- [x] [`yolov8.md`](detect/yolov8.md) — YOLOv8 <sub>`14.3 KB`</sub>
- [x] [`yolov9.md`](detect/yolov9.md) — YOLOv9（t/s/m/c/e） <sub>`14.7 KB`</sub>
- [x] [`yolov10.md`](detect/yolov10.md) — YOLOv10（NMS-free） <sub>`16.6 KB`</sub>
- [x] [`yolo11.md`](detect/yolo11.md) — YOLO11 ★ 主力 <sub>`11.6 KB`</sub>
- [x] [`yolo12.md`](detect/yolo12.md) — YOLO12（区域注意力） <sub>`14.6 KB`</sub>
- [x] [`yolo26.md`](detect/yolo26.md) — YOLO26（端到端 E2E） <sub>`15.6 KB`</sub>

### 任务头（9）

- [x] [`segment.md`](task/segment.md) — 实例分割 Segment/Segment26 <sub>`12.9 KB`</sub>
- [x] [`pose.md`](task/pose.md) — 关键点 Pose <sub>`10.0 KB`</sub>
- [x] [`obb.md`](task/obb.md) — 旋转框 OBB/OBB26 <sub>`11.0 KB`</sub>
- [x] [`classify.md`](task/classify.md) — 图像分类 Classify <sub>`9.2 KB`</sub>
- [x] [`semantic.md`](task/semantic.md) — 语义分割 Semantic <sub>`9.0 KB`</sub>
- [x] [`depth.md`](task/depth.md) — 深度估计 Depth <sub>`9.0 KB`</sub>
- [x] [`video_cls.md`](task/video_cls.md) — 视频分类 <sub>`8.4 KB`</sub>
- [x] [`pose_action.md`](task/pose_action.md) — 行为识别 <sub>`8.6 KB`</sub>
- [x] [`attribute.md`](task/attribute.md) — 多标签属性 <sub>`10.0 KB`</sub>

### OCR / 车牌（4）

- [x] [`plate_det.md`](ocr/plate_det.md) — 车牌检测 <sub>`10.0 KB`</sub>
- [x] [`plate_rec.md`](ocr/plate_rec.md) — 车牌识别 CRNN+CTC <sub>`11.3 KB`</sub>
- [x] [`pgnet_lite.md`](ocr/pgnet_lite.md) — PGNet_lite 端到端 ⭐ <sub>`17.8 KB`</sub>
- [x] [`ppocrv6.md`](ocr/ppocrv6.md) — PP-OCRv6 两阶段 <sub>`13.1 KB`</sub>

### 3D / 点云 / 车道（4）

- [x] [`centerpoint.md`](pc/centerpoint.md) — CenterPoint-Pillars <sub>`12.1 KB`</sub>
- [x] [`squeezesegv3.md`](pc/squeezesegv3.md) — SqueezeSegV3 <sub>`10.9 KB`</sub>
- [x] [`bev_lanedet.md`](pc/bev_lanedet.md) — BEV-LaneDet <sub>`12.1 KB`</sub>
- [x] [`lane_row_seg.md`](pc/lane_row_seg.md) — 车道线 row/seg <sub>`11.9 KB`</sub>

### 音频 SOTA（4）

- [x] [`panns_cnn14.md`](audio/panns_cnn14.md) — PANNs CNN14（语音分类） <sub>`10.5 KB`</sub>
- [x] [`ecapa_tdnn.md`](audio/ecapa_tdnn.md) — ECAPA-TDNN（说话人） <sub>`10.2 KB`</sub>
- [x] [`mdtc.md`](audio/mdtc.md) — MDTC（关键词） <sub>`10.6 KB`</sub>
- [x] [`kokoro.md`](audio/kokoro.md) — kokoro-82M（TTS） <sub>`14.7 KB`</sub>

### 时间序列（22）

- [x] [`ts_overview.md`](ts/ts_overview.md) — ★ 总览与选型（先读这篇） <sub>`4.6 KB`</sub>
- [x] [`nbeats.md`](ts/nbeats.md) — NBEATS <sub>`4.7 KB`</sub>
- [x] [`nhits.md`](ts/nhits.md) — NHiTS <sub>`4.3 KB`</sub>
- [x] [`mlp.md`](ts/mlp.md) — MLP <sub>`3.5 KB`</sub>
- [x] [`dlinear.md`](ts/dlinear.md) — DLinear <sub>`4.2 KB`</sub>
- [x] [`tcn.md`](ts/tcn.md) — TCN <sub>`4.2 KB`</sub>
- [x] [`rnn.md`](ts/rnn.md) — RNN(LSTM/GRU) <sub>`3.8 KB`</sub>
- [x] [`lstnet.md`](ts/lstnet.md) — LSTNet <sub>`4.0 KB`</sub>
- [x] [`transformer.md`](ts/transformer.md) — Transformer <sub>`3.9 KB`</sub>
- [x] [`scinet.md`](ts/scinet.md) — SCINet <sub>`4.3 KB`</sub>
- [x] [`informer.md`](ts/informer.md) — Informer <sub>`4.2 KB`</sub>
- [x] [`deepar.md`](ts/deepar.md) — DeepAR（概率） <sub>`3.9 KB`</sub>
- [x] [`tft.md`](ts/tft.md) — TFT（功率预测首选） <sub>`5.2 KB`</sub>
- [x] [`anomaly_ae.md`](ts/anomaly_ae.md) — 异常检测：AutoEncoder <sub>`3.7 KB`</sub>
- [x] [`anomaly_vae.md`](ts/anomaly_vae.md) — 异常检测：VAE <sub>`3.2 KB`</sub>
- [x] [`anomaly_usad.md`](ts/anomaly_usad.md) — 异常检测：USAD <sub>`3.5 KB`</sub>
- [x] [`anomaly_mtadgat.md`](ts/anomaly_mtadgat.md) — 异常检测：MTAD-GAT <sub>`3.9 KB`</sub>
- [x] [`anomaly_at.md`](ts/anomaly_at.md) — 异常检测：Anomaly Transformer <sub>`4.5 KB`</sub>
- [x] [`ts_classify.md`](ts/ts_classify.md) — 时序分类 CNN/InceptionTime <sub>`4.1 KB`</sub>
- [x] [`ts_embed.md`](ts/ts_embed.md) — 表示学习 TS2Vec/CoST <sub>`4.3 KB`</sub>
- [x] [`ts_rul.md`](ts/ts_rul.md) — RUL 剩余寿命 <sub>`5.3 KB`</sub>
- [x] [`changepoint.md`](ts/changepoint.md) — 变点/漂移检测（纯统计） <sub>`4.1 KB`</sub>

### 基础设施（3）

- [x] [`trainer.md`](infra/trainer.md) — ★ 训练器与统一口径（先读这篇） <sub>`3.9 KB`</sub>
- [x] [`data_formats.md`](infra/data_formats.md) — ★ 数据格式与 YAML 配置 <sub>`2.6 KB`</sub>
- [x] [`align_method.md`](infra/align_method.md) — ★ 对齐方法论（复现验收标准） <sub>`2.9 KB`</sub>

---

## 统计

| 项 | 值 |
|---|---|
| 已完成 | **55 篇** |
| 缺失 | **0 篇** |
| 总体积 | **463 KB**（约 208 页 A4） |
| 覆盖 | 7 大类 / 26 个任务 |

---

## 已知取舍（如实记录）

| 项 | 说明 |
|---|---|
| **FLOPs / TOPS** | ⚠️ 本框架**未集成** `thop` / `fvcore`，所有文档的 FLOPs 列一律写「未统计」。**不编造数字。** |
| **本机推理耗时** | 各篇标注的 ms 数为**实测**或标注「粗估」；无实测的写「未测」。 |
| **v6 / semantic / depth / video_cls / pose_action** | 框架**只有结构定义**（YAML + 模块），**未做权重加载与数值对齐** → 对应篇明确标注「未对齐 / 无官方权重」。 |
| **plate_det / plate_rec / PP-OCRv6** | 托管的是**官方/业务权重**，未做跨框架逐位对齐 → 明确标注「未验证逐位一致」。 |
| **DLinear** | ⚠️ **不在 PaddleTS 1.1.0 中** → **无对齐基准**（框架自带扩展）。 |
| **BiLSTM+Attn / TransformerRegressor** | 框架**新增**（为 RUL 设计），**无 paddlets 对照**。 |

---

## 复现验收标准（各篇「公开指标」节均按此）

| # | 项 | 判据 |
|---|---|---|
| ① | 权重加载 | `missing=0 / unexpected=0`（或仅差函数式 DFL） |
| ② | 同权重逐层前向 | maxdiff ~1e-5 |
| ③ | 单步 loss / 梯度 | loss 相对误差 <1e-3；梯度仅算子级微差 |
| ④ | 同权重推理指标 | 差 ≤0.02 |
| ⑤ | 端到端训练 | 终值差 ≤0.02~0.03（含随机性） |

> **1-4 项对齐即复现成功**；第 5 项天生有噪声。
> 方法论详见 [`infra/align_method.md`](infra/align_method.md)。
