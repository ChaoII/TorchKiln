# TorchKiln 模型文档索引

> **一个模型一个文档**。每篇含：论文出处 / 网络结构（ASCII + 逐模块表）/
> 共 **55 篇**（索引不含 `_TEMPLATE.md` / `_PLAN.md`）。时序部分先读 `ts/ts_overview.md`。
> 输出与后处理（anchor、NMS、DFL、旋转 NMS、CTC 等）/ 配置用法 / 规模与速度 /
> 公开指标 / 选型建议 / 参考对比 / 已知问题。

**写作纪律**：指标只能来自实测或官方公开表，**查不到写「未统计 / 未验证」，绝不编造**。

## 📊 附表

| 表 | 内容 |
|---|---|
| [_FLOPS.md](_FLOPS.md) | **61 个模型的 FLOPs / MACs / 参数量实测**（PyTorch 内置 FlopCounterMode，无需 thop）<br>口径已由 yolo11/v8 十个模型与 **ultralytics 官方表逐个吻合**验证 |
| [_TEMPLATE.md](_TEMPLATE.md) | 9 大节模板（写新模型文档时复制） |
| [_PLAN.md](_PLAN.md) | 规划与完成状态、已知取舍 |
| [../WEIGHTS_INVENTORY.md](../WEIGHTS_INVENTORY.md) | 预训练权重清单（本地 vs ModelScope 远程） |

---

## 目录

### 2D 检测家族（9 篇）

> Ultralytics YOLO 家族（3~26 代），含论文解读、anchor/NMS/DFL、逐代升级路径

- **[YOLO11](detect/yolo11.md)** — Ultralytics 2024 年 9 月发布的主力检测家族；用 **C3k2 + C2PSA** 替换 YOLOv8 的 C2f <sub>`11.6 KB`</sub>
- **[YOLO12](detect/yolo12.md)** — 2025-02 的 **"以注意力为中心"（attention-centric）** 检测器：在 v11 的 `C3k2` 骨架里把深层的 <sub>`14.6 KB`</sub>
- **[YOLO26](detect/yolo26.md)** — Ultralytics 2026-01 发布的最新主力家族，是 **YOLO11 的端到端后继**： <sub>`15.6 KB`</sub>
- **[YOLOv10](detect/yolov10.md)** — 清华团队 2024-05 提出的 **NMS-free（端到端）实时检测器**：训练时用 <sub>`16.6 KB`</sub>
- **[YOLOv3](detect/yolov3.md)** — Redmon 2018 的经典单阶段检测器；本框架复刻的是 **Ultralytics 的 `u` 变体**—— <sub>`15.5 KB`</sub>
- **[YOLOv5](detect/yolov5.md)** — Ultralytics 2020 年起维护的实时检测器；本框架对齐的是 **`u` 变体**—— <sub>`13.7 KB`</sub>
- **[YOLOv6](detect/yolov6.md)** — 美团 2022 年提出的工业级实时检测器。**本框架只复刻了它的 YAML 结构骨架（`torchkiln/cfg/models/v6/yo… <sub>`12.1 KB`</sub>
- **[YOLOv8](detect/yolov8.md)** — Ultralytics 2023-01-10 发布的"anchor-free + split head"通用检测基线； <sub>`14.3 KB`</sub>
- **[YOLOv9](detect/yolov9.md)** — 王建尧团队 2024-02 发布，用 **PGI（可编程梯度信息）+ GELAN（广义高效层聚合网络）** 缓解深层网络的信息丢失 <sub>`14.7 KB`</sub>

### 任务头（9 篇）

> 同一 backbone 上的不同任务头：实例分割 / 关键点 / 旋转框 / 分类 / 语义 / 深度 / 视频 / 行为 / 属性

- **[多标签属性识别](task/attribute.md)** — 行人/车辆多属性**识别 —— 轻量 CPU 友好的 `PP-LCNet_x1_0` backbone + <sub>`10.0 KB`</sub>
- **[图像分类](task/classify.md)** — YOLO 的**分类头** —— 复用检测 backbone（C3k2/C2PSA），末尾用 <sub>`9.2 KB`</sub>
- **[单目深度估计](task/depth.md)** — 多尺度特征融合后输出**单通道 log-depth**，`depth = exp(logit)`（米） <sub>`9.0 KB`</sub>
- **[旋转框检测](task/obb.md)** — 在检测头之上加一条 **角度塔（`cv4`）**，回归 `(cx, cy, w, h, theta)` 旋转框； <sub>`11.0 KB`</sub>
- **[关键点检测](task/pose.md)** — 在检测头之上加一条 **关键点塔（`cv4`）**，每个目标同时回归 `nk` 个关键点的 <sub>`10.0 KB`</sub>
- **[骨架行为识别](task/pose_action.md)** — 基于骨架序列**的行为识别 —— 空间图卷积（人体关节点拓扑） <sub>`8.6 KB`</sub>
- **[实例分割](task/segment.md)** — 在检测头之上加一条 **mask 系数塔（`cv4`）** 与一个 **原型分支（`Proto`） <sub>`12.9 KB`</sub>
- **[语义分割](task/semantic.md)** — 逐像素分类**（不区分实例）—— 多尺度特征（P3/P4/P5）融合后输出 <sub>`9.0 KB`</sub>
- **[视频分类](task/video_cls.md)** — 视频片段级**行为/场景分类 —— 2D CNN backbone + **Temporal Shift（TSM） <sub>`8.4 KB`</sub>

### OCR / 车牌（4 篇）

> 两阶段（det+rec）与端到端（PGNet）两路线，含成本模型与选型树

- **[PGNet_lite](ocr/pgnet_lite.md)** — PaddleOCR **PGNet** 端到端方案（检测框 + 字符 + 方向一次输出）的**轻量化改写 <sub>`17.8 KB`</sub>
- **[车牌检测](ocr/plate_det.md)** — 车牌场景的**专用轻量检测** —— YOLOv5n 的 **0.5 宽度版 <sub>`10.0 KB`</sub>
- **[车牌识别](ocr/plate_rec.md)** — 检测到的车牌框 → 识别字符。**CRNN**（CNN + RNN + CTC） <sub>`11.3 KB`</sub>
- **[PP-OCRv6](ocr/ppocrv6.md)** — 百度飞桨 OCR 主力版本，**tiny/small/medium** 三档 <sub>`13.1 KB`</sub>

### 3D / 点云 / 车道（4 篇）

> CenterPoint、SqueezeSegV3、BEV-LaneDet（Paddle3D 移植，权重逐命名对齐）

- **[BEV-LaneDet](pc/bev_lanedet.md)** — 单目 3D 车道线检测**；用 ResNet34 提图特征，再用 **FCTransform <sub>`12.1 KB`</sub>
- **[CenterPoint-Pillars](pc/centerpoint.md)** — LiDAR 点云 **3D 检测**的 SOTA 基线；用 **2D pillar** 把无序点云压成伪图像 <sub>`12.1 KB`</sub>
- **[车道线](pc/lane_row_seg.md)** — 本框架内置的**两条轻量 2D 车道线方案**（不带 3D）： <sub>`11.9 KB`</sub>
- **[SqueezeSegV3](pc/squeezesegv3.md)** — LiDAR **点云语义分割**的高效网络；把点云球面投影为 **range image <sub>`10.9 KB`</sub>

### 音频 SOTA（4 篇）

> 语音分类 / 说话人 / 关键词 / TTS（PaddleSpeech + kokoro 移植，四条验收全过）

- **[ECAPA-TDNN](audio/ecapa_tdnn.md)** — 说话人验证（Speaker Verification）的 SOTA 级 TDNN 架构；用 <sub>`10.2 KB`</sub>
- **[kokoro-82M](audio/kokoro.md)** — 82M 参数的高质量 TTS**（Text-to-Speech），架构 = **PL-BERT 文本编码器 + <sub>`14.7 KB`</sub>
- **[MDTC](audio/mdtc.md)** — 小足迹**关键词唤醒**（Keyword Spotting）模型；用**多尺度膨胀时序卷积 <sub>`10.6 KB`</sub>
- **[PANNs CNN14](audio/panns_cnn14.md)** — 音频模式识别（AudioSet 527 类）的经典 SOTA 基线；**CNN14** 是 PANNs 家族 <sub>`10.5 KB`</sub>

### 时间序列（22 篇）

> 5 个任务 / 21 个模型 / 4 个变点算法（PaddleTS 移植，fp64 判定）

- **[AutoEncoder](ts/anomaly_ae.md)** — 最基础的异常检测：**重建误差**大即为异常 <sub>`3.7 KB`</sub>
- **[Anomaly Transformer](ts/anomaly_at.md)** — 关联差异**（Association Discrepancy）：用「注意力分布与高斯先验的 KL」作为异常判据 <sub>`4.5 KB`</sub>
- **[MTAD-GAT](ts/anomaly_mtadgat.md)** — 双图注意力**（特征图 + 时间图）+ GRU 的多任务异常检测 <sub>`3.9 KB`</sub>
- **[USAD](ts/anomaly_usad.md)** — 对抗式双解码器**自编码器：用两个 decoder 对抗训练提升鲁棒性 <sub>`3.5 KB`</sub>
- **[VAE](ts/anomaly_vae.md)** — 在 AE 上加**变分推断**：隐空间建模为高斯，重建更稳定 <sub>`3.2 KB`</sub>
- **[变点 / 概念漂移检测](ts/changepoint.md)** — 纯统计工具**（无模型、无训练）：流式监控数据是否发生结构性变化 <sub>`4.1 KB`</sub>
- **[DeepAR](ts/deepar.md)** — 概率预测**：输出高斯分布参数，可采样得到任意区间 <sub>`3.9 KB`</sub>
- **[DLinear](ts/dlinear.md)** — 质疑 Transformer 的极简模型**：序列分解 + 单层线性，5K 参数打赢一堆 Transformer <sub>`4.2 KB`</sub>
- **[Informer](ts/informer.md)** — 长序列预测的**稀疏注意力**方案（ProbSparse），但含算法固有随机性 <sub>`4.2 KB`</sub>
- **[LSTNet](ts/lstnet.md)** — skip-RNN 建模周期**：CNN 抓短期 + RNN 抓长期 + 跳跃 RNN 抓季节性 <sub>`4.0 KB`</sub>
- **[MLP](ts/mlp.md)** — 最朴素的基线：展平窗口后全连接，**没有时序结构 <sub>`3.5 KB`</sub>
- **[NBEATS](ts/nbeats.md)** — 可解释的纯 MLP 预测器：把预测分解成 trend/season 两条基函数路径 <sub>`4.7 KB`</sub>
- **[NHiTS](ts/nhits.md)** — NBEATS 的**多率**升级版：分层插值 + 多分辨率采样，长周期预测更强 <sub>`4.3 KB`</sub>
- **[RNN](ts/rnn.md)** — 最经典的循环结构，作为对照与轻量方案 <sub>`3.8 KB`</sub>
- **[SCINet](ts/scinet.md)** — 下采样后再卷积**：把长序列变短以降低复杂度，递归构建 even/odd 采样树 <sub>`4.3 KB`</sub>
- **[TCN](ts/tcn.md)** — 因果膨胀卷积**：并行度高、感受野指数增长，适合在线流式预测 <sub>`4.2 KB`</sub>
- **[TFT](ts/tft.md)** — ⭐ **功率预测首选**：可解释 + 多 horizon + 分位数 + 最强协变量建模 <sub>`5.2 KB`</sub>
- **[Transformer](ts/transformer.md)** — 标准自注意力编码器，建模任意距离依赖 <sub>`3.9 KB`</sub>
- **[时序分类](ts/ts_classify.md)** — 把一段序列判成某个**工况/故障类型 <sub>`4.1 KB`</sub>
- **[表示学习](ts/ts_embed.md)** — 自监督**学通用时序表征，再送下游分类/聚类/预测 <sub>`4.3 KB`</sub>
- [时序任务总览与选型](ts/ts_overview.md) <sub>`4.6 KB`</sub>
- **[RUL 剩余寿命预测](ts/ts_rul.md)** — 预测设备**还能运行多少周期**（预测性维护核心） <sub>`5.3 KB`</sub>

### 基础设施（3 篇）

> 训练器统一口径、数据格式、对齐方法论 —— 读模型文档前建议先读这三篇

- **[对齐方法论](infra/align_method.md)** — 与上游（ultralytics / PaddleTS / Paddle3D / PaddleSpeech / PaddleOCR）对齐的**… <sub>`2.9 KB`</sub>
- **[数据格式与 YAML 配置](infra/data_formats.md)** — 每个任务吃什么数据、配置怎么写 <sub>`2.6 KB`</sub>
- **[训练器与统一口径](infra/trainer.md)** — 唯一**的训练/评估循环 —— 26 个任务共用，口径在这里统一 <sub>`3.9 KB`</sub>

---

## 阅读建议

| 你是 | 建议顺序 |
|---|---|
| 新用户 | `infra/trainer` → `infra/data_formats` → 目标家族 → `infra/align_method` |
| 选模型 | `ts/ts_overview`（时序）→ 目标家族各篇的「选型建议」+「参考对比」 |
| 复现上游 | `infra/align_method`（五条验收 + fp64 判定法）→ 目标篇的「公开指标」 |
| 移植/改模型 | 目标篇「网络结构」+「已知问题」（坑都在这） |

---

## 统计

| 家族 | 篇数 |
|---|---|
| [2D 检测家族](detect/) | 9 |
| [任务头](task/) | 9 |
| [OCR / 车牌](ocr/) | 4 |
| [3D / 点云 / 车道](pc/) | 4 |
| [音频 SOTA](audio/) | 4 |
| [时间序列](ts/) | 22 |
| [基础设施](infra/) | 3 |
| **合计** | **55** |
