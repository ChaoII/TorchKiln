# 视频分类（TSM-LCNet）

> **定位**：**视频片段级**行为/场景分类 —— 2D CNN backbone + **Temporal Shift（TSM）**
> + **分段共识（segment consensus）**，无需 3D 卷积，算力友好。
> **任务**：`video_cls`（别名 `video`） · **模型**：`TSMVideoNet`（backbone = `PPLCNetX1_0`）
> **权重**：**无官方权重**（本框架自实现，未做权重对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **TSM：Temporal Shift Module for Efficient Video Understanding**（arXiv:1811.08383，ICCV 2019）；backbone **PP-LCNet**（arXiv:2109.15099） |
| 机构 | TSM：MIT 等；PP-LCNet/PP-TSM：百度 PaddleVideo |
| 官方代码 | TSM：https://github.com/mit-han-lab/temporal-shift-module ；PP-TSM：PaddleVideo |
| 官方权重 | **本框架未使用官方权重**（任务为自研端口） |
| 本框架实现 | 模型/数据/损失/指标全部在 `torchkiln/tasks/video_cls.py`<br>backbone 复用 `torchkiln/nn/attribute.py::PPLCNetX1_0`<br>配置 `configs/video/video_cls.yml` |
| 移植方式 | **按论文思想自实现**（非逐键复刻），未做与官方权重的对齐验证 |

### 要解决的问题

- 安防/工业需要**一段视频**的行为判断（打架、攀爬、摔倒、越界）。
- 3D 卷积（I3D）参数大、算力高、预训练数据稀缺。**TSM** 的思路：
  在 2D CNN 的通道上**沿时间轴平移一部分通道**（前后各挪一帧，**无参数、零 FLOPs**），
  让 2D 卷积"看到"相邻帧 → 兼具 2D 的效率与 3D 的时序建模能力。
- **PP-TSMv2** 进一步用 **LCNetV2** 这类轻量 backbone + **分段共识**（把长视频切成 N 段、每段取 1 帧，
  各段特征平均）实现高效长时建模 —— 本框架即按此设计（backbone 用 `PPLCNetX1_0`）。

---

## 2. 网络结构

### 2.1 整体框图

```
Input  (N, T, C, H, W)                 # T = num_segments × frames_per_seg
  │ reshape → (N*T, C, H, W)
  ▼ TemporalShift (无参数, shift_div=8)
  │   1/8 通道后移一帧、1/8 通道前移一帧、其余不变
  ▼ PPLCNetX1_0 backbone（轻量 2D CNN，depthwise separable + SE）
  │   → (N*T, C_out, h, w)
  ▼ AdaptiveAvgPool2d(1) → flatten → view(N, T, -1) → .mean(1)   # 分段共识
  ▼ Dropout → Linear(C_out, num_classes)
  输出 (N, num_classes) logits
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | 论文名字 |
|---|---|---|---|---|
| **`TemporalShift`** | `out[:,1:,:fold]=x[:,:-1,:fold]`；`out[:,:-1,fold:2fold]=x[:,1:,fold:2fold]`；其余直通 | `(N*T,C,H,W)→同形` | ★ 时序位移，**0 参数** | Temporal Shift |
| `PPLCNetX1_0` | `conv1` + 5 个 `DepthwiseSeparable` 段（末两段带 SE） | `(N*T,3,224,224)→(N*T,512 or 256,h,w)` | 2D 特征提取 | PP-LCNet |
| 分段共识 | `mean(over T)` | `(N,T,C)→(N,C)` | ★ 长时聚合 | Segment Consensus |
| `fc` | `Linear(C_out, num_classes)` | `(N,C)→(N,nc)` | 分类 | — |
| `dropout` | `Dropout(p)`（可配） | — | 正则 | — |

> `TemporalShift` 在 `shift_div<=1`、`nt % n_segment != 0`、或 `fold<=0` 时**直通**（安全退化）。

### 2.3 与图像分类的差异

| 项 | classify | **video_cls** |
|---|---|---|
| 输入 | `(B,3,H,W)` 单帧 | **`(N,T,3,H,W)` 片段** |
| 时序 | 无 | **TemporalShift + 分段共识** |
| backbone | YOLO backbone | **PPLCNetX1_0（轻量 CPU 友好）** |
| 采样 | 无 | **num_segments × frames_per_seg** |
| 指标 | top-1/5 | top-1/5 |

---

## 3. 输出与后处理（★ 重点）

### 3.1 输出

```
logits = model(clips)                 # (N, num_classes)
prob   = softmax(logits, dim=1)
label  = argmax(prob); score = max(prob)
```

后处理 `VideoClsPostProcess` 返回每样本 `{"logits", "label", "score"}`。

### 3.2 片段采样与前处理（`VideoClsDataset`）

| 项 | 说明 |
|---|---|
| 数据来源 | **视频文件**（`cv2.VideoCapture`）或**帧目录**（按文件名排序读图） |
| 采样 | 分 `num_segments` 段，每段取 `frames_per_seg` 帧（训练**随机**取、评估**中心/均匀**取） |
| 预处理 | 短边 resize 到 `image_size` → 训练随机 crop / 评估中心 crop 到 `crop_size` → RGB → 除 255 → ImageNet 归一化 |
| 增广 | 随机水平翻转（`augment.flip`，训练，默认 0.5） |
| 标签文件 | yolo 风格：`path/to/video.mp4 <action_id>` |
| 指标 | `VideoClsMetric`：`acc`(top-1) / `top5` |
| 损失 | `VideoClsLoss`：CrossEntropy（`label_smoothing` 可配） |

---

## 4. 配置与用法

### 4.1 最小配置（`configs/video/video_cls.yml`）

```yaml
Architecture:
  task: video_cls
  algorithm: TSM-LCNet
  Backbone: {name: TSM, num_segments: 8, shift_div: 8, scale: 0.5}
  Head: {num_classes: 3, dropout_prob: 0.5}
Loss:  {name: VideoClsLoss, label_smoothing: 0.1}
Metric: {name: VideoClsMetric, main_indicator: acc, topk: 5}
PostProcess: {name: VideoClsPostProcess}
Train:
  dataset: {name: VideoClsDataset, data_dir: datasets/video_cls_demo,
            label_file_list: [.../train.txt],
            transform: {num_segments: 8, frames_per_seg: 1, image_size: 80, crop_size: 64},
            augment: {flip: 0.5}}
  loader: {batch_size_per_card: 4, num_workers: 0}
```

> `T = num_segments × frames_per_seg`。`Backbone.scale`（如 0.5）控制 `PPLCNetX1_0` 宽度。
> ⚠️ `num_segments` 必须与 `T` 的采样设置一致，否则 `TemporalShift` 形状不匹配（会安全直通，但失去时序）。
> Windows 下建议 `num_workers: 0`（视频解码 + 多 worker 易 OOM）。

### 4.2 四条链路

```bash
tkiln train   -c configs/video/video_cls.yml
tkiln val     -c configs/video/video_cls.yml --weights output/video_cls/best_accuracy.pth
tkiln predict -c configs/video/video_cls.yml --weights ... --input videos/
tkiln check   -c configs/video/video_cls.yml
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| backbone | `PPLCNetX1_0`（scale 可调，末通道 `make_divisible(512*scale)`） |
| 头 | `Linear(C_out, num_classes)` |
| 参数量 | **未统计**（本框架未记录该任务的参数计数） |
| FLOPs / 本机耗时 / 显存 | **未统计** |

> ⚠️ `TemporalShift` 本身**零参数、零 FLOPs**；算力主要在 backbone × T 帧上。

---

## 6. 公开指标

| 项 | 状态 |
|---|---|
| **官方权重对齐** | **无官方权重可对**（本端口为自实现） |
| **Kinetics / UCF101 等公开基准** | **未统计**（未在本框架侧跑过公开视频数据集） |
| 设计目标 | TSM 的**工程可用性**（安防行为分类），非刷新公开榜 |

> ⚠️ **如实说明**：这是本框架**自研端口**，没有可直接对标的官方权重；没有任何"官方 vs 框架"数值。
> 若要评估质量，应先在自有视频数据集上训练并报告 top-1/top-5。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 短视频片段行为分类 | **本任务（TSM-LCNet）** | 2D 卷积、轻量、可改造 |
| **骨架**行为识别（已有 pose 输出） | 见 `docs/models/task/pose_action.md` | ST-GCN，输入是骨架序列 |
| 单帧图像分类 | 见 `docs/models/task/classify.md` | 更简单 |
| 需要 3D 卷积精度 | 自行接 I3D/SlowFast | 本框架未内建 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs classify** | 多时间维；TemporalShift + 分段共识 |
| **vs 3D CNN（I3D/SlowFast）** | TSM 用 2D 卷积 + 位移，参数/算力低得多，精度接近 |
| **vs pose_action（ST-GCN）** | 输入是**视频帧**（RGB）；ST-GCN 输入是**骨架序列** |
| **vs TimeSformer/ViViT** | 那些是 Transformer 视频模型，更重；本端口追求轻量 |
| **vs PP-TSM 官方** | 设计同源（PP-TSMv2 也用 LCNet + 分段共识），但本端口是自实现、未逐键复刻 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **无官方权重、无对齐验证**：本端口为自研，报告数值前必须加"未与官方对齐"的说明。
- ⚠️ **`num_segments` 与采样帧数必须匹配**：`T` 不等于 `num_segments` 的整数倍时 `TemporalShift` 直通。
- ⚠️ **Windows 视频解码**：`num_workers>0` 可能 "worker exited unexpectedly"；建议 0。
- ⚠️ **帧目录/视频均可**：帧目录按文件名排序；视频用 `cv2.VideoCapture` 全读入内存（长视频内存吃紧）。
- ⚠️ **无时序增广**（仅空间翻转）；如需时间裁剪/跳帧，需扩展 `_sample_idx`。
- ⚠️ **评估 fp32**：`_evaluate_loop` 强制 `autocast(enabled=False)`（该配置 `amp: false`）。
