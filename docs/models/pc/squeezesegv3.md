# SqueezeSegV3

> **定位**：LiDAR **点云语义分割**的高效网络；把点云球面投影为 **range image**，
> 用**空间自适应卷积（SAC）** 代替普通卷积 —— 同一 kernel 在不同空间位置权重不同。
> **任务**：`pc_seg`（`Architecture.algorithm: squeezesegv3`）
> **权重**：`squeezesegv3_rangenet53_semantickitti.pth`（`weights/`，ModelScope `ChaoII0987/TorchKiln → pretrained/`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation** |
| arXiv | **2004.01803**（2020-04-03，已核实标题/作者） |
| 作者 | Chenfeng Xu、Bichen Wu、Zining Wang、Wei Zhan、Peter Vajda、Kurt Keutzer、Masayoshi Tomizuka |
| 会议 | ECCV 2020 |
| 官方代码 | https://github.com/chenfengxu714/SqueezeSegV3 |
| 本框架参考实现 | **Paddle3D** `squeezesegv3_rangenet53_semantickitti`（SAC-RangeNet53） |
| 本框架实现 | `torchkiln/nn/sac_rangenet.py`（网络 + loss + 后处理）、`torchkiln/data/pc.py`（`project_range_image`）、<br>`torchkiln/tasks/pc_seg.py`、`torchkiln/models/pc.py` |
| 移植方式 | PyTorch 逐命名移植 Paddle3D + 权重逐键对齐（**missing=0 / unexpected=0，524 张量**） |

### 论文要解决的问题

1. **球面投影后的空间非均匀性**：range image 的每一行是固定俯仰角，投影导致
   **近处点数密、远处稀**，且同一 kernel 在图像不同区域（上/下、左/右）
   对应的**物理尺度完全不同** → 普通卷积的权重共享不合适。
2. **低延迟需求**：自动驾驶要求实时，参数量/算力受限。

SqueezeSegV3 的核心创新：
- **空间自适应卷积（SAC）**：kernel 权重由**该点空间坐标 `(x,y,z)`** 动态生成，
  等价于「卷积核随位置变化」，解决了投影空间非均匀问题；
- 在多个分辨率上做 SAC → **多尺度解码**，兼顾小/大目标；
- 以更少参数达到 SOTA（SemanticKITTI mIoU）。

### 关键结论

| 数据集 | 指标 | 原文 |
|---|---|---|
| SemanticKITTI test | **mIoU** | **SqueezeSegV3-53 ≈ 55.9**（论文主结果） |
| SemanticKITTI | 每类 IoU | car/road/…；公路类最高（~96% road） |
| 速度 | FPS | 论文强调**实时**（远高于 RangeNet++） |

> ⚠️ 本框架**尚未跑过 SemanticKITTI 的 mIoU**（数据集 80.9GB，未下载），
> 因此第 6 节的「本框架实测」只有**前向/损失数值对齐**，无 mIoU。

---

## 2. 网络结构

### 2.1 整体框图（SAC-RangeNet53）

```
range image (N, 5, H, W)   # 5 通道 = [range, x, y, z, remission]
  │  （球面投影：fov_up=+3°, fov_down=-25°，最近点优先）
  ▼
_Encoder
  · conv_1: Conv3×3(5→32)+BN+LeakyReLU
  · 5 个 _EncoderStage，num_stage_blocks = (1,2,8,8,4)
      每 stage = n×_SACISKBlock，前 3 个 stage 末尾 `_DownsampleBlock`（stride=(1,2)）
      xyz 与 feature 同步下采样（xyz 用 bilinear 插值）
  · 通道 (32,64) (64,128) (128,256) (256,256) (256,256)
  ▼ 5 个尺度 + short_cuts（每 stage 前的 feature.detach()）
_Decoder
  · up_channels = (256,256) (256,256) (256,128) (128,64) (64,32)
  · 前 2 个 stage 不 upsample（Conv3×3），后 3 个用 Deconv[1,4] stride[1,2]
  · 每个 upsample stage：feature + short_cuts.pop()
  ▼ 5 个尺度特征
heads: 5× Conv(1×1)（最后一层 3×3）→ 每尺度 num_classes logits
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 |
|---|---|---|---|
| `_ConvBNLayer` / `_DeconvBNLayer` | Conv / ConvTranspose + BN | — | 基础块；**bias 默认存在**（对齐 Paddle `bias=None`） |
| **`_SACISKBlock`** | `F.unfold(feature,3,padding=1)`（取 9 邻域）→ `sigmoid(Conv7×7(xyz))` 得 attention → 相乘 → `position_mlp`（1×1→ReLU→3×3→ReLU） → **残差 + feature** | (xyz, feat) → (xyz, feat) | ★ 空间自适应块 |
| `_DownsampleBlock` | Conv3×3 stride=(1,2) + LeakyReLU；xyz 同步 `F.interpolate` | — | 下采样（只压宽不压高） |
| `_EncoderStage` | n×`_SACISKBlock` (+ 可选 downsample) + `Dropout2d` | — | 编码级 |
| `_InvertedResidual` | Conv1×1 → LeakyReLU → Conv3×3 → LeakyReLU，**残差** | — | 解码块 |
| `_DecoderStage` | Deconv[1,4]s[1,2]（或 Conv3×3）→ LeakyReLU → `_InvertedResidual` | — | 解码级 |
| `SqueezeSegV3` | `SACRangeNet` + 5 个 1×1/3×3 head | (N,5,H,W)→ list[5] | ★ 整网 |

- `num_layers=53`（`num_stage_blocks=(1,2,8,8,4)`）；也支持 `21`（`(1,1,2,2,1)`）。
- `bn_momentum` 默认 `0.99`（构造参数），`SACRangeNet` 默认 `0.99` —— 注意
  注册表的 `_ConvBNLayer` 默认是 `0.9`（→ torch `momentum=0.1`）。

### 2.3 与同类的差异

| 对比对象 | 差异点 | 为什么 | 效果 |
|---|---|---|---|
| **RangeNet++** | 普通 2D 卷积 | 未处理投影空间非均匀 | SqueezeSegV3 mIoU 更高 |
| **SqueezeSeg(V1/V2)** | V3 加 **SAC** + 更深 backbone | 提升表达力 | 精度↑ |
| **PointNet/PointNet++** | 直接在点上算 | 无投影信息损失，但慢 | V3 快得多、可实时 |
| **`_SACISKBlock` vs 论文 SAC** | 本实现用 `unfold(9 邻域) × sigmoid(xyz-attention)`，属于 Paddle3D 的 SAC 变体 | 对齐 Paddle3D 权重 | 权重可 1:1 加载 |

---

## 3. 输出与后处理

### 3.1 输出张量

`SqueezeSegV3.forward(x)`（推理/训练）返回 **list，5 个尺度的 logits**：

| 尺度 | 形状 | 通道来源 |
|---|---|---|
| s0（最粗） | `(B, C, H, W)` | 256ch 特征 |
| s1 | `(B, C, H, W)` | 256ch |
| s2 | `(B, C, H, W)` | 128ch |
| s3 | `(B, C, H, W)` | 64ch |
| s4（全分辨率） | `(B, C, H, W)` | 32ch（3×3 head） |

### 3.2 球面投影（`project_range_image`）

```
depth = ||(x,y,z)|| ；过滤 depth ≤ 1e-3
yaw   = -atan2(y, x)
pitch =  arcsin(z / depth)
fov   = (|fov_up| + |fov_down|) * π/180
px = (0.5 * (yaw/π + 1) * W)
py = (1 - (pitch + |fov_down|*π/180) / fov) * H

通道 = [range=depth, x, y, z, remission]        # ← Paddle3D 顺序
多点到同像素：按 depth 从远到近写入 → 最近点**最后写、胜出**
```

- 默认 `fov_up=3.0°`、`fov_down=-25.0°`（KITTI HDL-64E 典型）。
- 无点云/全过滤时返回全 0 图 + 全 `ignore_index` 标签（不破坏 batch）。

### 3.3 损失 / 解码

| 项 | 说明 |
|---|---|
| 损失 | `SqueezeSegV3Loss` = **5 尺度加权 NLL**（`F.log_softmax` + `NLLLoss(ignore_index=255)`）；对 target 用 `F.interpolate(mode="nearest")` 对齐每尺度 |
| 回归方式 | 无 DFL/anchor；**纯逐像素分类** |
| 后处理 | `SqueezeSegV3PostProcess`：取**全分辨率尺度（list[-1]）** → `argmax(1)`；若 `size` 给定则 `interpolate` 到该尺寸 |
| ignore | `ignore_index=255` |

---

## 4. 配置与用法

```yaml
# 最小配置
Architecture:
  model_family: pc
  task: pc_seg
  algorithm: squeezesegv3
  Head: {num_classes: 5, in_channels: 5, num_layers: 53, encoder_dropout_prob: 0.01, decoder_dropout_prob: 0.01}
Loss:        {name: SqueezeSegV3Loss, ignore_index: 255}
Metric:      {name: SemMetric, main_indicator: mIoU, num_classes: 5, ignore_index: 255}
PostProcess: {name: SqueezeSegV3PostProcess}
Train:
  dataset:
    name: PointCloudDataset
    data_dir: datasets/pc_demo
    label_file_list: [datasets/pc_demo/train.txt]
    range_image: true          # ★ 开 range image 模式
    range_image_h: 64
    range_image_w: 256
    num_classes: 5
    ignore_index: 255
```

数据目录布局：

```
<data_dir>/train.txt                 # 相对点云路径清单
<data_dir>/clouds/train/*.bin|.npy   # float32 N×3 或 N×4
<data_dir>/labels/train/*.txt|.npy   # 每点一个整型标签
```

```bash
tkiln check -c configs/pc/squeezesegv3-pcseg.yml
tkiln train -c configs/pc/squeezesegv3-pcseg.yml -o Global.pretrained_model=squeezesegv3_rangenet53_semantickitti
tkiln val   -c configs/pc/squeezesegv3-pcseg.yml --weights output/squeezesegv3-pcseg/best_accuracy.pth
```

> ⚠️ `smoke_all` 会强制 `batch=4` 且同进程连跑；range image 必须**足够小**
> （示例用 64×256，batch=1），否则直接 OOM 崩进程。

---

## 5. 规模与速度

| 项 | 值 | 来源 |
|---|---|---|
| 参数（`in_channels=5, num_classes=20, num_layers=53`） | **25.9596 M** | 本框架实算 |
| 权重体积 | **99.7 MB**（`weights/squeezesegv3_rangenet53_semantickitti.pth`） | AGENTS 记录 |
| 官方 `.pdparams` | 524 张量 | AGENTS 记录 |
| 推理耗时（本机） | **未统计** | — |
| 训练显存 | **未统计**（示例 64×256 batch=1 可跑） | — |

---

---

---

---

---

### 📊 FLOPs（实测）

| 项 | 值 |
|---|---|
| **FLOPs** | **130.51 GFLOPs** |
| **MACs** | **65.25 GMACs** |
| 参数量 | **25.960 M** |
| 输入规格 | `range-view 64x256` |
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
| SemanticKITTI 类 64 线点云分割 | **本模型** | 实时 + 精度高 |
| 只有稠密 BEV 栅格、要分割 | `algorithm: pointpillars`（BEV 分割头） | 无 range 投影 |
| 极小显存 | 用 `num_layers=21`（`(1,1,2,2,1)`）| 参数/显存更小 |
| 需要逐点标签（非像素） | 需自行把像素标签散回点 | 本框架目前是**像素级**分割 |
| 32 线 / 非 KITTI FOV | 改 `dataset.fov` | 默认 `[3.0, -25.0]` |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs RangeNet++** | 后者是普通卷积 + KNN 后处理；本模型 SAC 直接缓解投影非均匀 |
| **vs PointPillars（pc_seg）** | 本框架另有一条 `pointpillars-seg.yml`（BEV 柱级多类），两者任务相近但表征不同 |
| **vs Paddle3D 原版** | 权重/前向/损失**逐位对齐**（1e-5~1e-6） |
| **vs Cylinder3D / 稀疏卷积** | 后者精度更高但更慢，需稀疏卷积库；本模型纯 2D |

---

## 9. 已知问题 / 注意事项

- ⚠️ **Paddle `ConvBNLayer(bias=None)` = 默认有 bias**（仅 `bias=False` 才无）
  —— 本框架 `_ConvBNLayer(bias=None)` → `bias=(bias is not False)`，如果漏掉会权重形状不符。
- ⚠️ 本框架的 `_SACISKBlock` 是 **Paddle3D 的 SAC 变体**（unfold + xyz attention），
  不是论文里描述的「kernel 生成」形式；**以 Paddle3D 为准**（权重只此一份）。
- ⚠️ **`ignore_index=255`** 贯穿 loss/metric；数据集里空像素必须写 255。
- ⚠️ 标签是**像素级**（range image 分辨率），不是点级。
- ⚠️ **mIoU 未验证**（数据集未下载）；只有前向/损失对齐。
- ⚠️ `smoke_all` 里大 range image 会 OOM；务必用小图 + `batch=1`。
- ⚠️ 本框架**未集成 FLOPs 统计**。
