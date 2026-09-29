# 单目深度估计（Depth）

> **定位**：多尺度特征融合后输出**单通道 log-depth**，`depth = exp(logit)`（米）。
> **任务**：`depth` · **头**：`Depth`（仅 YOLO26 家族提供 `yolo26-depth.yaml`）
> **权重**：`yolo26{n,s,m,l,x}-depth.pth` 官方存在；本框架**尚未做权重/数值对齐验证**（见 §6）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **无独立论文**。Ultralytics YOLO26 引入 `-depth` 任务；本框架为**自实现多尺度融合解码器** |
| 参考 | 单目深度估计通用范式（log-depth 回归 + L1/SILog 损失、δ 指标），**非** Depth Anything / MiDaS 的直接复刻 |
| 官方代码 | https://github.com/ultralytics/ultralytics |
| 官方权重 | `yolo26{n,s,m,l,x}-depth.pt` |
| 本框架实现 | 头：`torchkiln/nn/modules.py::Depth`<br>结构：`torchkiln/cfg/models/26/yolo26-depth.yaml`<br>损失/指标：`torchkiln/depth.py` · 任务：`torchkiln/tasks/depth.py` · 数据：`torchkiln/data/depth.py` |
| 移植方式 | YAML 逐层复刻（backbone/neck 与 detect 相同）+ **自实现解码头**（非逐键对齐，未验证） |

### 要解决的问题

- 自动驾驶/机器人/AR 需要**稠密深度**（每个像素到相机距离），而非稀疏 LiDAR 点。
- 直接回归绝对深度数值范围大、难优化。通用做法：**预测 log-depth**，用 L1（保全局尺度）
  + 可选的 **SILog**（尺度不变）损失训练，评价用 **δ1/δ2/δ3、AbsRel、RMSE**。
- Ultralytics YOLO26 把深度估计作为独立任务，复用检测 backbone。

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,3,H,W)
  │ backbone（与 detect 相同：Conv/C3k2/SPPF/C2PSA）→ P3(8),P4(16),P5(32)
  │ neck（PAN）→ T1,T2,T3
  ▼ Depth
  ┌──────────────────────────────────────────────────────┐
  │  lat3/lat4/lat5 : 1×1 Conv(c→hidden)                  │
  │  fuse = Conv(hidden,3×3) → Conv(hidden,3×3)           │
  │  head = Conv2d(hidden, 1, 1)                          │
  └──────────────────────────────────────────────────────┘
     融合：y = lat3(p3) + Upsample×2(lat4(p4)) + Upsample×4(lat5(p5))
     输出 (B,1,H/8,W/8) log-depth；(DepthModel 另一次 ×2 → stride 4)
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | ultra 名字 |
|---|---|---|---|---|
| `lat3/lat4/lat5` | `Conv(c, hidden, k=1)` | 各尺度→`hidden` | 统一通道 | `lat*` |
| `fuse` | `Conv(hidden,hidden,3)×2` | 同维 | 融合 | `fuse` |
| `head` | `nn.Conv2d(hidden, 1, 1)` | →`1` | 单通道 log-depth | `head` |
| `DepthModel.up` | `Upsample(scale_factor=2)` | stride 8→4 | 提分辨率 | — |

> yaml 中 `Depth` 的 `c_mid=256`（`[[16, 19, 22], 1, Depth, [256]]`）。
> `Depth` 头**无额外参数解析**（`parse_model` 里 `"Depth"/"Semantic"` 走 `pass` 分支）。

### 2.3 与 detect / semantic 头的差异

| 项 | Detect | SemanticSegment | **Depth** |
|---|---|---|---|
| 输出 | 框+类 | `(B,nc,H/4,W/4)` | **`(B,1,H/4,W/4)` log-depth** |
| 任务 | 定位 | 逐像素分类 | **逐像素回归** |
| 损失 | box+cls+dfl | CE(+Dice) | **log-L1(+SILog)** |
| 指标 | mAP | mIoU | **δ1/2/3、AbsRel、RMSE** |
| 只用于 | 全家族 | yolo11/26 | **仅 yolo26** |

---

## 3. 输出与后处理（★ 重点）

### 3.1 模型输出 vs 深度值

```
logit = model(x)                       # (B, 1, H/4, W/4) log-depth，无激活
# DepthPostProcess：
depth = exp(logit).clamp(min_depth=1e-3, max_depth=1e3)     # 米
# 若需要，先双线性插值到原图分辨率
```

> 模型预测的是 **log-depth**，因此头输出**不加任何激活**（无 sigmoid/relu）。

### 3.2 数据集格式（`torchkiln/data/depth.py::DepthDataset`）

```
<data_dir>/images/<split>/xxx.jpg     # RGB
<data_dir>/depth/<split>/xxx.png      # uint16，默认毫米（depth_scale=1000）
<list.txt>                            # 只列图片相对路径
```

| 项 | 说明 |
|---|---|
| 深度存储 | **uint16 PNG**，默认**毫米**（`depth_scale=1000` → 米） |
| 无效像素 | **`0` 表示无效**（损失/指标里 `valid = target > 0` 过滤） |
| 路径映射 | `images/...` 自动替换为 `depth/....png` |
| 也支持 | `.npy` 深度（float） |
| 增广 | `DenseAugmenter`（图像与深度同步变换） |
| letterbox | 无增广时用 `letterbox`（深度用 nearest 插值同步） |

### 3.3 损失（`DepthLoss`）

```
valid  = target > 0
pred_log = logit[valid]
gt_log   = log(clamp(target[valid], min_depth))
loss = L1(pred_log, gt_log)                                   # 主项（保绝对尺度）
if silog_weight > 0:
    d = pred_log - gt_log
    silog = mean(d^2) - 0.5 * mean(d)^2                       # 尺度不变
    loss += silog_weight * clamp(silog, min=0)
```

### 3.4 指标（`DepthMetric`）

| 指标 | 公式 |
|---|---|
| `delta1/2/3` | `max(pred/gt, gt/pred) < 1.25^k` 的像素占比（k=1,2,3） |
| `abs_rel` | `Σ|pred-gt| / Σ gt` |
| `rmse` | `sqrt(mean((pred-gt)^2))` |
| 有效像素 | `gt > 0` |

---

## 4. 配置与用法

### 4.1 最小配置（`configs/yolo/yolo26-depth.yml`）

```yaml
Architecture:
  task: depth
  yaml_file: torchkiln/cfg/models/26/yolo26-depth.yaml
  scale: n
  Head: {num_classes: 1, reg_max: 1, end2end: true}
Loss:  {name: DepthLoss, silog_weight: 0.0, min_depth: 1e-3}
Metric: {name: DepthMetric, main_indicator: delta1}
PostProcess: {name: DepthPostProcess, min_depth: 1e-3, max_depth: 1e3}
Train:
  dataset: {name: DepthDataset, data_dir: datasets/depth_demo, label_file_list: [.../train.txt],
            depth_scale: 1000, transform: {image_size: 256}}
```

> ⚠️ **`depth_scale` 必须与标注单位一致**：毫米标注用 1000；若用米标注则设 1.0。

### 4.2 四条链路

```bash
tkiln train   -c configs/yolo/yolo26-depth.yml -o Global.pretrained_model=yolo26n-depth
tkiln val     -c configs/yolo/yolo26-depth.yml --weights output/yolo26-depth/best_accuracy.pth
tkiln predict -c configs/yolo/yolo26-depth.yml --weights ... --input imgs/   # 输出深度图
tkiln export  -c configs/yolo/yolo26-depth.yml --weights ... --onnx
tkiln check   -c configs/yolo/yolo26-depth.yml
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ |
|---|---|
| yolo26n-depth | 2.57 |
| yolo26s-depth | 10.01 |
| yolo26m-depth | 21.90 |
| yolo26l-depth | 26.30 |
| yolo26x-depth | 58.99 |

¹ 取自 `torchkiln/cfg/models/26/yolo26-depth.yaml` 的 scales 注释（与 sem 同 backbone）。
**FLOPs / 本机耗时 / 显存 未在本框架侧统计（不编造）。**

---

## 6. 公开指标

| 项 | 状态 |
|---|---|
| **权重加载对齐** | **未验证**（AGENTS 未记录 `yolo26*-depth` 的 missing/unexpected；官方权重存在） |
| **前向逐层对齐** | **未验证** |
| **端到端 δ1/AbsRel/RMSE** | **未统计** |
| 结构 | backbone/neck 与 detect 相同（已对齐）；头为**自实现多尺度融合**，**未与 ultra 逐键对齐** |

> ⚠️ **如实说明**：任务已实现且可训练/评估/推理（`smoke_all` 覆盖该配置），
> 但**未做与 ultralytics 的对齐验证**，故本页不提供"官方 vs 框架"数值表。
> 需要注意：官方 yolo26-depth 头可能与本框架的 `Depth` 头在结构上不一致（未核对）。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 单目稠密深度（室内/驾驶） | **yolo26n-depth** | 复用检测 backbone，轻 |
| 需要 3D 框（LiDAR/点云） | 见 `docs/models/pc/centerpoint.md` | CenterPoint-Pillars |
| 点云语义分割 | 见 `docs/models/pc/squeezesegv3.md` | range-view |
| 只需 2D 框 | detect | 更简单 |
| 相对深度（无绝对尺度） | 用 `silog_weight>0` | 尺度不变项 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs semantic** | 结构几乎相同（多尺度融合），差别只在输出通道（1 vs nc）与损失/指标 |
| **vs Depth Anything / MiDaS** | 那些是专用大模型（ViT/DPT）；本头是 YOLO backbone + 轻量融合，非复刻 |
| **vs 双目/LiDAR** | 本头是单目，成本低但精度/尺度弱于几何法 |
| **绝对 vs 相对深度** | L1（保尺度）vs SILog（尺度不变）；两者可叠加 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **对齐状态未验证**：头为自实现，**未做权重加载/数值对齐**（见 §6）。
- ⚠️ **仅 yolo26 家族有 `-depth.yaml`**（yolo11/v8 无）。
- ⚠️ **`depth_scale` 与标注单位必须一致**，否则损失数量级错误。
- ⚠️ **`0` 是无效像素**：损失与指标都按 `gt > 0` 过滤；标注时要保证背景/无效区为 0。
- ⚠️ **log-depth 输出无激活**，后处理才 `exp`；不要对 logits 先 sigmoid。
- ⚠️ 深度增广必须**图像与深度同步**（`DenseAugmenter`）；letterbox 用 nearest 插值深度。
- ⚠️ **评估 fp32**：`_evaluate_loop` 强制 `autocast(enabled=False)`。
