# 语义分割（SemanticSegment）

> **定位**：**逐像素分类**（不区分实例）—— 多尺度特征（P3/P4/P5）融合后输出
> `(B, nc, H/4, W/4)` 类别 logits，上采样回原图取 argmax。
> **任务**：`semantic`（别名 `sem`） · **头**：`SemanticSegment`
> **权重/权重对齐**：`yolo26*-sem.pth` 官方存在；本框架**尚未做权重加载/数值对齐验证**（见 §6）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **无独立论文**。Ultralytics YOLO26 引入 `-sem` 任务；本框架为**结构复刻 + 自实现解码头** |
| 参考 | 语义分割通用范式（FCN / 多尺度融合 / 逐像素 CE），**非** U-Net / DeepLab 的直接复刻 |
| 官方代码 | https://github.com/ultralytics/ultralytics |
| 官方权重 | `yolo26{n,s,m,l,x}-sem.pt`（Cityscapes 等） |
| 本框架实现 | 头：`torchkiln/nn/modules.py::SemanticSegment`<br>结构：`torchkiln/cfg/models/11/yolo11-sem.yaml`、`torchkiln/cfg/models/26/yolo26-sem.yaml`<br>损失/指标：`torchkiln/sem.py` · 任务：`torchkiln/tasks/semantic.py` · 数据：`torchkiln/data/sem.py` |
| 移植方式 | YAML 逐层复刻（backbone/neck 与 detect 相同）+ **自实现解码头**（非逐键对齐，未验证） |

### 要解决的问题

- 有些场景不需要区分实例，只需知道**每个像素属于哪类**（道路、天空、可行驶区域、车道）。
- 相比实例分割，语义分割**无 NMS、无实例匹配、无掩码系数**，更轻更快，适合可行驶区域/车道等密集场景。
- Ultralytics YOLO26 把语义分割作为独立任务，复用检测 backbone 的多尺度特征。

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,3,H,W)
  │ backbone（与 detect 相同：Conv/C3k2[/SPPF]/C2PSA）→ P3(8),P4(16),P5(32)
  │ neck（PAN，自顶向下+自底向上）→ T1,T2,T3
  ▼ SemanticSegment
  ┌──────────────────────────────────────────────────────┐
  │  lat3/lat4/lat5 : 1×1 Conv(c→hidden)  逐尺度降维      │
  │  fuse = Conv(hidden,3×3) → Conv(hidden,3×3)           │
  │  head = Conv2d(hidden, nc, 1)                          │
  └──────────────────────────────────────────────────────┘
     融合方式：y = lat3(x0) + Upsample(lat4(x1)) + Upsample(lat5(x2))   # 对齐到 P3 分辨率
     输出 stride 8 的 logits；(SemModel 另有一次 ×2 上采样 → stride 4)
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | ultra 名字 |
|---|---|---|---|---|
| `lat3/lat4/lat5` | `Conv(c, hidden, k=1)` | 各尺度→`hidden` | 统一通道 | `lat*` |
| `fuse` | `Conv(hidden,hidden,3)→Conv(hidden,hidden,3)` | 同维 | 融合 | `fuse` |
| `head` | `nn.Conv2d(hidden, nc, 1)` | →`nc` | 逐像素类别 | `head` |
| `SemModel.up` | `nn.Upsample(scale_factor=2, nearest)` | stride 8→4 | 提分辨率 | — |

> ⚠️ 头定义在 `SemanticSegment`（`__init__(nc, ch, hidden=128, reg_max)`）；
> `models/sem.py::SemModel` 在其后又加一次 ×2 上采样，最终 logits 在 **stride 4**。
> 两处都有实现（`models/sem.py` 与 `nn/modules.py`），按构建路径（手写模型 vs YAML 图模型）选用。

### 2.3 与 detect / segment 头的差异

| 项 | Detect | Segment | **SemanticSegment** |
|---|---|---|---|
| 输出 | `(B,4*reg_max+nc,N)` | `(B,4*reg_max+nc+32,N)`+proto | **`(B,nc,H/4,W/4)`** |
| 是否逐实例 | 是（框） | 是（框+实例掩码） | **否（每像素）** |
| NMS | 有 | 有 | **无** |
| 损失 | box+cls+dfl | +mask | **逐像素 CE（+Dice）** |
| 指标 | mAP | box+mask mAP | **mIoU / acc** |

---

## 3. 输出与后处理（★ 重点）

### 3.1 输出与解码

```
logits = model(x)                       # (B, nc, H/4, W/4)，无 sigmoid
# 后处理 SemPostProcess：先双线性插值到标签尺寸，再 argmax
logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
pred   = logits.argmax(1)               # (B, H, W) 每像素类别 id
```

### 3.2 损失（`torchkiln/sem.py::SemLoss`）

```
# 主损失：逐像素交叉熵（可忽略 ignore_index=255）
loss = CrossEntropyLoss(logits, mask, ignore_index=255, label_smoothing=0)
# 可选 Dice（dice_weight>0）：softmax → one-hot → (2*inter+1)/(denom+1)
loss = loss + dice_weight * (1 - dice)
# 若 logits 尺寸与 mask 不同，先双线性插值对齐
```

### 3.3 指标（`SemMetric`）

| 指标 | 计算 |
|---|---|
| `mIoU` | 混淆矩阵 → 每类 `inter/union`，只对 `union>0` 的类平均 |
| `acc` | 全局像素正确率（`correct/total`） |
| `ignore_index` | 默认 255（不计入混淆矩阵） |

---

## 4. 配置与用法

### 4.1 最小配置（`configs/yolo/yolo26-sem.yml`）

```yaml
Architecture:
  task: semantic
  yaml_file: torchkiln/cfg/models/26/yolo26-sem.yaml
  scale: n
  Head: {num_classes: 3, reg_max: 1, end2end: true}
Loss:  {name: SemLoss, ignore_index: 255, dice_weight: 0.0}
Metric: {name: SemMetric, main_indicator: mIoU, ignore_index: 255}
PostProcess: {name: SemPostProcess}
Train:
  dataset: {name: SemDataset, data_dir: datasets/sem_demo, label_file_list: [.../train.txt],
            transform: {image_size: 256}}
```

yolo26-sem 的 yaml 末尾：
```yaml
head:
  ...
  - [[16, 13], 1, SemanticSegment, [nc]]   # 用 P3(idx16) 与 P4(idx13)
```
> yolo26-sem yaml 只用了 **两个尺度**（P3/P4）进 `SemanticSegment`；
> `yolo11-sem.yaml` 用三个尺度 `[[16,19,22], SemanticSegment, [nc]]`。

### 4.2 四条链路

```bash
tkiln train   -c configs/yolo/yolo26-sem.yml -o Global.pretrained_model=yolo26n-sem
tkiln val     -c configs/yolo/yolo26-sem.yml --weights output/yolo26-sem/best_accuracy.pth
tkiln predict -c configs/yolo/yolo26-sem.yml --weights ... --input imgs/     # 输出类别图
tkiln export  -c configs/yolo/yolo26-sem.yml --weights ... --onnx
tkiln check   -c configs/yolo/yolo26-sem.yml
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ |
|---|---|
| yolo26n-sem | 2.57 |
| yolo26s-sem | 10.01 |
| yolo26m-sem | 21.90 |
| yolo26l-sem | 26.30 |
| yolo26x-sem | 58.99 |

¹ 取自 `torchkiln/cfg/models/26/yolo26-sem.yaml` 注释。
**FLOPs / 本机耗时 / 显存 未在本框架侧统计（不编造）。**

---

## 6. 公开指标

| 项 | 状态 |
|---|---|
| **权重加载对齐** | **未验证**（AGENTS 未记录 yolo26-sem 的 missing/unexpected；官方 `yolo26*-sem.pt` 存在） |
| **前向逐层对齐** | **未验证** |
| **端到端 mIoU** | **未统计** |
| 结构 | backbone/neck 与 detect 相同（已对齐）；头为**自实现多尺度融合**，**未与 ultra 逐键对齐** |

> ⚠️ **如实说明**：本任务在本框架中**已实现并可训练/评估/推理**（`smoke_all` 中含 `yolo26-sem` 配置），
> 但**尚未做与 ultralytics 的权重加载、逐层前向、单步 loss/梯度、端到端 mIoU 的对齐验证**。
> 因此本页不提供"官方 vs 框架"的数值对比表；需要时按统一对齐流程（见 `docs/models/_PLAN.md` 的 infra 篇）补做。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 可行驶区域 / 车道 / 天空分割 | **yolo26n-sem** | 逐像素、无 NMS，密集场景快 |
| 需要区分同一类别的不同实例 | **segment**（实例分割） | 语义分割做不到 |
| 车道线（行锚点） | 见 `configs/yolo/yolo11-lane-row.yml`（`LaneRow`） | UFLD 行分类，比像素分割更适合线 |
| 车道线（分割式） | 见 `configs/yolo/yolo11-lane-seg.yml` | |
| 3D/点云语义 | 见 `docs/models/pc/squeezesegv3.md` | range-view |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs segment** | semantic 每像素一类、无实例、无 NMS；segment 出实例掩码 |
| **vs FCN/U-Net** | 本头用 YOLO backbone + 多尺度相加融合，不做逐级解码卷积 |
| **vs DeepLab（ASPP）** | 本头无空洞金字塔，融合更轻 |
| **yolo11-sem vs yolo26-sem** | y11 用 3 尺度进头；y26 用 2 尺度 + `SemanticSegment`；y26 的 seg 头另有语义辅助分支 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **对齐状态未验证**：本头为自实现，**未做权重加载/数值对齐**（见 §6）。若需与 ultra 严格复现，
  应优先核对 `SemanticSegment` 的通道/尺度选择与官方 `-sem` 头的定义。
- ⚠️ **yaml 进头尺度不同**：`yolo26-sem` 只用 `[16,13]`（P3/P4）；`yolo11-sem` 用 `[16,19,22]`（P3/P4/P5）。
  改 yaml 时注意 `lat*` 数量需与输入尺度数匹配。
- ⚠️ **两处实现**：`models/sem.py::SemModel`（手写，多一次 ×2 上采样）与 `nn/modules.py::SemanticSegment`
  （图模型，输出 stride 8）。二者路径不同，勿混用。
- ⚠️ **`ignore_index` 必须与标签一致**（默认 255）；越界类 id 会被混淆矩阵忽略。
- ⚠️ **评估 fp32**：`_evaluate_loop` 强制 `autocast(enabled=False)`（fp16 会失真）。
- ⚠️ `forward_train` 直接返回模型输出，loss 内自行插值对齐尺寸。
