# YOLO12

> **定位**：2025-02 的 **"以注意力为中心"（attention-centric）** 检测器：在 v11 的 `C3k2` 骨架里把深层的
> `C3k2` 换成 **`A2C2f`（Area-Attention C2f）**，用 **区域注意力（area attention）** 换取全局建模能力。
> **任务**：`detect`（另有 `yolo12-seg/pose/obb/cls` 的 YAML，权重只发布了检测）
> **权重**：`yolo12{n,s,m,l,x}.pt`（本机 `\\tsclient\D\项目资料\ultralytics_models\yolo12\`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **YOLO12: Attention-Centric Object Detection**，Yunjie Tian、Qixiang Ye、David Doermann，**arXiv:2502.12524**（2025-02） |
| 官方代码 | `https://github.com/sunsmarterjie/yolov12`<br>`https://github.com/ultralytics/ultralytics`（cfg `12/yolo12.yaml`） |
| 官方文档 | `https://docs.ultralytics.com/models/yolo12` |
| 本框架实现 | `torchkiln/cfg/models/12/yolo12.yaml`<br>`torchkiln/nn/modules.py`（**`A2C2f` / `ABlock` / `AAttn`**，以及复用的 `C3k2`/`C3k`/`Bottleneck`/`Detect26`）<br>`torchkiln/nn/graph.py::parse_model`（l/x 规模追加 `(True, 1.2)`） |
| 移植方式 | **结构逐层复刻 + 权重逐键对齐**：n/s/m/l/x 全 **missing=0**（unexpected 仅"旧版权重的 `attn.pe.conv.bias`"，见 §9） |

### 论文要解决的问题 / 核心创新

注意力模块通常**代价高**（全局 self-attention 的复杂度随分辨率平方增长），YOLO12 的做法是：

1. **区域注意力（Area Attention）**：把特征图在**空间上切成 `area` 块**，
   **块内**做 attention、块间不交互 ⇒ 复杂度按 `1/area` 下降，仍保留一定的全局感受野；
2. **`A2C2f`**：把 `C2f` 风格的 CSP 结构（`cv1` 压缩 → 逐级拼接 → `cv2` 融合）与
   **`ABlock`（区域注意力 + FFN）** 结合；
3. **R-ELAN 式残差**：`A2C2f(..., residual=True)` 时引入**可学习缩放 `gamma` 的残差**
   （`x + gamma * y`，`gamma` 初始化为 0.01），稳定深层训练；
4. 结果：官方 AP 40.6~55.2（n~x），代价是**速度比 YOLO11 同档慢**（T4 上 n 档 1.64 ms ONNX CPU / 2.6 ms TRT）。

> ⚠️ **ultralytics 官方立场**：其对比文档写明 *"We currently do not recommend YOLO12 or YOLO13 for production use."*
> 该模型在 ultralytics 生态里被定位为"社区模型/基准"，本框架复刻它主要用于**对齐与结构研究**。

### 官方指标（ultralytics 文档 Detection(COCO) 表）

| 型号 | mAP50-95 | 参数(M) | FLOPs(B) | 对比（官方注释） |
|---|---|---|---|---|
| YOLO12n | **40.6** | 2.6 | 6.5 | +2.1% mAP / -9% 速度（vs YOLOv10n） |
| YOLO12s | **48.0** | 9.3 | 21.4 | +0.1% / +42%（vs RT-DETRv2） |
| YOLO12m | **52.5** | 20.2 | 67.5 | +1.0% / -3%（vs YOLO11m） |
| YOLO12l | **53.7** | 26.4 | 88.9 | +0.4% / -8%（vs YOLO11l） |
| YOLO12x | **55.2** | 59.1 | 199.0 | +0.6% / -4%（vs YOLO11x） |

---

## 2. 网络结构

### 2.1 整体框图（`12/yolo12.yaml`，n 档）

```
Input (B,3,640,640)
  │  [0] Conv(3→64,   k3, s2)                  → P1 320×320
  ├─ [1] Conv(64→128, k3, s2)                  → P2 160×160
  ├─ [2] C3k2(128→256, n=2, c3k=False, e=0.25)  ← v11 风格 CSP
  ├─ [3] Conv(256→256,k3, s2)                  → P3  80×80
  ├─ [4] C3k2(256→512, n=2, c3k=False, e=0.25)
  ├─ [5] Conv(512→512,k3, s2)                  → P4  40×40
  ├─ [6] A2C2f(512→512, n=4, a2=True, area=4)  ← ★ 区域注意力（area=4）
  ├─ [7] Conv(512→1024,k3,s2)                  → P5 20×20
  ├─ [8] A2C2f(1024→1024, n=4, a2=True, area=1) ← ★ area=1（等价全局    + 1/2 通道）
       │
       ▼ Neck（PAN：P3/P4 用 A2C2f，P5 用 C3k2）
  [9]  Upsample×2  [10] Concat([9],[6])   [11] A2C2f(1024→512, n=2, a2=False, area=-1)
  [12] Upsample×2  [13] Concat([12],[4])  [14] A2C2f(512→256,  n=2, a2=False, area=-1) → T2 (P3)
  [15] Conv(256→256,k3,s2) [16] Concat([15],[11]) [17] A2C2f(512→512, n=2, a2=False)  → T3 (P4)
  [18] Conv(512→512,k3,s2) [19] Concat([18],[8])  [20] C3k2(1024→1024, n=2, True)     → T4 (P5)
       ▼
  [21] Detect([14,17,20], nl=3, strides=[8,16,32])   ← 新 DWConv 头 + DFL(reg_max=16)
```

> `area=-1` 是 YAML 里表示"**不做区域切分**（等价 area=1）"的写法（`ABlock`/`AAttn` 内部按 `area>1` 判分支）。
> `a2=False` 表示该 `A2C2f` 的 `m` 用 `C3k` 而不是 `ABlock`（即"外形是 A2C2f、内层还是卷积块"）。

### 2.2 逐模块说明

| 模块 | 框架实现 | 结构 | 作用 |
|---|---|---|---|
| **`AAttn`** | `modules.py::AAttn` | `qkv=Conv(dim, 3*all_head_dim, 1, act=False)` → 若 `area>1` 先 reshape 成 `(B*area, N/area, 3H)` → `softmax(q·k/√d)·v` → `x + pe(v)` → `proj=Conv(...,1,act=False)`；`pe=Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)`（**深度卷积位置编码，无 bias**） | ★ 区域注意力 |
| **`ABlock`** | `modules.py::ABlock` | `x = x + attn(x)`；`x = x + mlp(x)`；`mlp = Seq(Conv(dim, int(dim*mlp_ratio), 1), Conv(..., 1, act=False))`；`_init_weights` 对 `Conv2d` 用 **`trunc_normal_(std=0.02)`** | ★ 注意力块 |
| **`A2C2f`** | `modules.py::A2C2f` | `cv1=Conv(c1, c_, 1)`（**不切半**）→ `m = n×Seq(ABlock, ABlock)`（`a2=True`）或 `n×C3k`（`a2=False`）→ `cv2=Conv((1+n)*c_, c2, 1)`；`a2 and residual` 时 `y = x + gamma.view(-1,c,1,1)*y`，`gamma=0.01*ones(c2)` | ★ 区域注意力 C2f |
| `C3k2` / `C3k` / `Bottleneck` | `modules.py` | 与 v11 相同（`attn=False` 分支） | 骨架浅层/P5 |
| `SPPF` / `Concat` / `nn.Upsample` | 同其它家族 | — | 池化/PAN |
| **`Detect26`（新 DWConv 头）** | `modules.py::Detect26` | `cv2=[Conv,Conv,Conv2d]`（reg）+ `cv3=[Seq(DWConv,Conv), Seq(DWConv,Conv), Conv2d]`（cls） | ★ v11/v12 的新头 |

> **`A2C2f` 的两处关键实现细节**：
> ① `c_ = int(c2*e)` 必须 **`c_ % 32 == 0`**（`ABlock` 的 `num_heads = c_ // 32`，断言会直接报错）；
> ② `__init__` 签名是 `(c1, c2, n, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True)`，
> 与 YAML 的 `[c2, a2, area]` 三参数对应。

### 2.3 与上一代（YOLO11）的差异

| 对比对象 | YOLO11 | **YOLO12** | 为什么改 | 效果 |
|---|---|---|---|---|
| 深层块 | `C3k2` | **`A2C2f`**（区域注意力） | 引入空间注意力 | 官方 mAP(n) **39.5 → 40.6** |
| 注意力 | `C2PSA`（在 SPPF 后，单点） | **`A2C2f` 遍布 P4/P5 与颈部** | 更早更深地做注意力 | 建模能力↑ |
| 残差 | — | `residual` + **`gamma`（0.01）** | 稳定深层注意力 | 训练更稳 |
| 头 | 新 DWConv 头 + DFL(16) | **同**（`Detect26` + DFL16） | — | 复用损失/NMS |
| 参数(n) | 2.6M | 2.6M | 同量级 | 官方 mAP +1.1 |
| 速度 | 更快 | **更慢**（官方"速度下降"注释） | 注意力开销 | 见 §5 |
| 生产定位 | 官方推荐 | **官方不推荐生产使用** | — | 见 §1 |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 头类型 | **`Detect26`（新 DWConv 头）**；实测 `model.21`，`nl=3`，`no=144`（nc=80, reg_max=16） |
| 输出布局 | 每尺度 `[reg(4*reg_max), cls(nc)]` |
| Anchor | **无**（anchor-free，网格中心 +0.5） |
| 解码 | `dfl_project(reg_logits, 16)` → `dist2bbox` → `× stride` |
| 损失 | **单 assigner** 的 `DetLoss`（`v8DetectionLoss` 移植；TAL `topk=10 / alpha=0.5 / beta=6.0`），**非 end2end** |
| NMS | `torchvision.ops.nms`；评估 `conf 0.001 / iou 0.7`，推理 `conf 0.25 / iou 0.45`；`max_det 300 / max_nms 3000` |
| strides | `[8, 16, 32]` |

> 注：YAML 里的 `area`（4 / 1 / -1）只影响 **`AAttn` 内部的空间切分**，不影响输出布局与后处理。

---

## 4. 配置与用法

### 4.1 最小配置（`configs/yolo/yolo12-det.yml` 关键项）

```yaml
Architecture:
  model_family: yolo
  task: detect
  algorithm: yolo12
  yaml_file: torchkiln/cfg/models/12/yolo12.yaml
  scale: n                 # n/s/m/l/x（yaml 里有 scales 字典，五档齐全）
  in_channels: 3
  Head:
    num_classes: 80
    reg_max: 16            # ⚠️ 加载官方 yolo12 权重必须 16
Loss:   {name: DetLoss, topk: 10, alpha: 0.5, beta: 6.0, cls_gain: 0.5, box_gain: 7.5, dfl_gain: 1.5}
Metric: {name: DetMetric, main_indicator: mAP50-95}
PostProcess: {name: DetPostProcess, conf_thres: 0.001, iou_thres: 0.7, strides: [8, 16, 32]}
Train:
  loader: {batch_size_per_card: 8, num_workers: 4}   # 注意力更吃显存，先用 8
```

### 4.2 四条链路

```bash
# 结构自检
tkiln check -c configs/yolo/yolo12-det.yml

# 训练（微调官方 yolo12n）
tkiln train -c configs/yolo/yolo12-det.yml \
    -o Architecture.Head.reg_max=16 -o Global.pretrained_model=weights/yolo12n.pth

# 评估（官方口径）
tkiln val -c configs/yolo/yolo12-det.yml -o Architecture.Head.reg_max=16 \
    --weights output/yolo12-det/best_accuracy.pth

# 预测 / 导出
tkiln predict -c configs/yolo/yolo12-det.yml --weights ... --input imgs/ --output out.jpg
tkiln export  -c configs/yolo/yolo12-det.yml -o Architecture.Head.reg_max=16 \
    --weights ... --save-dir output/onnx --onnx

# 其它任务的 YAML（权重需自行训练）
tkiln check -c configs/yolo/yolo12-seg.yml
tkiln check -c configs/yolo/yolo12-pose.yml
tkiln check -c configs/yolo/yolo12-obb.yml
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ | 官方参数(M)² | 官方 FLOPs(B)² | 官方 mAP50-95² | 官方 .pt³ | head idx / strides |
|---|---|---|---|---|---|---|
| n | **2.60** | 2.6 | 6.5 | 40.6 | 5.3 MB | 21 / 8-16-32 |
| s | **9.28** | 9.3 | 21.4 | 48.0 | — | 21 / 8-16-32 |
| m | **20.20** | 20.2 | 67.5 | 52.5 | — | 21 / 8-16-32 |
| l | **26.45** | 26.4 | 88.9 | 53.7 | — | 21 / 8-16-32 |
| x | **59.21** | 59.1 | 199.0 | 55.2 | 113.8 MB | 21 / 8-16-32 |

¹ 本框架实测（`nc=80`、`reg_max=16`）；与 `yolo12.yaml` 注释里的官方参数**只差 16**（DFL 是函数式）。
² ultralytics 官方文档表。³ 本机官方权重磁盘体积（fp16）。

> ⚠️ **本框架未集成 FLOPs 计数**；表中 FLOPs 为官方值。
> **速度注意**：官方对比注释显示 YOLO12 相对 v11/v10 **更慢**（注意力开销）。本机未做统一速度基准（未统计）。

---

## 6. 公开指标

| 数据集/口径 | 指标 | ultralytics 官方 | **本框架实测** | 差异原因 |
|---|---|---|---|---|
| COCO val2017 | mAP50-95 `n/s/m/l/x` | 40.6 / 48.0 / 52.5 / 53.7 / 55.2 | 权重加载 **missing=0**；**unexpected = 旧版权重的 `attn.pe.conv.bias`**（n 9 个 / x 17 个） | 见 §9（AAttn.pe 的 bias 历史） |
| nc=1 重建权重（8.4.154 源码口径） | 加载 | — | **missing=0 / unexpected=1** | 仅函数式 `model.21.dfl.conv.weight` |
| `dx_ocr` 车牌（nc=1，6ep） | mAP50-95 | 0.800 | **0.794** | 差 0.006，数据顺序/随机性 |

**对齐验证（单步，同权重同输入同 GT，yolo12n，`reg_max=16`，单 assigner）**：

| 项 | 本框架 | ultralytics | 差异 |
|---|---|---|---|
| 权重加载（nc1 重建） | missing=0 / unexpected=1 | — | 仅函数式 dfl |
| box（raw） | **0.5237** | 同 | 一致 |
| cls（raw） | **11.6909** | 同 | 一致 |
| dfl（raw） | **2.5324** | 同 | 一致 |
| total | **13.5719** | **13.5719** | ~0 |
| 梯度 maxdiff（n/s/m/l/x） | — | — | **0.006 ~ 0.026**（算子级，worst 多在 `model.0.conv.weight`） |

> 复测记录：**yolo12n 13.571861 ↔ 13.571886**，梯度 **0.013**（在移除 `REPEAT_MODULES` 里的 `Bottleneck` 之后复测，无回归）。

**端到端训练（`dx_ocr`，nc=1，6 epoch，SGD/batch=8，关增广，EMA 0.9999）**：

| epoch | 1 | 2 | 3 | 4 | 5 | 6（终值） |
|---|---|---|---|---|---|---|
| 本框架 mAP50-95 | 0.478 | 0.698 | 0.744 | 0.751 | 0.743 | **0.794** |
| ultralytics | 0.369 | 0.538 | 0.755 | 0.754 | 0.776 | **0.800** |

---

## 7. 选型建议

| 场景 | 推荐档位 | 理由 |
|---|---|---|
| 精度优先（离线） | **x / l** | 55.2 / 53.7，参数与 v11x/v11l 相当 |
| 通用检测 | **s / m** | 48.0 / 52.5 |
| **边缘 / 低功耗** | **不推荐** | 区域注意力对 CPU/端侧不友好，且官方不推荐生产 |
| 需要稳定交付 | 用 **v11 / v26** | 官方推荐；本仓库对齐记录更完整（含端到端评估路径） |
| 研究"注意力 vs 卷积" | **n**（2.6M，与 y11n 同参） | 同参数下 n 档官方 +1.1 mAP，是最干净的对照 |
| 需要 seg/pose/obb/cls | 有 YAML，**无官方权重** | 官方只发布检测权重（文档明确） |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs YOLO11n** | 同参 2.6M：**40.6 vs 39.5**（+1.1）；v12 用 `A2C2f` 换掉了部分 `C3k2`，代价是速度 |
| **vs YOLOv10n** | v10n 38.5 / 6.7 GFLOPs（**NMS-free**）；v12n 40.6 / 6.5 GFLOPs（**仍需 NMS**）。官方注释：+2.1% mAP、-9% 速度 |
| **vs YOLO26n** | 40.9（e2e 40.1）/ 2.57M：v26 用**双头 + 去 DFL**取得更高 AP 且 NMS-free；v12 的注意力路线在部署上更重 |
| **vs RT-DETR** | 都是注意力/Transformer 路线；v12 保持 CNN 骨架 + 局部注意力，速度更快（官方注释：v12s +0.1% mAP / +42% 速度） |
| **定位** | v12 在本仓库的作用：验证 **`AAttn` 的区域切分语义、`ABlock/AAttn` 的权重命名与 `trunc_normal` 初始化、
以及 `A2C2f` 在 l/x 档的 `residual + mlp_ratio=1.2`** 三条链路 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`A2C2f` 的约束：`c_ % 32 == 0`**：`ABlock` 的 `num_heads = c_ // 32`，`A2C2f.__init__` 里有
  `assert c_ % 32 == 0`。改 YAML 的通道/`e` 时务必保证（否则直接 AssertionError）。
- ⚠️ **`A2C2f` 的 l/x 差异在 `parse_model` 里补**：ultralytics 对 `scale ∈ {l, x}` 会给 `A2C2f` 追加
  **`residual=True, mlp_ratio=1.2`**（即 `args.extend((True, 1.2))`）。本框架已复刻；
  否则 l/x 的 `gamma` 参数与 mlp 通道对不上，权重加载会缺键。
- ⚠️ **`AAttn.pe` 的 bias 坑（本框架已改回 `bias=False`，但**官方旧权重会多出 bias 键**）**：
  - ultralytics 8.4.154 源码：`self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)`
    ——它的 `Conv` 默认 **`bias=False`**，所以**新版权重没有 pe bias**；
  - 但**已发布的官方 `yolo12*.pt`**（由更早的代码产出）**带 pe bias**；
  - 本框架曾为了对齐旧版权重把 `AAttn.pe` 设成 `bias=True`，结果在"用 8.4.154 重建 nc=1 权重"时
    出现 **missing=8**（`attn.pe.conv.bias`）；改成 `bias=False` 后 **nc1 权重 missing=0 / unexpected=1**，
    并且 **yolo26 官方权重仍 missing=0 / unexpected=0**（没被破坏）；
  - 现在的行为：**加载官方 `yolo12n.pt` 会有 unexpected 键（n 9 个、x 17 个 `…attn.pe.conv.bias`），
    但 missing=0，前向/损失正常**（多出的 bias 不参与计算）。
  > **口径以当前 ultralytics 源码（8.4.154）为准**。
- ⚠️ **加载官方权重必须 `Head.reg_max: 16`**（`configs/yolo/yolo12-det.yml` 默认写 1）。
- ⚠️ **`A2C2f` 的 `area=-1` 在 YAML 中的含义**：表示"不做区域切分"（等效 `area=1`），
  实现上 `AAttn.forward` 只在 `area > 1` 时做 reshape。手工改 YAML 时不要把它当"负数面积"。
- ⚠️ **注意力家族更吃显存**：同 batch 下显存高于 v8/v11；`imgsz=1024` 建议 **batch=4 甚至 2**。
- ⚠️ **梯度差异 0.006~0.026 属算子级**（worst `model.0.conv.weight`），不是逻辑 bug。
- ⚠️ 通用修复（DFL target 不 clamp、评估 fp32、`torchvision` NMS、`accumulate` 读取位置、
  `REPEAT_MODULES` 不含 `Bottleneck`/`PSA` 等）见 `yolo11.md` §9。
- ⚠️ 本框架**未集成 FLOPs 统计**，§5 的 FLOPs 为官方值；本机也**未做统一速度基准**。
