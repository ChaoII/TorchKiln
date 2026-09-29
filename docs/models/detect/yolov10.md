# YOLOv10

> **定位**：清华团队 2024-05 提出的 **NMS-free（端到端）实时检测器**：训练时用
> **一致性双分配（consistent dual assignments）**——one2many 头 + one2one 头，推理只用 one2one，**不需要 NMS**。
> **任务**：`detect`
> **权重**：`yolov10{n,s,m,b,l,x}.pt`（本机 `\\tsclient\D\项目资料\ultralytics_models\yolov10\`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **YOLOv10: Real-Time End-to-End Object Detection**，Ao Wang、Hui Chen、Lihao Liu、Kai Chen、Zijia Lin、Jungong Han、Guiguang Ding，**arXiv:2405.14458**（NeurIPS 2024） |
| 机构 | 清华大学（THU-MIG）|
| 官方代码 | `https://github.com/THU-MIG/yolov10`<br>`https://github.com/ultralytics/ultralytics`（cfg `v10/yolov10*.yaml`） |
| 官方文档 | `https://docs.ultralytics.com/models/yolov10` |
| 本框架实现 | `torchkiln/cfg/models/v10/yolov10{n,s,m,l,x,b}.yaml`<br>`torchkiln/nn/modules.py`（`SCDown`/`PSA`/`CIB`/`RepVGGDW`/`C2fCIB`/`Detect10`）<br>`torchkiln/nn/graph.py`（**按规模替换 `C2f`→`C2fCIB`** 的 `v10_sw` 表） |
| 移植方式 | **结构逐层复刻 + 权重逐键对齐**：n/s/m/l/x 全 **missing=0 / unexpected=1**（仅函数式 `model.23.dfl.conv.weight`） |

### 论文要解决的问题 / 核心创新

NMS 是后处理瓶颈，也是端到端部署的障碍。YOLOv10 的两条主线：

1. **一致性双分配（NMS-free 训练）**：训练时**同时**接 one-to-many 头（丰富监督）与 one-to-one 头（每目标 1 个预测），
   两者共用匹配度量 ⇒ **推理只跑 one2one，省掉 NMS**。
   > ultralytics 的 `predict`/`val` **默认仍走 one2many + NMS**（要 NMS-free 需 `nms=False`）；
   > 本框架对应开关是 **`PostProcess.end2end`**（true=one2one 无 NMS，false=one2many+NMS）。
2. **面向效率-精度的整体设计**：轻量分类头（深度可分离）、**SCDown**（空间-通道解耦下采样）、
   **等级引导的块设计**（深层用 `C2fCIB` 替换 `C2f`）、**大核卷积（CIB 的 `lk`）+ 部分自注意力（PSA）**。

### 官方指标（ultralytics 文档，latency 为 T4 TensorRT FP16）

| 型号 | AP(val) | FLOPs(G) | Latency(ms) | 论文表参数(M) |
|---|---|---|---|---|
| YOLOv10n | **38.5** | 6.7 | 1.84 | 2.3 |
| YOLOv10s | **46.3** | 21.6 | 2.49 | 7.2 |
| YOLOv10m | **51.1** | 59.1 | 4.74 | 15.4 |
| YOLOv10b | **52.5** | 92.0 | 5.74 | 未列出 |
| YOLOv10l | **53.2** | 120.3 | 7.28 | 24.4 |
| YOLOv10x | **54.4** | 160.4 | 10.70 | 29.5 |

---

## 2. 网络结构

### 2.1 整体框图（`v10/yolov10n.yaml`，n 档）

```
Input (B,3,640,640)
  │  [0] Conv(3→64,   k3, s2)          → P1 320×320
  ├─ [1] Conv(64→128, k3, s2)          → P2 160×160
  ├─ [2] C2f(128→128, n=3, True)
  ├─ [3] Conv(128→256,k3, s2)          → P3  80×80
  ├─ [4] C2f(256→256, n=6, True)
  ├─ [5] SCDown(256→512, k3, s2)       ← ★ 空间-通道解耦下采样 → P4 40×40
  ├─ [6] C2f(512→512, n=6, True)
  ├─ [7] SCDown(512→1024,k3, s2)       → P5 20×20
  ├─ [8] C2f(1024→1024, n=3, True)     ← 更大规模时这里换成 C2fCIB（见 2.3）
  ├─ [9] SPPF(1024, k5)
  └─[10] PSA(1024)                     ← ★ 部分自注意力（Partial Self-Attention）
       │
       ▼ Neck（PAN；下采样同样用 SCDown）
  [11] Upsample×2  [12] Concat([11],[6])   [13] C2f(1024→512, n=3)
  [14] Upsample×2  [15] Concat([14],[4])   [16] C2f(512→256,  n=3)  → T2 (P3)
  [17] Conv(256→256,k3,s2) [18] Concat([17],[13]) [19] C2f(512→512,n=3) → T3 (P4)
  [20] SCDown(512→512,k3,s2)[21] Concat([20],[10])
  [22] C2fCIB(512→1024, n=3, shortcut=True, lk=True) ← ★ 大核 CIB 块（v10n 固定）       → T4 (P5)
       ▼
  [23] v10Detect([16,19,22], nl=3, strides=[8,16,32])
       ├─ one2many: cv2(reg 4*reg_max) + cv3(cls, DWConv 塔)
       └─ one2one : one2one_cv2 / one2one_cv3（推理用）
```

### 2.2 逐模块说明

| 模块 | 框架实现 | 结构 | 作用 |
|---|---|---|---|
| **`SCDown`** | `modules.py::SCDown` | `cv1=Conv(c1,c2,1)` → `cv2=Conv(c2,c2,k=3,s=2,g=c2,act=False)`（深度卷积下采样） | ★ 空间-通道解耦下采样 |
| `C2f` | `modules.py::C2f` | 见 `yolov8.md` §2.2 | 常规 CSP |
| **`C2fCIB`** | `modules.py::C2fCIB` | 继承 `C2f`，把内层换成 **`CIB`**（`cv1` = DWConv→Conv→(**`RepVGGDW`** 或 DWConv)→Conv→DWConv） | ★ 深层高效块 |
| **`CIB`** | `modules.py::CIB` | `Conv(k3,g=c1)` → `Conv(1×1, 2c_)` → `**RepVGGDW(2c_)`（`lk=True`）或 `Conv(k3,g=2c_)`** → `Conv(1×1,c2)` → `Conv(k3,g=c2)`；`add = shortcut and c1==c2` | Compact Inverted Block |
| **`RepVGGDW`** | `modules.py::RepVGGDW` | `act(conv7x7_dw(x) + conv3x3_dw(x))`（`conv=Conv(ed,ed,7,1,3,g=ed,act=False)`、`conv1=Conv(ed,ed,3,1,1,g=ed,act=False)`） | ★ `CIB` 的大核分支（v10/v9 共用） |
| **`PSA`** | `modules.py::PSA` | `cv1` 切两半 → 后半 `+ attn`（`Attention`：qkv/proj/pe）→ `+ ffn` → `cv2` 融合；**`e=0.5` 是展开比，不是"重复次数"** | ★ 部分自注意力 |
| `SPPF` | `modules.py::SPPF` | 同 v8；**`args ≤ 3` 时恢复 `cv1` 的 SiLU** | 池化 |
| **`Detect10`（`v10Detect`）** | `modules.py::Detect10` | 继承 `Detect26`：`cv2/cv3`（one2many）+ `one2one_cv2/one2one_cv3`；训练返回 `(o2m, o2o)`，推理按 `self.end2end` 选分支 | ★ NMS-free 双头 |

### 2.3 `C2f → C2fCIB`：v10 **按规模替换**（`graph.py::v10_sw`）

ultralytics 的 v10 是**每档一个 YAML**（`yolov10n.yaml` … `yolov10x.yaml`），并且**随规模增大把深层 `C2f` 换成 `C2fCIB`**。
本框架只有一份"基础 YAML"，用 `build_from_arch` 里的替换表按 `scale` 改写：

| 档位 `scale` | backbone 中替换 | head 中替换 | head[11] 的 `args` |
|---|---|---|---|
| `n` | 无 | 无 | `[1024, True, True]`（lk=True） |
| `s` | idx 8 → `C2fCIB(1024, lk=True)` | 无 | `[1024, True, True]` |
| `m` | idx 8 → `C2fCIB(1024, lk=False)` | idx 8 → `C2fCIB(512)` | `[1024, True]` |
| `l` | idx 8 → `C2fCIB(1024, lk=False)` | idx **2** → `C2fCIB(512)`、idx 8 → `C2fCIB(512)` | `[1024, True]` |
| `x` | idx **6** → `C2fCIB(512, lk=False)`、idx 8 → `C2fCIB(1024, lk=False)` | idx 2、8 → `C2fCIB(512)` | `[1024, True]` |
| `b` | （`v10b.yaml` 已烘焙全部 `C2fCIB`） | — | `[1024, True]` |

> ⚠️ **替换只在"路径含 `/v10/` 且 `scale` 正确"时生效**。若配置里 `scale: n` 而 yaml 是 `yolov10s.yaml`，
> 既不会做替换，也会回落到 `DEFAULT_SCALES["n"]` → **静默构建出错误的（缩水的）模型**（实测：8.13M → 1.88M，见 §9）。

### 2.4 与上一代（YOLOv8）的差异

| 对比对象 | YOLOv8 | **YOLOv10** | 为什么改 | 效果 |
|---|---|---|---|---|
| 后处理 | NMS | **NMS-free**（one2one 头） | 去掉后处理瓶颈 | 延迟更稳、易部署 |
| 头 | 单头（one2many） | **双头**（one2many + one2one） | 训练监督 + 推理去重 | AP↑ 且免 NMS |
| 下采样 | strided `Conv` | **`SCDown`** | 解耦空间/通道 | 省算力 |
| 深层块 | `C2f` | **`C2fCIB`（含 `RepVGGDW` 大核）** | 大核提感受野 | 同算力 AP↑ |
| 全局建模 | 无 | **`PSA`（部分自注意力）** | 低开销全局信息 | AP↑ |
| 官方 AP(n) | 37.3 / 8.7 GFLOPs | **38.5 / 6.7 GFLOPs** | 整体设计 | **AP+1.2、算力-23%** |
| 回归 | DFL(16) | **DFL(16)（同）** | — | 复用 v8 损失 |

---

## 3. 输出与后处理

### 3.1 输出张量

| 阶段 | 返回 | 形状（每尺度） | 说明 |
|---|---|---|---|
| **训练** | `(one2many, one2one)` **元组** | 每个 `(B, 4*reg_max + nc, H, W)` = `(B, 144, H, W)` | 两个头都算损失 |
| **推理** | 单分支列表 | 同上 | `self.end2end=True` → one2one；`False` → one2many |

### 3.2 梯度回传（重要）

`Detect10.forward` 在 `self.training` 时对 **one2one 分支的输入特征做 `.detach()`**：
**one2one 只训练自己的头，不把梯度回传给 backbone**（对齐 ultralytics：`x_detach = [xi.detach() for xi in x]`）。
早先未 detach 时，backbone 梯度 = one2many + one2one 之和，**全模型梯度 maxdiff 达 40.4**；
修复后降到 **0.00034**（算子级）。

### 3.3 解码

```
cls = sigmoid(cls_logits)
l,t,r,b = dfl_project(reg_logits, 16)     # reg_max=16 的 DFL
boxes = dist2bbox(...) * stride
```

### 3.4 NMS 与阈值

| 项 | `PostProcess.end2end: false`（one2many） | `true`（one2one，NMS-free） |
|---|---|---|
| NMS | `torchvision.ops.nms` | **不做**（one2one 已去重，`_nms` 直接返回全部索引） |
| conf / iou | 0.001 / 0.7（评估），0.25 / 0.45（推理） | 同上（iou 不起作用） |
| max_det / max_nms | 300 / 3000 | 同 |
| strides | `[8,16,32]` | 同 |

> **评估建议用 `end2end: false`**（one2many + NMS）：与 ultralytics 的 `val` 默认口径一致。
> **训练必须 `Head.end2end: true`**（否则没有 one2one 分支，损失里 `use_one2one` 会落空）。

---

## 4. 配置与用法

### 4.1 关键配置（对齐官方权重的写法）

```yaml
Architecture:
  model_family: yolo
  task: detect
  algorithm: yolov10
  yaml_file: torchkiln/cfg/models/v10/yolov10n.yaml   # ★ 每档一个文件（n/s/m/l/x/b）
  scale: n                                            # ★ 必须与该 yaml 的 scales 键一致
  in_channels: 3
  Head:
    num_classes: 80
    reg_max: 16            # ★ 必须 16（DFL）；configs/yolo/*.yml 示例默认写 1
    end2end: true          # ★ 训练用双头
Loss:
  name: DetLoss
  strides: [8, 16, 32]
  topk: 10                 # one2many
  alpha: 0.5
  beta: 6.0
  cls_gain: 0.5
  box_gain: 7.5
  dfl_gain: 1.5
  use_one2one: true        # ★ one2one 分支
  one2one_topk: 1
  one2one_gain: 1.0
PostProcess:
  name: DetPostProcess
  conf_thres: 0.001
  iou_thres: 0.7
  max_det: 300
  strides: [8, 16, 32]
  end2end: false           # ★ 评估走 one2many + NMS（与 ultra val 口径一致）
```

### 4.2 四条链路

```bash
# 结构自检
tkiln check -c configs/yolo/yolov10n-det.yml

# 训练（微调官方 v10n；注意覆盖 scale/reg_max/end2end）
tkiln train -c configs/yolo/yolov10n-det.yml \
    -o Architecture.scale=n -o Architecture.Head.reg_max=16 -o Architecture.Head.end2end=true \
    -o Global.pretrained_model=weights/yolov10n.pth

# 评估（推荐 one2many+NMS 口径）
tkiln val -c configs/yolo/yolov10n-det.yml \
    -o Architecture.Head.reg_max=16 -o PostProcess.end2end=false \
    --weights output/yolov10n-det/best_accuracy.pth

# 预测 / 导出（想要 NMS-free 就把 PostProcess.end2end 设 true）
tkiln predict -c configs/yolo/yolov10n-det.yml -o PostProcess.end2end=true --weights ... --input imgs/
tkiln export  -c configs/yolo/yolov10n-det.yml -o Architecture.Head.reg_max=16 \
    --weights ... --save-dir output/onnx --onnx
```

---

## 5. 规模与速度

| 档位 | yaml | 参数(M)¹ | 论文表参数(M)² | 官方 AP² | 官方 FLOPs(G)² | 官方 .pt³ | head idx / strides |
|---|---|---|---|---|---|---|---|
| n | `v10/yolov10n.yaml` | **2.78** | 2.3 | 38.5 | 6.7 | 5.6 MB | 23 / 8-16-32 |
| s | `v10/yolov10s.yaml` | **8.13** | 7.2 | 46.3 | 21.6 | — | 23 / 8-16-32 |
| m | `v10/yolov10m.yaml` | **16.58** | 15.4 | 51.1 | 59.1 | — | 23 / 8-16-32 |
| b | `v10/yolov10b.yaml` | **20.57** | 未列出 | 52.5 | 92.0 | — | 23 / 8-16-32 |
| l | `v10/yolov10l.yaml` | **25.89** | 24.4 | 53.2 | 120.3 | — | 23 / 8-16-32 |
| x | `v10/yolov10x.yaml` | **31.81** | 29.5 | 54.4 | 160.4 | 61.4 MB | 23 / 8-16-32 |

¹ 本框架实测（`nc=80`、`reg_max=16`、`Head.end2end=true`，含 **one2one 头**的参数）。
² ultralytics / 论文口径：官方 AP 与 FLOPs 来自文档与论文表，**参数是 `model.fuse()` 且移除 one2many 辅助头之后**的数字，
所以比左边"含双头"的实测值小 0.3~2.3M——**口径不同，不是实现差异**。
³ 本机官方权重磁盘体积（fp16）。

> ⚠️ **本框架未集成 FLOPs 计数**；表中 FLOPs 为官方值。

---

## 6. 公开指标

| 数据集/口径 | 指标 | ultralytics 官方 | **本框架实测** | 差异原因 |
|---|---|---|---|---|
| COCO val2017 | AP `n/s/m/b/l/x` | 38.5 / 46.3 / 51.1 / 52.5 / 53.2 / 54.4 | 权重加载 **missing=0 / unexpected=1** | 仅函数式 `model.23.dfl.conv.weight` |
| `dx_ocr` 车牌（nc=1，6ep） | mAP50-95 | 0.811 | **0.785** | 差 0.026，属 E2E 训练波动 + 数据顺序 |

**对齐验证（单步，同权重同输入同 GT，`reg_max=16`，双头 `E2EDetectLoss` 口径）**：

| 档位 | 本框架 total | ultralytics total | 梯度 maxdiff |
|---|---|---|---|
| **n** | **23.91382** | **23.91383** | 0.0002~0.0013（worst `model.0.conv.weight`） |
| **s** | **26.213394** | **26.213394** | 同上量级 |
| **m** | **25.325804** | **25.325804** | 同上 |
| **l** | **21.74652** | **21.74652** | 同上 |
| **x** | **22.523449** | **22.523449** | 同上 |

（loss 走 `E2EDetectLoss`：one2many `tal_topk=10` + one2one `tal_topk=1`，与 yolo26 同一套代码路径。）

**端到端训练（`dx_ocr`，nc=1，6 epoch，SGD/batch=8，关增广，EMA 0.9999）**：

| epoch | 1 | 2 | 3 | 4 | 5 | 6（终值） |
|---|---|---|---|---|---|---|
| 本框架 mAP50-95 | 0.608 | 0.668 | 0.731 | 0.762 | 0.734 | **0.785** |
| ultralytics | 0.316 | 0.711 | 0.756 | 0.775 | 0.791 | **0.811** |

> 终值差 **0.026**，是本次对比中**最大**的一个。原因：E2E 双头训练本身更不稳（ultra 第 1 轮只有 0.316），
> 且两侧数据顺序不同（shuffle）。**单步 loss/梯度已逐位对齐（上表）**，故属训练随机性而非算法差异。

---

## 7. 选型建议

| 场景 | 推荐档位 | 理由 |
|---|---|---|
| 边缘部署、要稳定延迟 | **n**（2.78M，官方 38.5/6.7G） | NMS-free 时延迟最稳 |
| 通用检测 | **m / b** | b 是"balanced"档，52.5 AP 而 FLOPs 只有 l 的 76% |
| 高精度 | **l / x** | 53.2 / 54.4 |
| **端到端部署（无 NMS）** | 任意档 + `PostProcess.end2end=true` | 本框架支持 one2one 直出 |
| 需要与 ultra 对齐的评估数字 | `PostProcess.end2end=false` | 官方 val 默认 one2many + NMS |
| 需要 seg/pose/obb/cls | **不支持**（v10 只有检测） | 用 v8/v11/v26 的对应变体 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs YOLOv8** | v10n **38.5 vs 37.3**，算力 **6.7 vs 8.7 GFLOPs**；v10 多一个 one2one 头与 `SCDown/PSA/C2fCIB` |
| **vs YOLOv9c** | v10b "46% less latency、25% fewer params than YOLOv9-C at same accuracy"（官方对比）|
| **vs YOLO11** | y11n 39.5 / 2.6M（**无 NMS-free**）；v10 的价值在 **端到端** 与部署稳定性 |
| **vs YOLO26** | v26 把"双头 NMS-free"进一步做成"**去掉 DFL**（`reg_max=1` + L1）+ 更轻的头"，
并配 MuSGD/专属 loss；v10 仍用 DFL(16)。两者在本框架共用 **`Detect10` + `E2EDetectLoss`** 代码路径 |
| **定位** | v10 是本仓库验证"**双头 + detach + 双分支 loss**"的第二个家族（第一个是 yolo26），也是唯一带 **PSA** 的检测家族 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`PSA` 不能放进 `REPEAT_MODULES`（本框架已修）**：
  `PSA` 是**单个块**（内部自己做 split+attn+ffn），不是"n 个块的列表"。早期误列入后，
  `graph.py` 会把重复数插到 args 第 3 位当成 **`e`（展开比）**，得到 `PSA(c1, c2, e=1)`、`c=256`（应为 128），
  导致结构/参数全错。移除后 `PSA(c1, c2)` 正常（`e=0.5`）。
- ⚠️ **`CIB` 的大核分支必须用 `RepVGGDW`（本框架已修）**：
  `CIB` 的 `lk=True` 分支在 ultra 里是 `RepVGGDW`（`7×7` 深度卷积 + `3×3` 深度卷积，**求和后 SiLU**），
  框架早期用的是 `RepConvN`，结构与参数均不符。`RepVGGDW` 只被 v10/v9 使用，不影响 yolo26。
- ⚠️ **`SPPF` 的 `cv1` 激活要按 YAML 形态恢复（本框架已修）**：
  ultra 的 YOLO26 风格 SPPF 的 `cv1` 是 **无激活**；但 `args ≤ 3` 的旧式 SPPF（v8/v10/v11/v12）要 **恢复 SiLU**。
  框架原先恒为无激活，导致 v10 从 L9(SPPF) 起 maxdiff 5.26。修法：`parse_model` 里
  `if module_name == "SPPF" and len(args) <= 3: layer.cv1.act = Conv.default_act`。
- ⚠️ **`C2f→C2fCIB` 的替换依赖 `scale` 正确**：见 §2.3 与「§9 下一条」。
- ⚠️ **每档 YAML + 对应 `scale`（最容易静默出错的地方）**：
  `v10/yolov10s.yaml` 的 `scales` 键只有 `s`。如果配置写 `scale: n`（本仓库 `configs/yolo/yolov10{s,m,l,x,b}-det.yml` 默认就是 `n`），
  `graph.py` 会 `yaml_scales.get("n") → None` → 回落到 `DEFAULT_SCALES["n"]=(0.33,0.25,1024)`，
  **同时 `v10_sw` 替换也不会发生**。实测：`yolov10s-det.yml` 构建出 **1,877,236** 参数（正确应为 **8,128,256**）。
  **用 `-o Architecture.scale=s` 覆盖即可。**
- ⚠️ **加载官方权重必须 `Head.reg_max: 16` + `Head.end2end: true`**：
  `configs/yolo/yolov10*-det.yml` 默认 `reg_max: 1`（无 DFL，对不上权重）。
- ⚠️ **one2one 输入必须 `.detach()`**（本框架已修，见 §3.2）：否则 backbone 梯度翻倍，全模型梯度 maxdiff 40.4。
- ⚠️ **评估路径**：`Detect10.forward` 在 eval 时按 `self.end2end` 选分支；`evaluate()` 会临时把它设为
  `PostProcess.end2end`，从而"评估走 one2many+NMS、权重里的 one2one 不参与"，与 ultralytics 口径一致。
- ⚠️ **E2E 训练波动大**：`dx_ocr` 上终值差 0.026（本仓库 detect 家族最大），首轮尤其不稳（ultra 0.316）；
  建议多 seed 或延长训练后再比。
- ⚠️ 通用修复（DFL target 不 clamp、评估 fp32、`torchvision` NMS、`accumulate` 读取位置、MuSGD 等）见 `yolo11.md` §9。
- ⚠️ 本框架**未集成 FLOPs 统计**，§5 的 FLOPs 为官方值。
