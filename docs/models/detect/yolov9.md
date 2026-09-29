# YOLOv9（t / s / m / c / e）

> **定位**：王建尧团队 2024-02 发布，用 **PGI（可编程梯度信息）+ GELAN（广义高效层聚合网络）** 缓解深层网络的信息丢失。
> **任务**：`detect`（另有 `yolov9c-seg` / `yolov9e-seg`）
> **权重**：`yolov9{t,s,m,c,e}.pt`（本机 `\\tsclient\D\项目资料\ultralytics_models\yolov9\`）
> **注意**：v9 各档的**通道不可用同一套宽度缩放表达**，本框架为每档备了独立 YAML（见 §2.3 / §9）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information**，Wang Chien-Yao、Yeh I-Hau、Liao Hong-Yuan Mark，**arXiv:2402.13616**（ECCV 2024） |
| 机构 | 中央研究院（Academia Sinica）|
| 官方代码 | `https://github.com/WongKinYiu/yolov9`（原始实现）<br>`https://github.com/ultralytics/ultralytics`（cfg `v9/*.yaml`） |
| 官方文档 | `https://docs.ultralytics.com/models/yolov9` |
| 本框架实现 | `torchkiln/cfg/models/v9/yolov9.yaml`（c 档基准）、`yolov9t/s/m.yaml`、`yolov9c.yaml`、`yolov9e.yaml`<br>`torchkiln/nn/modules.py`（`RepNCSPELAN4`/`RepCSP`/`RepBottleneck`/`RepConv`/`ADown`/`AConv`/`SPPELAN`/`ELAN1`/`CBLinear`/`CBFuse`） |
| 移植方式 | **结构逐层复刻** + **权重逐键对齐**：t/s/m/c 实测 **missing=0 / unexpected=1**（仅函数式 `model.22.dfl.conv.weight`） |

### 论文要解决的问题 / 核心创新

1. **信息瓶颈原理**：深网络逐层传递会丢信息（`I(X,X) ≥ I(X,f(X)) ≥ I(X,g(f(X)))`）；
2. **PGI（Programmable Gradient Information）**：训练时提供一条**辅助可逆分支**与**辅助头**，
   让深层也能拿到可靠梯度；（注：**发布检查点里只有主分支**——框架加载官方权重 `missing=0` 即证明这一点）
3. **GELAN**：用 `RepNCSPELAN4` 这种"可编程"的 ELAN 结构，兼顾参数利用率与计算效率；
4. 结果：v9c 比 YOLOv7-AF **少 42% 参数、少 21% 计算量**且精度相当；v9e 比 v8x **少 15% 参数、少 25% 计算量、+1.7 AP**。

### 官方指标（ultralytics 文档 Detection(COCO) 表）

| 型号 | mAP50-95 | mAP50 | 参数(M) | FLOPs(B) |
|---|---|---|---|---|
| YOLOv9t | **38.3** | 53.1 | 2.0 | 7.7 |
| YOLOv9s | **46.8** | 63.4 | 7.2 | 26.7 |
| YOLOv9m | **51.4** | 68.1 | 20.1 | 76.8 |
| YOLOv9c | **53.0** | 70.2 | 25.5 | 102.8 |
| YOLOv9e | **55.6** | 72.8 | 58.1 | 192.5 |

（seg 版：`yolov9c-seg` box 52.4 / mask 42.2；`yolov9e-seg` box 55.1 / mask 44.3。）

---

## 2. 网络结构

### 2.1 整体框图（`v9/yolov9.yaml` + `scale: c`）

```
Input (B,3,640,640)
  │  [0] Conv(3→64,  k3, s2)      → P1 320×320
  ├─ [1] Conv(64→128, k3, s2)     → P2 160×160
  ├─ [2] RepNCSPELAN4(256,128,64,1)        ← ★ GELAN 块（GELAN 的核心）
  ├─ [3] ADown(256)                        ← ★ 平均池 + 最大池 双路下采样 → P3 80×80
  ├─ [4] RepNCSPELAN4(512,256,128,1)
  ├─ [5] ADown(512)                        → P4 40×40
  ├─ [6] RepNCSPELAN4(512,512,256,1)
  ├─ [7] ADown(512)                        → P5 20×20
  ├─ [8] RepNCSPELAN4(512,512,256,1)
  └─ [9] SPPELAN(512,256)                  ← ★ SPP-ELAN
       │
       ▼ Neck（PAN，用 ADown 做下采样而非 strided Conv）
  [10] Upsample×2  [11] Concat([10],[6])  [12] RepNCSPELAN4(512,512,256,1)
  [13] Upsample×2  [14] Concat([13],[4])  [15] RepNCSPELAN4(256,256,128,1)  → T2 (P3)
  [16] ADown(256)  [17] Concat([16],[12]) [18] RepNCSPELAN4(512,512,256,1)  → T3 (P4)
  [19] ADown(512)  [20] Concat([19],[9])  [21] RepNCSPELAN4(512,512,256,1)  → T4 (P5)
       ▼
  [22] Detect([15,18,21], nl=3, strides=[8,16,32])   ← legacy Conv 头 + DFL(reg_max=16)
```

### 2.2 逐模块说明

| 模块 | 框架实现 | 结构 | 作用 |
|---|---|---|---|
| **`RepNCSPELAN4`** | `modules.py` | `cv1=Conv(c1,c3,1)` 切两半 → `cv2=Seq(RepCSP(c3/2,c4,n), Conv(c4,c4,3))` → `cv3=Seq(RepCSP(c4,c4,n), Conv(c4,c4,3))` → `cv4=Conv(c3+2*c4, c2, 1)`（**ELAN 式逐级拼接**） | ★ GELAN 主块 |
| **`RepCSP`** | `modules.py` | **继承 `C3`（不是 `C2f`）**：`cv1/cv2=Conv(c1,c_,1)`、`cv3=Conv(2c_,c2,1)`；`m = n×RepBottleneck(c_,c_,shortcut,e=1.0)` | ★ GELAN 内的 CSP |
| `RepBottleneck` | `modules.py` | 继承 `Bottleneck`，把 `cv1` 换成 **`RepConv`** | 可重参数化瓶颈 |
| **`RepConv`** | `modules.py` | `forward = act(conv1(x) + conv2(x) + id)`；`conv1=Conv(k3,g=g)`、`conv2=Conv(k1, p=p-k//2)`、可选 `bn` 恒等分支；`default_act=SiLU` | 3×3+1×1+恒等的可融合块 |
| **`ADown`** | `modules.py` | `avg_pool2d(2,1,0,False,True)` → 切两半：`cv1=Conv(k3,s2)`；`cv2=Conv(k1)` 套在 `max_pool2d(3,2,1)` 上 → concat | ★ v9 的下采样 |
| **`SPPELAN`** | `modules.py` | `cv1(1×1)` → 3 个 `MaxPool(k5)` **串联** → concat → `cv5(1×1)` | ★ 替代 SPPF |
| `ELAN1`（仅 t/s） | `modules.py` | `cv1` 切半 → `cv2(3×3)`、`cv3(3×3)` → `cv4(cat(a,b,b1,b2))` | t/s 档的轻量 ELAN |
| `AConv`（仅 t/s/m） | `modules.py` | `avg_pool2d(2,1,0,False,True)` + `Conv(k3,s2)` | t/s/m 档的下采样（**不是 ADown**） |
| `CBLinear` / `CBFuse`（仅 e） | `modules.py` | `CBLinear`：1×1 卷积一次产出多组通道并 `split`；`CBFuse`：按 `idx` 选组 + `interpolate` 对齐后**求和** | ★ e 档的渐进式辅助融合 |
| `Detect`（legacy） | `modules.py` | plain Conv 塔 + `4*reg_max`/`nc` 输出 | v9 走 legacy 头 |

### 2.3 v9 是"**按规模分族**"的（关键，不能只换 scale）

| 档位 | 专用 YAML | 主块 | 下采样 | 特殊块 |
|---|---|---|---|---|
| **t** | `yolov9t.yaml` | `ELAN1`（+少量 `RepNCSPELAN4`） | **`AConv`** | — |
| **s** | `yolov9s.yaml` | `ELAN1` + `RepNCSPELAN4` | **`AConv`** | — |
| **m** | `yolov9m.yaml` | `RepNCSPELAN4` | **`AConv`**（非 ADown） | — |
| **c** | `yolov9.yaml`（`scales: {c: [1.0,1.0,512]}`） | `RepNCSPELAN4` | **`ADown`** | — |
| **e** | `yolov9e.yaml` | `RepNCSPELAN4`（n=2） | **`ADown`** | **`CBLinear` + `CBFuse`** |

> ⚠️ **v9 各档的通道是"烘焙固定"的**（`yolov9t/s/m.yaml` 的 `scales` 只有自己那一档、值为 `[1.0,1.0,512]`），
> **不能**用 `scale: n` 去缩放（会得到错的通道数）。这是与 v8/v11/v12（一份 yaml + 五档 scale）最大的不同。

### 2.4 与上一代（YOLOv8）的差异

| 对比对象 | YOLOv8 | **YOLOv9** | 为什么改 | 效果 |
|---|---|---|---|---|
| 主块 | `C2f`（Bottleneck） | **`RepNCSPELAN4`（GELAN）** | 可编程的分支聚合 + RepConv | 同参更准 |
| 下采样 | strided `Conv` | **`ADown`**（avg+max 双路） | 减少下采样信息损失 | — |
| 池化块 | `SPPF` | **`SPPELAN`** | 与 ELAN 结构统一 | — |
| 训练侧 | — | **PGI**（辅助可逆分支/辅助头，**不在发布检查点内**） | 缓解信息瓶颈 | 收敛更稳 |
| 头 | legacy Conv + DFL(16) | **同**（legacy Conv + DFL(16)） | — | 复用 v8 损失/后处理 |
| 官方 mAP(c) | v8m 50.2 / 25.9M | **v9c 53.0 / 25.5M** | 综合 | 同参 **+2.8** |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 头类型 | **`Detect`（legacy plain Conv 塔）**；实测 `model.22`，`nl=3`，`no=144`（nc=80, reg_max=16） |
| 输出布局 | 每尺度 `[reg(4*16), cls(80)]` |
| Anchor | **无**（anchor-free，网格中心 `+0.5`） |
| 解码 | `dfl_project(reg_logits, 16)` → `dist2bbox` → `× stride` |
| 分配器/损失 | **单 assigner 的 `v8DetectionLoss`**（`DetLoss`，TAL `topk=10 / alpha=0.5 / beta=6.0`），**非 end2end**（无 one2one 分支） |
| NMS | `torchvision.ops.nms`；评估 `conf 0.001 / iou 0.7`，推理 `conf 0.25 / iou 0.45`；`max_det 300 / max_nms 3000` |
| strides | `[8, 16, 32]` |

---

## 4. 配置与用法

### 4.1 最小配置（c 档临界写法，务必照抄）

```yaml
Architecture:
  model_family: yolo
  task: detect
  algorithm: yolov9
  yaml_file: torchkiln/cfg/models/v9/yolov9.yaml   # ★ c 档用这个
  scale: c                                        # ★ 必须 c（yaml 的 scales 键就是 c）
  in_channels: 3
  Head:
    num_classes: 80
    reg_max: 16                                   # ★ 加载官方权重必须 16
Loss:   {name: DetLoss, topk: 10, alpha: 0.5, beta: 6.0, cls_gain: 0.5, box_gain: 7.5, dfl_gain: 1.5}
Metric: {name: DetMetric, main_indicator: mAP50-95}
PostProcess: {name: DetPostProcess, conf_thres: 0.001, iou_thres: 0.7, strides: [8, 16, 32]}
Train:
  loader: {batch_size_per_card: 8, num_workers: 4}
```

| 档位 | `yaml_file` | `scale` |
|---|---|---|
| t | `torchkiln/cfg/models/v9/yolov9t.yaml` | **`t`** |
| s | `torchkiln/cfg/models/v9/yolov9s.yaml` | **`s`** |
| m | `torchkiln/cfg/models/v9/yolov9m.yaml` | **`m`** |
| c | `torchkiln/cfg/models/v9/yolov9.yaml` | **`c`** |
| e | `torchkiln/cfg/models/v9/yolov9e.yaml` | **`e`**（且该 yaml **无 scales**，须补 `scale_params: [1.0, 1.0, 1024]`，见 §9） |

### 4.2 四条链路

```bash
# 结构自检
tkiln check -c configs/yolo/yolov9c-det.yml

# 训练（微调官方 v9c）
tkiln train -c configs/yolo/yolov9c-det.yml \
    -o Architecture.scale=c -o Architecture.Head.reg_max=16 \
    -o Global.pretrained_model=weights/yolov9c.pth

# 评估
tkiln val -c configs/yolo/yolov9c-det.yml -o Architecture.scale=c \
    --weights output/yolov9c-det/best_accuracy.pth

# 预测 / 导出
tkiln predict -c configs/yolo/yolov9c-det.yml -o Architecture.scale=c --weights ... --input imgs/
tkiln export  -c configs/yolo/yolov9c-det.yml -o Architecture.scale=c --weights ... --save-dir output/onnx --onnx
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ | 官方参数(M)² | 官方 FLOPs(B)² | 官方 mAP50-95² | 官方 .pt³ | head idx / strides |
|---|---|---|---|---|---|---|
| t | **2.13** | 2.0 | 7.7 | 38.3 | 4.7 MB | 22 / 8-16-32 |
| s | **7.32** | 7.2 | 26.7 | 46.8 | — | 22 / 8-16-32 |
| m | **20.22** | 20.1 | 76.8 | 51.4 | — | 22 / 8-16-32 |
| c | **25.59** | 25.5 | 102.8 | 53.0 | 49.4 MB | 22 / 8-16-32 |
| e | **58.20**⁴ | 58.1 | 192.5 | 55.6 | 112.1 MB | 42 / 8-16-32 |

¹ 本框架实测（`nc=80`、`reg_max=16`）。² ultralytics 官方文档表。³ 本机官方权重磁盘体积（fp16）。
⁴ e 档必须给 yaml 补 `scale_params`（本框架没有 `scales` 键时会回落 `n` 的缩放，见 §9），补后实测 **58,202,928**。

> ⚠️ **本框架未集成 FLOPs 计数**；表中 FLOPs 是 ultralytics 官方值。
> v9 训练比同档 v8 **更慢、更吃资源**（ultralytics 官方也这么提示）。

---

## 6. 公开指标

| 数据集/口径 | 指标 | ultralytics 官方 | **本框架实测** | 差异原因 |
|---|---|---|---|---|
| COCO val2017 | mAP50-95 t/s/m/c | 38.3 / 46.8 / 51.4 / 53.0 | 权重加载全 **missing=0 / unexpected=1** | 仅函数式 `model.22.dfl.conv.weight` |
| COCO val2017 | mAP50-95 e | 55.6 | 需补 `scale_params` 后 **missing=0 / unexpected=6** | 6 = 5 个 `CBLinear.conv.bias` + 函数式 dfl（见 §9） |
| `dx_ocr` 车牌（nc=1，6ep） | mAP50-95 | 0.794 | **0.789** | 数据顺序/随机性 |

**对齐验证（单步，同权重同输入同 GT，`reg_max=16`，单 assigner）**：

| 档位 | 本框架 total | ultralytics total | 三分量（box / cls / dfl） | 梯度 maxdiff |
|---|---|---|---|---|
| **c** | **14.742226** | **14.742228** | 0.3681 / 16.4717 / 2.4970（全对齐） | **0.000231**（worst `model.2.cv3.1.conv.weight`） |
| **t** | **19.053757** | **19.053761** | 0.5065 / 22.2459 / 2.7545 | 0.00044 |
| **s** | **18.813862** | **18.813862** | 0.7017 / 12.8065 / 4.7653 | 0.00029 |
| **m** | **18.858110** | **18.858112** | 0.4338 / 22.9186 / 2.7635 | 0.00024 |

> **v9c 的梯度 maxdiff 0.0786 不是 bug**：该层（`model.2.cv4.conv.weight`）梯度**绝对值本身约 67.4**，
> 相对误差仅 **~0.1%**，属 cuDNN 卷积反向的算子级微差（前向 o1/o2 已 diff=0）。
> t/s/m 的梯度 maxdiff 小（2e-4 量级）是因为对应层梯度幅度本就小。

**端到端训练（`dx_ocr`，nc=1，6 epoch，SGD/batch=8，关增广，EMA 0.9999）**：

| epoch | 1 | 2 | 3 | 4 | 5 | 6（终值） |
|---|---|---|---|---|---|---|
| 本框架 mAP50-95 | 0.566 | 0.698 | 0.722 | 0.724 | 0.744 | **0.789** |
| ultralytics | 0.001 | 0.655 | 0.657 | 0.719 | 0.779 | **0.794** |

---

## 7. 选型建议

| 场景 | 推荐档位 | 理由 |
|---|---|---|
| 边缘（超轻） | **t**（2.13M / 7.7 GFLOPs，38.3） | 比 y11n 略低 mAP 但更省算力 |
| 通用检测 | **s / m** | 46.8 / 51.4 |
| **精度-效率平衡点** | **c**（25.59M，53.0） | 官方"42% fewer params than YOLOv7-AF"，且与 v8m 同参下 +2.8 |
| 离线最高精度 | e（58.2M，55.6） | ⚠️ 当前需给 yaml 打 `scale_params` 补丁，且 §9 已列 6 个未对齐键 |
| 分割 | `yolov9c-seg` / `yolov9e-seg` | box 52.4/55.1，mask 42.2/44.3 |
| 训练资源紧张 | 建议改 v8/v11 | v9 训练**更慢更吃显存**（官方提示） |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs YOLOv8** | 同档参数下 v9 更高（v9c 53.0 vs v8m 50.2）；代价是 `RepNCSPELAN4`/`ADown`/`SPPELAN` 更复杂、训练更慢 |
| **vs YOLOv5u** | v5u 是"经典 CSP + PAN"；v9 是"GELAN + PGI"，参数量效率明显更高（v9t 2.13M/38.3 vs v5nu 2.65M/34.3） |
| **vs YOLOv10 / YOLO26** | 后两者主打 **NMS-free**；v9 仍是"anchor-free + NMS"的经典形态 |
| **vs YOLO11** | y11n 2.6M/39.5 与 v9t 2.13M/38.3 接近；v11 的 `C3k2/C2PSA` 更易部署 |
| **PGI 的边界** | PGI 的辅助可逆分支/辅助头**不在发布检查点中**（框架加载官方权重 missing=0 即证），
所以本框架"只复刻主分支"与官方权重完全一致；若要复现论文的 PGI 训练收益，需要自行搭建辅助分支 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`RepCSP` 必须继承 `C3`，不是 `C2f`（最容易踩的坑）**：
  ultra 的 `RepCSP`（`cv1/cv2 = Conv(c1,c_,1)`、`cv3 = Conv(2c_,c2,1)`、`m = n×RepBottleneck`）与 **`C3` 同构**。
  最初误继承 `C2f` 会导致 `c=32 vs 16`、键名与通道全不符（例如 `model.2.0.cv1.conv` 之类）。
  改成继承 `C3` 后参数从 25.59M 一路对上官方 25.59M。
- ⚠️ **每档必须用对应 YAML + 对应 `scale`**：`yolov9t/s/m.yaml` 的 `scales` 只有自身那一档、值为 `[1.0,1.0,512]`。
  若写成 `scale: n`，`graph.py` 会 `yaml_scales.get("n") → None` → **回落到 `DEFAULT_SCALES["n"]=(0.33,0.25,1024)`**，
  静默得到"缩小到 1/4 宽"的错误模型（实测：v9t 由 2,128,704 参数变成 1,007,098）。**必须显式写 `scale: t/s/m`**。
- ⚠️ **c 档要用 `yolov9.yaml` + `scale: c`**，不要用 `yolov9c.yaml`（后者无 `scales` 键 → 同样回落）。
  本框架的 `configs/yolo/yolov9*.yml` 示例配置默认写 `scale: n`，**加载官方权重前必须覆盖**。
- ⚠️ **e 档当前不可用（未对齐）**：`yolov9e.yaml` 没有 `scales` 键，且 `Architecture` 无法注入 `scale_params`
  （`graph.py` 只从 YAML 顶层读 `scale_params`）。给 yaml 加 `scale_params: [1.0, 1.0, 1024]` 后：
  模型 **58,202,928** 参数，加载官方权重 **missing=0 / unexpected=6**，其中 5 个是
  **`CBLinear.conv.bias`**（ultra 的 `CBLinear` 用 `nn.Conv2d(..., bias=True)`，框架用的是 `bias=False`；改成 `bias=True` 即可消掉），
  另 1 个是函数式 `model.42.dfl.conv.weight`。**但前向/损失/梯度尚未做单步对齐验证**，故按"未对齐"对待。
- ⚠️ **`RepConv` 必须按 ultra 新实现**：`default_act=SiLU`、`bn` 恒等分支、`act` 属性、
  `forward = act(conv1(x)+conv2(x)+id_out)`；`conv1=Conv(k3,p=1,g=g)`、`conv2=Conv(k1,p=p-k//2,g=g)`，默认 `bn=False`。
- ⚠️ **v9c 梯度 maxdiff 0.0786 是算子级**（梯度幅度 67.4，相对 0.1%），不要误判为逻辑 bug。
- ⚠️ 通用修复（DFL target 不 clamp、评估 fp32、`torchvision` NMS、`accumulate` 读取位置、Booststrap 等）见 `yolo11.md` §9。
- ⚠️ 本框架**未集成 FLOPs 统计**，§5 的 FLOPs 为官方值。
