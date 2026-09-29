# YOLOv3（含 Ultralytics `u` 系列：yolov3u / yolov3-sppu / yolov3-tinyu）

> **定位**：Redmon 2018 的经典单阶段检测器；本框架复刻的是 **Ultralytics 的 `u` 变体**——
> 保留 v3 的 Darknet-53 + FPN 骨架，把 **anchor-based + obj 分支的头换成 YOLOv8 的 anchor-free 头**。
> **任务**：`detect`
> **权重**：`yolov3u.pt` / `yolov3-sppu.pt` / `yolov3-tinyu.pt`
> （本机官方权重目录 `\\tsclient\D\项目资料\ultralytics_models\yolov3u\`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 原始论文 | **YOLOv3: An Incremental Improvement**，Joseph Redmon & Ali Farhadi，**arXiv:1804.02767**（2018） |
| `u` 变体 | **无独立论文**（Ultralytics 工程改进：把 v8 的 anchor-free / objectness-free split head 装到 v3 骨架上） |
| 官方代码 | `https://github.com/ultralytics/ultralytics`（cfg `v3/yolov3*.yaml`）<br>`https://github.com/ultralytics/yolov3`（原始 anchor-based 实现，**与本框架不兼容**） |
| 官方文档 | `https://docs.ultralytics.com/models/yolov3` |
| 本框架实现 | `torchkiln/cfg/models/v3/yolov3.yaml`、`yolov3-spp.yaml`、`yolov3-tiny.yaml`（结构）<br>`torchkiln/nn/graph.py::parse_model`（YAML→网络）<br>`torchkiln/nn/modules.py`（`Conv`/`Bottleneck`/`SPP`/`Detect`） |
| 移植方式 | **结构逐层复刻** + **权重逐键对齐**（实测 `missing=0 / unexpected=1`，仅差函数式 `dfl.conv.weight`） |

### 论文要解决的问题

原版 YOLOv3 的两条主线：
1. **Darknet-53 主干**（残差 Bottleneck 堆叠）+ **多尺度预测**（3 个尺度，13×13 / 26×26 / 52×52）；
2. **锚框 + objectness + 多标签 sigmoid 分类**的检测头。

Ultralytics 的 `u` 变体做的唯一改动是**换头**：
- 去掉 **预定义 anchors** 与 **objectness 分支**，改为 **网格中心 anchor-free + split head（`reg` / `cls` 两条塔）**；
- 回归改用 **DFL（Distribution Focal Loss，`reg_max=16`）**，损失由 `v8DetectionLoss`（TAL 分配）接管；
- 好处：不再依赖 anchor 超参，对不同尺度/长宽比目标更鲁棒（官方说法）。

> ⚠️ **小差异要知道**：`u` 变体的头是 **v8 风格 anchor-free**，而 **v3 的骨架/颈部完全保留**
> （包括 `Conv(32,3,1)` 开头、每级 1×1+3×3 的 Bottleneck 链、FPN 只做自顶向下）。
> Ultralytics 官方**没有发布 v3u 的 COCO mAP 对比表**（文档只列支持的任务/模式），
> 因此本文件不引用任何 v3u 的公开 mAP 数字。

### 关键结论

| 项 | 结论 |
|---|---|
| 权重加载 | 三型号全 **missing=0 / unexpected=1**（唯一未加载键是函数式 `model.{28,20}.dfl.conv.weight`） |
| 单步 loss（同权重同输入同 GT） | u / spp / tiny 三分量全对齐，total 差 ≤5e-6（见 §6） |
| 显存 | 参数 **12M~105M**，fp16 权重文件 **23MB~200MB**（见 §5） |
| 适用场景 | 仅建议**老项目迁移**或**教学对比**；新项目用 v8/v11/v26 |

---

## 2. 网络结构

### 2.1 整体框图（`yolov3.yaml`）

```
Input (B,3,640,640)
  │  Conv(3→32, k3, s1)                     [0]
  ├─ Conv(32→64, k3, s2)     → P1 320×320   [1]
  ├─ Bottleneck(64)×1                       [2]
  ├─ Conv(64→128, k3, s2)    → P2 160×160   [3]
  ├─ Bottleneck(128)×2                      [4]  → nn.Sequential(2 个)
  ├─ Conv(128→256,k3, s2)    → P3  80×80    [5]
  ├─ Bottleneck(256)×8                      [6]
  ├─ Conv(256→512,k3, s2)    → P4  40×40    [7]
  ├─ Bottleneck(512)×8                      [8]
  ├─ Conv(512→1024,k3,s2)    → P5  20×20    [9]
  └─ Bottleneck(1024)×4                     [10]
       │
       ▼ FPN（只做自顶向下；-2 表示"再往前一层"取出 P4 通路）
  [11] Bottleneck(1024, shortcut=False)
  [12] Conv(1024→512,k1)   [13] Conv(512→1024,k3)   [14] Conv(1024→512,k1)
  [15] Conv(512→1024,k3)  → P5 输出（大目标）
  [16] Conv(1024→256,k1, from -2=[14])
  [17] Upsample ×2   [18] Concat([17], [8])   [19][20] Bottleneck(512, F)
  [21] Conv(512→256,k1)   [22] Conv(256→512,k3)  → P4 输出（中目标）
  [23] Conv(512→128,k1, from -2=[20])
  [24] Upsample ×2   [25] Concat([24], [6])   [26][27] Bottleneck(256, F)×2
                                                            → P3 输出（小目标）
       ▼
  [28] Detect([27, 22, 15])   ← legacy 头（plain Conv 塔），anchor-free + DFL(reg_max=16)
```

`yolov3-spp.yaml` 只改 **head[1]**：把 `Bottleneck(1024,F)` 换成 **`SPP [512, [5,9,13]]`**（并行 3 个最大池化 + 1×1 压缩），其余完全相同。

`yolov3-tiny.yaml` 用 **MaxPool 下采样 + ZeroPad**，只有 **2 个检测尺度**：

```
Conv(3→16,k3,1) → MaxPool2 → Conv(32,k3,1) → MaxPool2 → Conv(64,…) → MaxPool2
→ Conv(128,…) → MaxPool2 → Conv(256,…) → MaxPool2 → Conv(512,…)
→ ZeroPad2d([0,1,0,1]) → MaxPool2d(2,1,0)            # 保下采样的"偏移 pad"
→ Conv(1024,k3,1) → Conv(256,k1,1) → Conv(512,k3,1)  # 15 → P5(32×) 分支
→ Conv(128,k1,1,from -2) → Upsample2 → Concat(→ layer 8)
→ Conv(256,k3,1)                                     # 19 → P4(16×) 分支
→ Detect([19, 15], nl=2, strides=(16,32))
```

### 2.2 逐模块说明

| 模块 | 框架实现 | 结构 | 作用 |
|---|---|---|---|
| `Conv` | `modules.py::Conv` | `Conv2d(bias=False) → BN → SiLU` | 基础卷积（v3 的 `Conv` 是 **k3/s1**，与 v5 的 k6/s2 stem 不同） |
| **`Bottleneck`** | `modules.py::Bottleneck` | `cv1=Conv(1×1) → cv2=Conv(3×3)`，`add = shortcut and c1==c2` | v3 的特征提取单元（**不是 C3/C2f**） |
| `SPP`（仅 spp 版） | `modules.py::SPP` | `cv1(1×1) → MaxPool{k5,9,13} 并联 → cat → cv2(1×1)` | 扩大感受野 |
| `nn.Upsample` | torch 原生 | `scale=2, mode=nearest` | FPN 上采样 |
| `Concat` | `modules.py::Concat` | `torch.cat(dim=1)` | 特征拼接 |
| **`Detect`（legacy）** | `modules.py::Detect(_BaseHead)` | `cv2=[Conv,Conv,Conv2d]`（回归塔）+ `cv3=[Conv,Conv,Conv2d]`（分类塔），输出 `[reg(4*reg_max), cls]` | v3/v5/v8/v9 共用的**旧式 Conv 头** |

> 头路由（重要）：`graph.py::build_from_arch` 会按 `yaml_file` 路径判定家族，
> 含 `/v3/ /v5/ /v8/ /v9/` 的走 **legacy 头**（`LEGACY_HEAD`，plain Conv 塔）；
> `v10 / v11 / v12 / v26` 走**新 DWConv 头**（`Detect26 / Detect10`）。
> 本框架实测 v3u 头在 `model.28`，`nl=3`，`strides=[8,16,32]`，`no=4*16+80=144`。

### 2.3 与"上一代/同代"的差异

| 对比对象 | YOLOv3（原版） | **本框架的 v3u** | 为什么改 | 效果 |
|---|---|---|---|---|
| 检测头 | 3 尺度 + anchors + **obj 分支** + 多标签 | **anchor-free split head**（v8 头） | 去掉 anchor 超参 | 同骨架下更稳 |
| 回归 | `tx,ty,tw,th` 直接回归 | **DFL**（`reg_max=16`，softmax 期望） | 分布建模 | loss 更平滑 |
| anchors | 9 个（每尺度 3 个） | **无**（`Detect.anchors` 仅占位） | — | 配置无需调 anchor |
| 损失 | 原版 obj+cls+coord | **`DetLoss`（v8DetectionLoss 移植，TAL）** | 与 v8/v11 统一 | 见 §6 |
| 骨架/颈部 | Darknet-53 + FPN | **同原版**（不改） | — | 保留 v3 特性 |
| 参数 | 61.5M（Darknet-53 原版，80 类） | **103.75M**（w=1.0 全宽 Bottleneck 链，实测） | ultralytics 的 yaml 更宽 | 见 §5 |

---

## 3. 输出与后处理

### 3.1 输出张量

| 阶段 | 形状 | 含义 |
|---|---|---|
| 训练 | 3 个尺度（tiny 为 2 个），每个 `(B, 4*reg_max + nc, H, W)` = `(B, 144, H, W)`（nc=80, reg_max=16） | 原始 logits |
| 推理（后处理前） | 按尺度展平后拼成 `(B, 144, N)`，`N = Σ H·W` | 供解码 |

### 3.2 Anchor

**anchor-free**：每个网格中心即参考点，直接回归 `l,t,r,b` 四边距离（`DetPostProcess` 用 `make_anchors` 生成网格中心 + stride）。

### 3.3 解码公式（本框架 `det/ops.py` / `det/postprocess.py`）

```
cx = j + 0.5, cy = i + 0.5                      # 网格中心（make_anchors, offset=0.5）
cls_score = sigmoid(cls_logits)                 # 多标签 sigmoid
# 回归：reg_max>1 → DFL 期望；reg_max==1 → 直接取 4 个通道
l,t,r,b = dfl_project(reg_logits, reg_max)      # softmax(16 个 bin) · [0..15]
x1 = (cx - l) * stride, y1 = (cy - t) * stride
x2 = (cx + r) * stride, y2 = (cy + b) * stride  # 对齐 ultra dist2bbox
```

### 3.4 NMS 与阈值

| 项 | 值 |
|---|---|
| 实现 | **`torchvision.ops.nms`**（C++；原 Python 逐框循环 16800 框要 10s/图，换后 **2.5s/50 图，约 200×**） |
| 评估 | `conf_thres=0.001`、`iou_thres=0.7`、`max_det=300`、`max_nms=3000` |
| 推理 | `conf_thres=0.25`、`iou_thres=0.45` |
| strides | u/spp：`[8,16,32]`；**tiny：必须 `[16,32]`**（`nl=2`，写成三档会解码错位） |

---

## 4. 配置与用法

### 4.1 最小配置（`configs/yolo/yolov3-det.yml`，截取关键项）

```yaml
Architecture:
  model_family: yolo
  task: detect
  algorithm: yolov3
  yaml_file: torchkiln/cfg/models/v3/yolov3.yaml
  scale: n                 # ⚠️ v3 yaml 无 scales → 见 §9
  in_channels: 3
  Head:
    num_classes: 80
    reg_max: 16            # ⚠️ 加载官方 v3u 权重必须 16（demo 配置默认写 1）
Loss:   {name: DetLoss, topk: 10, alpha: 0.5, beta: 6.0, cls_gain: 0.5, box_gain: 7.5}
Metric: {name: DetMetric, main_indicator: mAP50-95}
PostProcess:
  name: DetPostProcess
  conf_thres: 0.001
  iou_thres: 0.7
  strides: [8, 16, 32]     # tiny 必须改成 [16, 32]
Train:
  dataset: {name: DetDataset, data_dir: datasets/det_demo, label_file_list: [datasets/det_demo/train.txt]}
  loader:  {batch_size_per_card: 8, num_workers: 4}
```

### 4.2 四条链路（与 `tkiln` CLI）

```bash
# 结构自检（构建模型/损失/指标，不训练）——先跑这个
tkiln check -c configs/yolo/yolov3-det.yml

# 训练（从官方 v3u 预训练微调；先设 reg_max=16 并把权重放到 pretrained_model）
tkiln train -c configs/yolo/yolov3-det.yml \
    -o Architecture.Head.reg_max=16 -o Global.pretrained_model=weights/yolov3u.pth

# 评估（官方口径 conf=0.001 / iou=0.7）
tkiln val -c configs/yolo/yolov3-det.yml --weights output/yolov3-det/best_accuracy.pth

# 预测 / 导出 ONNX
tkiln predict -c configs/yolo/yolov3-det.yml --weights ... --input imgs/ --output output/pred.jpg
tkiln export  -c configs/yolo/yolov3-det.yml --weights ... --save-dir output/onnx --onnx
```

> 加载权重的小技巧（本仓做法）：ultralytics 的 `.pt` 里 `model.28.dfl.conv.weight` 是**函数式 DFL**的键，
> 框架 DFL 无参数，所以先把该键删掉再存成 `.pth`；框架 `load_state_dict(strict=False)` 即 **missing=0**。

---

## 5. 规模与速度

| 型号 | yaml | 参数(M)¹ | 官方 .pt 体积² | 检测尺度 | 训练显存(320×320, batch8)³ |
|---|---|---|---|---|---|
| `yolov3u` | `v3/yolov3.yaml` | **103.75** | 198.3 MB | P3/P4/P5 | ~6 GB |
| `yolov3-sppu` | `v3/yolov3-spp.yaml` | **104.80** | 200.3 MB | P3/P4/P5 | ~6 GB |
| `yolov3-tinyu` | `v3/yolov3-tiny.yaml` | **12.17** | 23.3 MB | **P4/P5（stride 16/32）** | ~2 GB |

¹ 本框架实测（`nc=80`、`reg_max=16`、**本机 pytorch 构建后 `sum(p.numel())`**）。
² 官方 `*.pt` 磁盘体积（ultralytics 以 **fp16** 保存），本机 `\\tsclient\D\项目资料\ultralytics_models\yolov3u\` 实测。
³ 本机 RTX 4060 Ti 16GB 粗估；`imgsz=1024` 时请把 batch 降到 4（AGENTS「显存约束」条）。

> ⚠️ **本框架未集成 FLOPs 计数**（`thop` 等），故上表**不列 FLOPs**，避免编造。
> ⚠️ **v3 的 5 个 scale 名得到同一个网络**：`yolov3*.yaml` 用的是 `depth_multiple: 1.0 / width_multiple: 1.0`
> 且**没有 `scales` 字典**，所以 `scale: n/s/m/l/x` 都构建出 **103,754,128** 参数的同一模型（实测）。
> 想要"小号 v3"，需要自己写 `depth_multiple/width_multiple` 或加 `scales`。

---

## 6. 公开指标

**官方值**：ultralytics 文档**未发布** v3u / v3-sppu / v3-tinyu 的 COCO mAP 表（只列任务与模式支持），
因此**本框架不引用 v3u 的公开 mAP**（原版 v3 的论文数值不适用于换了头的 `u` 变体）。

**本框架实测（对齐验证）**：

| 项 | 本框架 | ultralytics | 差异 |
|---|---|---|---|
| 权重加载（u / spp / tiny） | **missing=0 / unexpected=1** | — | 仅函数式 `model.{28,28,20}.dfl.conv.weight` |
| 单步 total loss（u） | **19.047052** | **19.047056** | 4e-6（仅函数式 `model.28.dfl.conv.weight`） |
| 单步 total loss（spp） | **18.116039** | **18.116039** | 0 |
| 单步 total loss（tiny） | **14.158577** | **14.158577** | 0 |
| 三分量（u） | box/cls/dfl 全对齐 | 同 | — |
| 梯度 maxdiff（u/spp/tiny） | **0.00016 / 0.00020 / 0.00007** | — | cuDNN 卷积反向的算子级微差 |

**端到端训练（`dx_ocr` 单类车牌，nc=1，6 epoch，SGD/batch=8/关增广/EMA exponential 0.9999）**：

| epoch | 1 | 2 | 3 | 4 | 5 | 6（终值） |
|---|---|---|---|---|---|---|
| 本框架 mAP50-95 | 0.562 | 0.612 | 0.712 | 0.757 | 0.749 | **0.786** |
| ultralytics mAP50-95 | 0.348 | 0.663 | 0.679 | 0.749 | 0.772 | **0.800** |

> 终值差 **0.014**，属训练数据顺序（框架 `shuffle=false` vs ultra `shuffle=true`）+ 随机性；
> 与 v11/v8 的端到端结论同源（见 `yolo11.md` §6）。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 老项目/论文复现 v3 系列 | `yolov3u` / `yolov3-sppu` | 与 ultralytics v3u 权重逐键兼容，可作为基线 |
| 极低算力、只要"能跑" | `yolov3-tinyu` | 12M 参数 / 23MB，2 尺度 |
| 精度优先的通用检测 | **不要选 v3** | 用 `yolov8x` / `yolo11x` / `yolo26x`（更少参数、更高 mAP） |
| 需要 seg/pose/obb/cls | **不要选 v3** | v3 家族只有检测；见 `docs/models/task/` 下各任务文档 |
| 大输入 / 小目标 | 用 `yolov8-p2` / `yolo26-p2` | v3 无 P2 变体 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs 原版 YOLOv3（anchor-based）** | 本框架实现的是 **ultralytics v3u**（换头），**不能加载**原版 v3 的 Darknet 权重；原版 v3 的 anchors/obj 分支在本框架中不存在 |
| **vs YOLOv5u** | v5u 是 **C3 + SPPF + PAN** 的现代骨架；v3u 是 **Bottleneck 链 + FPN**。同权重/同输入下二者走**同一套损失与后处理**（都是 legacy 头 + v8DetectionLoss） |
| **vs YOLOv8** | v8 用 `C2f` + `k3/s2` stem；v8n 只需 **3.16M** 参数，而 v3u 要 **103.75M**（ultralytics 的 v3 yaml 不缩放通道） |
| **vs YOLOv9** | v9 引入 GELAN/PGI 与 `RepNCSPELAN4`，同精度下参数远小于 v3u |
| **基准** | 若以"同权重同输入同 GT 的单步 loss"为准，v3u/spp/tiny 均已与 ultralytics 对齐（§6） |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`REPEAT_MODULES` 必须不含 `Bottleneck`（本框架已修，勿回退）**：
  `Bottleneck` 是"**不内化 n**"的 `SCALED` 模块，YAML 里的重复数应由外层包装成 `nn.Sequential`。
  早期把它误列入 `REPEAT_MODULES`，导致 `[-1, 2, Bottleneck, [128]]` 被当成 `Bottleneck(c1, c2, shortcut=2, …)`，
  键名变成 `model.4.cv1` 而非 `model.4.0.cv1`，**v3 权重整体加载不上**。
  移除后走 `graph.py` 的 `nn.Sequential` 包装（`model.4.0 / model.4.1`），与 ultralytics 一致。
  `Bottleneck` 不直接出现在其它家族 YAML（只在 `C3/C3k` 内部），故该修复**不影响 v5/v8/v9**。
- ⚠️ **tiny 的 strides 必须是 `[16, 32]`**：`Detect` 只有 2 个尺度（`nl=2`），
  若沿用 `[8,16,32]`，解码会把 P4 特征当成 stride 8，框全部错位。
- ⚠️ **加载官方权重必须 `Head.reg_max: 16`**：`configs/yolo/yolov3-det.yml` 默认写 `reg_max: 1`（无 DFL、纯直接回归），
  只有 16 才能与官方 `yolov3u.pt` 的参数形状对上。
- ⚠️ **v3 的 5 档 scale 是同一个网络**（yaml 无 `scales`），别以为 `scale: x` 会变大。
- ⚠️ **DFL target 不可 clamp 框坐标**（本框架已修，影响所有 detect/OBB 训练）：
  只 clamp `bbox2dist` 之后的 **ltrb 距离**到 `reg_max-1-0.01`。
- ⚠️ **对比口径**：ultra 自报 mAP 用 `rect=True`（矩形 letterbox）会抬高数值，与本框架方图不可直接比；
  跨端对比请"**同一评估器评双方权重**"。
- ⚠️ `imgsz=1024` 时显存吃紧：实测 `batch=8` 即 OOM，**用 `batch=4`**。
- ⚠️ 本框架**未集成 FLOPs 统计**，§5 无 FLOPs 列。
