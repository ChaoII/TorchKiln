# YOLOv6（Meituan，Ultralytics YAML 版）

> **定位**：美团 2022 年提出的工业级实时检测器。**本框架只复刻了它的 YAML 结构骨架（`torchkiln/cfg/models/v6/yolov6.yaml`），
> 没有官方权重、也没有做权重/损失/指标对齐**。本文件把真实状态说清楚，避免误用。
> **任务**：`detect`
> **权重**：**无**（ultralytics 官方明确"不发布 YOLOv6 的 `.pt`"）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文（v3.0） | **YOLOv6 v3.0: A Full-Scale Reloading**，Chuyi Li 等，**arXiv:2301.05586**（2023） |
| 论文（初版） | YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications，**arXiv:2209.02976**（2022） |
| 机构 | 美团（Meituan） |
| 官方代码 | `https://github.com/meituan/YOLOv6` |
| 官方文档（Ultralytics 转述） | `https://docs.ultralytics.com/models/yolov6` |
| 本框架实现 | `torchkiln/cfg/models/v6/yolov6.yaml`（**仅结构**）<br>`configs/yolo/yolov6-det.yml`（构建/训练用的示例配置） |
| 移植方式 | **只按 ultralytics 的 `yolov6.yaml` 复刻网络骨架**；未做权重转换、未做单步 loss/梯度对比、未做端到端训练对比 |

### 论文要解决的问题 / 主要创新（论文口径）

YOLOv6 面向**工业部署**，针对"速度-精度"折中做了一整套工程改造：
1. **Bi-directional Concatenation（BiC）**：颈部双向拼接，增强定位信号；
2. **Anchor-Aided Training（AAT）**：训练时用 anchor 辅助、推理退化为 anchor-free；
3. **重设计的 backbone/neck**（多一级 stage）+ **SimCSPSPPF** 块；
4. **自蒸馏（self-distillation）**：提升小模型表现。

> ⚠️ **本框架没有实现 BiC / AAT / SimCSPSPPF / RepBlock / 自蒸馏**。
> 本框架的 `yolov6.yaml` 是 **ultralytics 版的精简复现**（纯 `Conv` 堆叠 + `ConvTranspose2d` 上采样），
> **不等于**美团官方网络，因此**官方 mAP 数字不可直接挂在它身上**。

### 官方报告的指标（来源：ultralytics 文档转述美团数据，T4 GPU）

| 型号 | COCO mAP50-95 | T4 FPS |
|---|---|---|
| YOLOv6-N | **37.5** | 1187 |
| YOLOv6-S | **45.0** | 484 |
| YOLOv6-M | **50.0** | 226 |
| YOLOv6-L | **52.8** | 116 |
| YOLOv6-L6 | "实时场景 SOTA"（官方表述） | — |

> ⚠️ 另一个常被引用的来源（YOLOv10 论文的对比表）给的 YOLOv6-3.0 参数量是
> **N 4.7M / S 18.5M / M 34.9M / L 59.6M**、AP **37.0 / 44.3 / 49.1 / 51.8**。
> 两处数字口径不同，且**都与本框架 yaml 的参数量不一致**（见 §5）——再次说明这是"另一个网络"。

---

## 2. 网络结构

### 2.1 整体框图（`v6/yolov6.yaml`，n 档）

```
Input (B,3,640,640)
  │
  ├─ [0] Conv(3→64,   k3, s2)        → P1 320×320
  ├─ [1] Conv(64→128, k3, s2)        → P2 160×160
  ├─ [2] Conv(128→128,k3, s1) ×6     ← ★ 纯 Conv 堆叠（不是 C3/C2f/CSP）
  ├─ [3] Conv(128→256,k3, s2)        → P3  80×80
  ├─ [4] Conv(256→256,k3, s1) ×12
  ├─ [5] Conv(256→512,k3, s2)        → P4  40×40
  ├─ [6] Conv(512→512,k3, s1) ×18
  ├─ [7] Conv(512→1024,k3,s2)        → P5  20×20
  ├─ [8] Conv(1024→1024,k3,s1) ×6
  └─ [9] SPPF(1024, k5)
       │
       ▼ Neck（PAN，用 ConvTranspose2d 做上采样）
  [10] Conv(1024→256,k1)  [11] ConvTranspose2d(256, k2, s2, p0)
  [12] Concat([11],[6])   [13] Conv(256→256,k3)  [14] Conv(256→256,k3)×9
  [15] Conv(256→128,k1)   [16] ConvTranspose2d(128, k2, s2, p0)
  [17] Concat([16],[4])   [18] Conv(128→128,k3)  [19] Conv(128→128,k3)×9   → T2 (P3)
  [20] Conv(128→128,k3,s2) [21] Concat([20],[15]) [22] Conv(256→256,k3)
  [23] Conv(256→256,k3)×9                                                    → T3 (P4)
  [24] Conv(256→256,k3,s2) [25] Concat([24],[10]) [26] Conv(512→512,k3)
  [27] Conv(512→512,k3)×9                                                    → T4 (P5)
       ▼
  [28] Detect([19, 23, 27], nl=3, strides=[8,16,32])   ← 见 §3（新 DWConv 头 + reg_max=1）
```

### 2.2 逐模块说明

| 模块 | 框架实现 | 结构 | 备注 |
|---|---|---|---|
| `Conv` | `modules.py::Conv` | `Conv2d(bias=False) → BN → **SiLU**` | ⚠️ yaml 里写的 `activation: torch.nn.ReLU()` **框架没有读取**，实际激活是 SiLU |
| `Conv`（重复堆叠） | 同上 | YAML 的 `n` 重复由 `nn.Sequential` 包装 | 例如 `[-1, 6, Conv, [128, 3, 1]]` → `model.2.0 … model.2.5` |
| `SPPF` | `modules.py::SPPF` | `cv1(1×1) → MaxPool(k5)×3 → cat → cv2(1×1)`；`args` ≤3 时会恢复 `cv1` 的 SiLU（legacy 行为） | 与 v5/v8 相同 |
| `nn.ConvTranspose2d` | torch 原生（`get_module("nn.*")` 直通） | `ConvTranspose2d(c1, 256, k=2, s=2, p=0, bias=True)` | **无 BN/无激活**（v6 yaml 的原样写法） |
| `Concat` / `Detect` | `graph.py` + `modules.py` | 见 §3 | 头 idx=28 |

### 2.3 与本仓库其它家族的差异

| 对比对象 | YOLOv6（本框架 yaml） | YOLOv8 | YOLO11 |
|---|---|---|---|
| 特征提取单元 | **纯 `Conv` 堆叠** | `C2f`（CSP + Bottleneck） | `C3k2`（+`C2PSA`） |
| 颈上采样 | **`ConvTranspose2d`+`Concat`+Conv 堆叠** | `nn.Upsample` + `C2f` | `nn.Upsample` + `C3k2` |
| 检测头 | **新 DWConv 头（`Detect26`）**，`reg_max=1`（**无 DFL**） | legacy Conv 头 + DFL(16) | 新 DWConv 头 + DFL(16) |
| BiC / AAT / 自蒸馏 | **未实现** | 不涉及 | 不涉及 |
| 官方权重 | **无** | 有（n/s/m/l/x） | 有 |
| 对齐状态 | **未验证** | 已对齐（见 `yolov8.md`） | 已对齐（见 `yolo11.md`） |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 头类型 | **`Detect26`**（`REGISTRY["Detect"]` 的新 DWConv 头）；实测 `model.28`，`nl=3`，`no=84 = 4×1 + 80` |
| 输出布局 | 每尺度 `[reg(4), cls(nc)]`（因 `reg_max=1`，reg 只有 4 个通道） |
| Anchor | **无**（anchor-free，网格中心） |
| 解码 | `reg_max==1` → **不做 DFL**，直接 `l,t,r,b = reg`，再 `dist2bbox → ×stride`（`DetPostProcess._decode`） |
| 损失 | `DetLoss`：`reg_max<=1` 时走 **L1 分支**（对齐 ultralytics `BboxLoss` 的无 DFL 路径）+ BCE cls |
| NMS | `torchvision.ops.nms`；评估 `conf 0.001 / iou 0.7`，推理 `conf 0.25 / iou 0.45`；`max_det 300`、`max_nms 3000` |
| strides | `[8, 16, 32]` |

> ⚠️ 注意：**这不是美团 YOLOv6 的原始头**（原版是 anchor-based + obj 分支 + AAT），
> 而是"**ultralytics 的 v6 骨架 + 本框架的 anchor-free L1 头**"组合。

---

## 4. 配置与用法

### 4.1 示例配置（`configs/yolo/yolov6-det.yml` 关键项）

```yaml
Architecture:
  model_family: yolo
  task: detect
  algorithm: yolov6
  yaml_file: torchkiln/cfg/models/v6/yolov6.yaml
  scale: n                 # n/s/m/l/x 五档都在 yaml 的 scales 里
  in_channels: 3
  Head:
    num_classes: 80
    reg_max: 1             # v6 用 L1 回归（无 DFL）
Loss:   {name: DetLoss, topk: 10, alpha: 0.5, beta: 6.0, cls_gain: 0.5, box_gain: 7.5}
Metric: {name: DetMetric, main_indicator: mAP50-95}
PostProcess: {name: DetPostProcess, conf_thres: 0.001, iou_thres: 0.7, strides: [8, 16, 32]}
Train:
  dataset: {name: DetDataset, data_dir: datasets/det_demo, label_file_list: [datasets/det_demo/train.txt]}
  loader:  {batch_size_per_card: 8, num_workers: 4}
```

### 4.2 四条链路

```bash
# 结构自检（本框架唯一已验证到的程度：能构建 + 能前向 + 损失/指标可实例化）
tkiln check -c configs/yolo/yolov6-det.yml

# 训练（只能从零；没有预训练权重可加载）
tkiln train -c configs/yolo/yolov6-det.yml

# 评估
tkiln val -c configs/yolo/yolov6-det.yml --weights output/yolov6-det/best_accuracy.pth

# 预测 / 导出
tkiln predict -c configs/yolo/yolov6-det.yml --weights ... --input imgs/ --output out.jpg
tkiln export  -c configs/yolo/yolov6-det.yml --weights ... --save-dir output/onnx --onnx
```

`configs/yolo/yolov6-det.yml` 已被 `tools/check_graph_build.py` 覆盖（该工具遍历 `configs/yolo/*.yml`
并做一次 dummy 前向；仓库记录为 **55 OK / 0 FAIL**）。**除此之外没有更深的对齐证据。**

---

## 5. 规模与速度

| 档位 | yaml + scale | 参数(M)¹ | 官方 YOLOv6-3.0 参数(M)² | 官方 COCO mAP² | 备注 |
|---|---|---|---|---|---|
| n | `v6/yolov6.yaml` n | **4.19** | 4.7 | 37.5 | — |
| s | 同上 s | **15.98** | 18.5 | 45.0 | — |
| m | 同上 m | **51.14** | 34.9 | 50.0 | ⚠️ 比官方大 46% |
| l | 同上 l | **109.28** | 59.6 | 52.8 | ⚠️ 比官方大 83% |
| x | 同上 x | **170.38** | 未列出 | 未列出 | ⚠️ 更大 |

¹ 本框架实测（`nc=80`，`reg_max=1`，本机 `sum(p.numel())`）。
² 官方值**不是**本框架的对应值：mAP 来自 ultralytics 文档转述的美团数据；参数量来自 YOLOv10 论文的对比表。
**本框架的 yaml 与官方网络不同（见 §2.3），所以这一行的官方列只能当"背景参考"**。

> ⚠️ **本框架未集成 FLOPs 计数**，且**没有 v6 的官方权重**可测推理速度；
> 上表不列 FLOPs / 推理耗时，避免编造。

---

## 6. 公开指标

| 数据集 | 指标 | 官方值（美团/文档） | **本框架实测** | 差异原因 |
|---|---|---|---|---|
| COCO val2017 | mAP50-95 (N) | 37.5 | **未验证** | 无官方权重、结构为 ultralytics 精简复现、未做对齐 |
| COCO val2017 | mAP50-95 (S/M/L) | 45.0 / 50.0 / 52.8 | **未验证** | 同上 |
| — | 权重加载 missing/unexpected | — | **无权重可测** | ultralytics 不发布 v6 `.pt` |
| — | 单步 loss / 梯度对比 | — | **未做** | 无参考实现可对（官方头/损失与本框架不同） |
| — | 端到端训练对比 | — | **未做** | 同上 |

**本框架已具备的能力**：YAML→网络构建、dummy 前向、`DetLoss`/`DetMetric`/`DetPostProcess` 接入、
CLI 训练/评估/导出链路（`check_graph_build` 的 55 OK 覆盖）。

**明确缺失的能力**：BiC、AAT、SimCSPSPPF、RepBlock、自蒸馏；anchors/obj 分支；官方权重；任何数值对齐证据。

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| 生产/交付项目 | **不要选 v6**（无预训练、无对齐、结构非官方） |
| 想复现美团 YOLOv6 论文 | **不要用本框架的 v6**；请用官方 `meituan/YOLOv6` 仓库 |
| 教学：对比"纯 Conv 堆叠 vs CSP" | 可用 `configs/yolo/yolov6-det.yml`，与 `yolov8-det` / `yolov5-det` 横向比较 |
| 需要更小/更准的替代 | `yolo11n`（2.6M/39.5）、`yolo26n`（2.6M/40.9）、`yolov8n`（3.2M/37.3） |
| 需要 seg/pose/obb/cls | v6 家族**只有检测** YAML |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs 官方 YOLOv6（美团）** | 官方 = RepBlock/SimCSPSPPF + BiC/AAT/自蒸馏 + FastAPI 部署工具链；本框架 = 纯 Conv 骨架 + 本框架统一头。**两者只有"名字"相同** |
| **vs YOLOv8** | v8 用 `C2f` CSP 单元，参数量-精度更高效；官方权重可用 |
| **vs YOLOv5u** | v5u 的骨架同样朴素（C3+SPPF+PAN），但有官方权重与完整对齐记录（见 `yolov5.md`） |
| **定位** | v6 在本仓库是"**结构占位**"：证明 YAML 解析器能吃下 `ConvTranspose2d` + 纯 Conv 堆叠 + `nn.*` 直通模块 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`activation: torch.nn.ReLU()` 是"死配置"**：`torchkiln/nn/graph.py::parse_model` **没有读取 YAML 顶层的 `activation` 字段**，
  v6 全部 `Conv` 实际使用默认的 **SiLU**。若要 ReLU，需要自己改 `Conv.default_act` 或加解析逻辑。
- ⚠️ **没有预训练权重**：ultralytics 明确不发布 YOLOv6 的 `.pt`；本机官方权重目录 `\\tsclient\D\项目资料\ultralytics_models\`
  下也**没有 v6 子目录**（只有 v3u/v5/v8/v9/v10/v11/v12/v26）。
- ⚠️ **参数量与官方差距大（m/l/x 尤甚）**：说明 ultralytics 的 `yolov6.yaml` 与美团官方网络不同；
  不要把官方 mAP 直接当作本 yaml 的预期表现。
- ⚠️ **`scale` 必须在 `scales` 里**：`v6/yolov6.yaml` 的 `scales` 含 `n/s/m/l/x`（五档齐全），
  所以 `scale: n..x` 都合法；但注意 m 的 `max_channels=768`、l/x 的 `512`（宽度 1.0/1.25 会被上限截断）。
- ⚠️ **头是新 DWConv 头且 `reg_max=1`**：v6 家族路径不含 `/v6/` 的 legacy 判定，
  因此走 `REGISTRY["Detect"] = Detect26`（DWConv 分类塔），回归是不带 DFL 的 4 通道 + L1 损失。
- ⚠️ 通用注意事项（DFL target 不 clamp、评估 fp32、`torchvision` NMS、`accumulate` 读取位置、
  `imgsz=1024` 用 batch=4 等）同样适用，见 `yolo11.md` §9。
