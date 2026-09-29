# YOLO26

> **定位**：Ultralytics 2026-01 发布的最新主力家族，是 **YOLO11 的端到端后继**：
> **原生 NMS-free（双头 + one2one）+ 彻底去掉 DFL（`reg_max=1`，改用 L1 回归）**，
> 并配套 **MuSGD 优化器**与任务专属改进（分割语义损失、姿态 RLE、OBB 角度损失）。
> **任务**：`detect`（同家族还有 `segment`/`pose`/`obb`/`classify`/`semantic`/`depth` 与 **`-p2`/`-p6`** 变体）
> **权重**：`yolo26{n,s,m,l,x}.pt`（本机 `\\tsclient\D\项目资料\ultralytics_models\yolo26\`；ModelScope 亦有 `pretrained/` 镜像）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **Ultralytics YOLO26**。ultralytics 官方文档明确链接论文 **arXiv:2606.03748**（2026）<br>⚠️ **本仓库既有记录（`AGENTS.md`）把 v26 记为"无独立论文"**；本文件以官方文档指向的论文为准，并保留这条说明 |
| 作者/机构 | Glenn Jocher、Jing Qiu（Ultralytics），2026-01 |
| 官方代码 | `https://github.com/ultralytics/ultralytics`（cfg `26/yolo26*.yaml`） |
| 官方文档 | `https://docs.ultralytics.com/models/yolo26` |
| 本框架实现 | `torchkiln/cfg/models/26/yolo26.yaml`、`yolo26-p2.yaml`、`yolo26-p6.yaml`、`yolo26-depth.yaml`、`yolo26-sem.yaml` 等<br>`torchkiln/nn/modules.py`（`Detect10`/`Detect26`/`C3k2(attn)`/`C2PSA`/`SPPF(add)`）<br>`torchkiln/nn/graph.py`（`end2end`→`Detect10` 路由、`SPPF` 激活修复）<br>`torchkiln/det/loss.py::DetLoss`（`use_one2one` + `reg_max≤1` 的 L1 分支） |
| 移植方式 | **结构逐层复刻 + 权重逐键对齐**：n/s/m/l/x 全 **missing=0 / unexpected=0**（连函数式 dfl 都不存在，因为 v26 去掉了 DFL） |

### 要解决的问题 / 核心创新

1. **原生端到端（NMS-free）**：`Detect10` 双头（one2many + one2one），推理可直接用 one2one **免 NMS**。
2. **移除 DFL**：`reg_max=1`、回归头只 4 个通道、损失退化为 **L1**（导出更简单；官方称是 **CPU 推理最高 +43%** 的主因之一）。
3. **更轻的头**：分类塔 `Seq(DWConv, Conv)` 深度可分离。
4. **训练配方**：**MuSGD**（SGD+Muon）、`ProgLoss`+`STAL`；任务侧：分割语义损失、姿态 RLE、OBB 角度损失。
5. **`-p2` / `-p6` 只有 YAML、无官方权重**（官方明确不发布 `yolo26*-p2.pt` / `*-p6.pt`）。

### 官方指标（ultralytics 文档 Detection(COCO)，640）

| 型号 | mAP50-95 | **mAP50-95 (e2e)** | 参数(M) | FLOPs(B) | T4 TensorRT10(ms) |
|---|---|---|---|---|---|
| YOLO26n | **40.9** | 40.1 | 2.4 | 5.5 | 1.7 |
| YOLO26s | **48.6** | 47.8 | 9.5 | 20.9 | 2.5 |
| YOLO26m | **53.1** | 52.5 | 20.4 | 68.4 | 4.7 |
| YOLO26l | **55.0** | 54.4 | 24.8 | 86.8 | 6.2 |
| YOLO26x | **57.5** | 56.9 | 55.7 | 194.4 | 11.8 |

> 官方同表在 GitHub README 里的 FLOPs 有 ±0.1 的舍入差异（如 n 档 5.4 vs 5.5），属舍入口径；
> 表中数值取官方文档。官方明确：**参数量/FLOPs 是 `fuse()` 且移除未用检测分支后的值**，预训练检查点会更大。

---

## 2. 网络结构

### 2.1 整体框图（`26/yolo26.yaml`，n 档）

```
Input (B,3,640,640)
  │  [0] Conv(3→64,   k3, s2)                → P1 320×320
  ├─ [1] Conv(64→128, k3, s2)                → P2 160×160
  ├─ [2] C3k2(128→256, n=2, c3k=False, e=0.25)
  ├─ [3] Conv(256→256,k3, s2)                → P3  80×80
  ├─ [4] C3k2(256→512, n=2, c3k=False, e=0.25)
  ├─ [5] Conv(512→512,k3, s2)                → P4  40×40
  ├─ [6] C3k2(512→512, n=2, c3k=True)
  ├─ [7] Conv(512→1024,k3,s2)                → P5 20×20
  ├─ [8] C3k2(1024→1024, n=2, c3k=True)
  ├─ [9] SPPF(1024, 5, 3, True)              ← ★ 带 shortcut(add) 的 SPPF + cv1 无激活
  └─[10] C2PSA(1024, n=2)                    ← 位置敏感注意力
       │
       ▼ Neck（PAN）
  [11] Upsample×2  [12] Concat([11],[6])  [13] C3k2(1024→512, n=2, True)
  [14] Upsample×2  [15] Concat([14],[4])  [16] C3k2(512→256,  n=2, True)  → T2 (P3)
  [17] Conv(256→256,k3,s2)[18] Concat([17],[13]) [19] C3k2(512→512, n=2, True) → T3 (P4)
  [20] Conv(512→512,k3,s2)[21] Concat([20],[10])
  [22] C3k2(1024→1024, n=1, True, 0.5, attn=True)  ← ★ 带 PSABlock 的 C3k2（attn=True） → T4 (P5)
       ▼
  [23] Detect10([16,19,22], nl=3, strides=[8,16,32], reg_max=1)
       ├─ one2many: cv2(reg 4) + cv3(cls, DWConv 塔)
       └─ one2one : one2one_cv2 / one2one_cv3
```

**`yolo26-p2.yaml`**：多一条 **P2（stride 4）** 分支，`Detect10` 在 `model.29`、`nl=4`、`strides=[4,8,16,32]`（参数 2.66M）。
**`yolo26-p6.yaml`**：多一条 **P6（stride 64）**，`Detect10` 在 `model.31`、`nl=4`、`strides=[8,16,32,64]`（参数 4.06M）。
**`yolo26-depth.yaml`**：同一 backbone，头换成 `Depth(256)`（单输出，**无 one2one**）。

### 2.2 逐模块说明

| 模块 | 框架实现 | 结构 | 作用 |
|---|---|---|---|
| `C3k2` | `modules.py::C3k2` | `(c1,c2,n,c3k,e,attn,g,shortcut)`；`attn=True` 时内层是 `Seq(Bottleneck, PSABlock(attn_ratio=0.5, num_heads=max(c//64,1)))` | 主特征块（v26 的 P5 用 `attn=True`） |
| `C2PSA` | `modules.py::C2PSA` | `cv1` 切半 → `n×PSABlock` → `cv2` 融合 | 全局注意力 |
| **`SPPF`** | `modules.py::SPPF` | `cv1(1×1, **act=False**) → MaxPool(k5)×3 串联 → cat → `cv2`；**输出 `+ x`（`add=True` 且 `c1==c2`）** | ★ v26 带残差的 SPPF |
| **`Detect10`（`v10Detect`）** | `modules.py::Detect10` | 继承 `Detect26`（reg 塔 `[Conv,Conv,Conv2d]`、cls 塔 `[Seq(DWConv,Conv), Seq(DWConv,Conv), Conv2d]`）+ `one2one_cv2/one2one_cv3`；`bias_init` 与 ultra 一致 | ★ NMS-free 双头 |
| `Detect26` | `modules.py::Detect26` | 无 one2one 的单头版（v11/v12 用） | 新 DWConv 头 |

### 2.3 与上一代（YOLO11）的差异

| 对比对象 | YOLO11 | **YOLO26** | 为什么改 | 效果 |
|---|---|---|---|---|
| 回归 | **DFL（`reg_max=16`）** | **`reg_max=1` + L1** | 去掉分布头 | **导出更简单、CPU 更快（官方最高 +43%）** |
| 头 | 单头 | **双头（one2many + one2one）** | 原生端到端 | 免 NMS |
| SPPF | `SPPF(1024,5)`（无残差、cv1 有 SiLU） | **`SPPF(1024,5,3,True)`**（**add 残差**、cv1 无激活） | — | 前向对齐的关键差异（见 §9） |
| P5 块 | `C3k2(..., c3k=True)` | **`C3k2(..., 0.5, attn=True)`** | 在最后一层加注意力 | AP↑ |
| 优化器 | SGD/Adam | **MuSGD**（SGD+Muon，官方） | 收敛更稳 | 已移植到 `ptcore/muon.py` |
| 损失 | `v8DetectionLoss` | **`E2EDetectLoss`**（o2m topk=10 + o2o topk=1；`reg_max=1` 走 L1） | 配合双头 | 见 §6 |
| 官方 mAP(n) | 39.5 | **40.9** | 综合 | **+1.4** |
| 官方 mAP(x) | 54.7 | **57.5** | 综合 | **+2.8** |

---

## 3. 输出与后处理

### 3.1 输出张量

| 阶段 | 返回 | 形状（每尺度，nc=80、`reg_max=1`） |
|---|---|---|
| **训练** | `(one2many, one2one)` 元组 | 每个 `(B, 4 + 80, H, W)` = `(B, 84, H, W)` |
| **推理/评估** | 单分支列表 | `self.end2end=True` → one2one；否则 one2many |

### 3.2 梯度回传（关键修复）

`Detect10.forward` 在 `self.training` 时对 **one2one 分支的输入特征 `.detach()`**（对齐 ultra `x_detach`）：
**one2one 只训练自己的头，不向 backbone 回传梯度**。
未 detach 时 backbone 梯度 = o2m + o2o，**全模型梯度 maxdiff 40.4**；修复后 **0.00034**（worst `model.1.conv.weight`）。
（该修复同样适用于 **OBB26**；Segment26 的训练路径只返回 one2many，无需 detach。）

### 3.3 解码与损失

```
# reg_max == 1 ⇒ 不做 DFL，reg 直接就是 l,t,r,b
cls = sigmoid(cls_logits)
x1 = (cx - l)*stride ; y1 = (cy - t)*stride
x2 = (cx + r)*stride ; y2 = (cy + b)*stride
```

`DetLoss` 在 `reg_max <= 1` 时启用 **L1 分支**（对齐 ultralytics `BboxLoss` 的无 DFL 路径；
sanity 检查：`loss_dfl = dist_raw.sum() * 0`），分类仍是 BCE。

### 3.4 NMS 与阈值

| 项 | `PostProcess.end2end: false`（推荐评估口径） | `true`（NMS-free） |
|---|---|---|
| NMS | `torchvision.ops.nms` | **不做**（one2one 已去重） |
| conf / iou | 0.001 / 0.7（评估），0.25 / 0.45（推理） | 同（iou 不起作用） |
| max_det / max_nms | 300 / 3000 | 同 |
| strides | `[8,16,32]`（p2 加 4；p6 加 64） | 同 |

> **训练用 `Head.end2end: true`**（否则没有 one2one 分支，`Loss.use_one2one` 会落空）；
> **评估建议 `PostProcess.end2end: false`**，与 ultralytics 重载模型后的口径一致（见 §6）。

---

## 4. 配置与用法

### 4.1 关键配置（对齐官方权重的写法）

```yaml
Architecture:
  model_family: yolo
  task: detect
  algorithm: yolo26
  yaml_file: torchkiln/cfg/models/26/yolo26.yaml     # p2/p6/seg/pose/obb/sem/depth 换对应文件
  scale: n
  in_channels: 3
  Head: {num_classes: 80, reg_max: 1, end2end: true}   # ★ 无 DFL；训练双头
Loss:
  name: DetLoss
  strides: [8, 16, 32]
  topk: 10                # one2many
  alpha: 0.5
  beta: 6.0
  use_one2one: true       # ★ one2one 分支
  one2one_topk: 1
  one2one_gain: 1.0
PostProcess: {name: DetPostProcess, conf_thres: 0.001, iou_thres: 0.7,
              strides: [8, 16, 32], end2end: false}   # ★ 评估走 one2many + NMS
Optimizer: {name: auto}    # 官方推荐 MuSGD；框架 ptcore/muon.py 已移植
```

### 4.2 四条链路

```bash
# 结构自检 / 训练（裸名→ModelScope 自动下载权重）
tkiln check -c configs/yolo/yolo26-det.yml
tkiln train -c configs/yolo/yolo26-det.yml -o Global.pretrained_model=yolo26n

# 评估（one2many + NMS，官方 val 口径）
tkiln val -c configs/yolo/yolo26-det.yml --weights output/yolo26-det/best_accuracy.pth

# 预测（要 NMS-free 直出：-o PostProcess.end2end=true）/ 导出 ONNX
tkiln predict -c configs/yolo/yolo26-det.yml --weights ... --input imgs/ --output out.jpg
tkiln export  -c configs/yolo/yolo26-det.yml --weights ... --save-dir output/onnx --onnx

# 变体：p2（小目标）/ p6（大输入）/ depth / sem / seg / pose / obb
tkiln check -c configs/yolo/yolo26-p2-det.yml
tkiln check -c configs/yolo/yolo26-p6-det.yml
```

---

## 5. 规模与速度

| 型号 | yaml | 参数(M)¹ | 官方参数(M)² | 官方 mAP² | 官方 mAP(e2e)² | 官方 FLOPs(B)² | 官方 .pt³ | head idx / nl / strides |
|---|---|---|---|---|---|---|---|---|
| n | `26/yolo26.yaml` | **2.57** | 2.4 | 40.9 | 40.1 | 5.5 | 5.3 MB | 23 / 3 / 8-16-32 |
| s | 同上 | **10.01** | 9.5 | 48.6 | 47.8 | 20.9 | — | 23 / 3 |
| m | 同上 | **21.90** | 20.4 | 53.1 | 52.5 | 68.4 | — | 23 / 3 |
| l | 同上 | **26.30** | 24.8 | 55.0 | 54.4 | 86.8 | — | 23 / 3 |
| x | 同上 | **58.99** | 55.7 | 57.5 | 56.9 | 194.4 | 113.2 MB | 23 / 3 |
| **p2** | `26/yolo26-p2.yaml` | **2.66**（n） | — | — | — | 9.5（n） | **无官方权重** | 29 / 4 / **4-8-16-32** |
| **p6** | `26/yolo26-p6.yaml` | **4.06**（n） | — | — | — | 6.0（n） | **无官方权重** | 31 / 4 / **8-16-32-64** |
| depth | `26/yolo26-depth.yaml` | **2.62**（nc=1，`c_mid=256`） | — | — | — | — | 有（`yolo26*-depth`） | — |

¹ 本框架实测（`nc=80`、`reg_max=1`、`Head.end2end=true`，**含 one2one 头**）。
² 官方文档表（**`fuse()` + 移除未用分支后**的口径，故比实测小 0.2~3.3M）。
³ 本机官方权重磁盘体积（fp16）。

> ⚠️ **本框架未集成 FLOPs 计数**；FLOPs 一列是官方值（p2 9.5 / p6 6.0 取自 YAML 注释）。

---

---

---

---

---

### 📊 FLOPs（实测）

| 项 | 值 |
|---|---|
| **FLOPs** | **6.08 GFLOPs** |
| **MACs** | **3.04 GMACs** |
| 参数量 | **2.572 M** |
| 输入规格 | `nc=80, 640x640（n 档）` |
| 测量工具 | `torch.utils.flop_counter.FlopCounterMode`（PyTorch 内置） |
| 复现脚本 | `_downloads/flops_measure*.py` |

> **口径**：`FLOPs` 是乘加各计 1 次（×2），**与 ultralytics 官方表的 GFLOPs 同口径**
> （已由 yolo11/v8 十个模型逐个吻合验证，见 [`_FLOPS.md`](_FLOPS.md)）；
> `MACs = FLOPs / 2`。
> ⚠️ `FlopCounterMode` **不计自定义算子**（NMS / probiou / iSTFT 等后处理）⇒
> 此处是**网络主干**的 FLOPs。
## 7. 选型建议

| 场景 | 推荐档位 | 理由 |
|---|---|---|
| **新项目默认** | **n / s** | 2.57M / 10.01M，官方 40.9 / 48.6，且 **NMS-free** |
| 通用高精度 / 离线最强 | **m / l / x** | 53.1 / 55.0 / 57.5 |
| **小目标密集**（遥感/无人机/瑕疵） | `yolo26-p2-det.yml` | +P2/stride 4；⚠️ **无官方权重**，需自训 |
| **大输入 / 大目标** | `yolo26-p6-det.yml` | +P6/stride 64（1280 输入）；同样无官方权重 |
| **端侧 CPU 部署** | n + `PostProcess.end2end=true` | 去 DFL + NMS-free，官方称 CPU ONNX 最高 +43% |
| 深度 / 语义分割 / 分割 / 姿态 / OBB / 分类 | 对应 `yolo26-*.yml` | 头都含 one2one；OBB26 已同步 detach |
| 训练加速 | `Optimizer.name: auto`（迭代 >10000 自动 MuSGD） | 框架 10 epoch 实测比 SGD 好 **+0.02 mAP** |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs YOLO11** | 官方 mAP(n) 39.5 → **40.9**、mAP(x) 54.7 → **57.5**；关键差异是**去 DFL + 双头** |
| **vs YOLOv10** | v10 首创 NMS-free 双头；v26 继承双头并**去掉 DFL**、换更轻的头、配 MuSGD；
两者在本框架**共用 `Detect10` + `E2EDetectLoss` 代码路径**（v10 用 `reg_max=16`，v26 用 1） |
| **vs YOLO12 / RT-DETR** | v12 是注意力路线（40.6）、RT-DETR 是 Transformer；v26 走"轻头 + 端到端"，官方更快 |
| **本仓库定位** | v26 **对齐度最高**（`missing=0 / unexpected=0`，连 dfl 差异都没有）；
其 `detach`、`SPPF(add)`、评估路径三处修复是**其它家族的通用经验来源** |

---

## 9. 已知问题 / 注意事项

- ⚠️ **评估必须走 one2many + NMS（本框架已修）**：`Detect10.forward` 早期 eval 时**无条件返回 one2one**，
  而 ultralytics 重载模型后 `end2end=False`。同一 `best.pt`：框架 end2end **0.7833** vs ultra **0.8055**（差 0.022）；
  改用 one2many 后 **0.8035**（差 0.002）。修法：① eval 分支改 `return one2one if self.end2end else one2many`；
  ② `evaluate()` 评估前把 `end2end` 临时设为 `PostProcess.end2end`。
- ⚠️ **one2one 输入必须 `.detach()`（最关键的一条）**：见 §3.2（梯度 40.4 → 0.00034）。**OBB26 已同步**；
  Segment26 训练只回传 one2many，无需处理。
- ⚠️ **SPPF 的两个 v26 专属语义（已修）**：① `cv1` **无激活**；② `args ≥ 4` 时输出 **`+ x` 残差**（且需 `c1==c2`）。
  修复前 L9(SPPF) 起 maxdiff 4.43；修复后 <1e-3。**对照**：v8/v10/v11/v12 的旧式 SPPF（`args ≤ 3`）需**恢复 `cv1` 的 SiLU**。
- ⚠️ **v26 没有 DFL**：`reg_max` 必须 **1**（`DetLoss` 自动切 L1 分支）；误设 16 会因头结构不同而缺键。
- ⚠️ **`Architecture.Head.end2end=true`（训练双头）与 `PostProcess.end2end=false`（评估 one2many+NMS）是推荐组合**，别混。
- ⚠️ **`-p2` / `-p6` 没有官方权重也没有 COCO mAP**；§5 里的 p2/p6 只有参数/FLOPs 参考。
- ⚠️ **E2E 训练波动大**：6-epoch 终值差 0.013~0.026（不同记录），逐 epoch 会互相反超；
  评估侧已对齐 0.002，结论以**同权重推理**为准（通用验收第 4 条）。
- ⚠️ **MuSGD**：`ptcore/muon.py` 1:1 移植 ultra `optim/Muon`（`auto` 且迭代 >10000 时启用；
  `ndim∈{2,4}` 走 Muon，检测头 `cv3/one2one_cv3` 参数 `lr*3`）。小迭代数时会落回 AdamW/SGD，别用它做"快速对比"。
- ⚠️ 通用注意事项（评估 fp32、`torchvision` NMS、`accumulate` 读取位置、BN eps、`imgsz=1024` 用 batch=4、
  对比必须同评估器/同 val 清单）见 `yolo11.md` §9 与 `docs/models/infra/`。
- ⚠️ 本框架**未集成 FLOPs 统计**，§5 的 FLOPs 为官方值。
