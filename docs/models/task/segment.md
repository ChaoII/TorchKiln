# 实例分割（Segment / SegmentU / Segment26）

> **定位**：在检测头之上加一条 **mask 系数塔（`cv4`）** 与一个 **原型分支（`Proto`）**，
> 用「掩码系数 × 原型」生成每实例掩码（YOLACT 范式）。
> **任务**：`segment` · **头**：`Segment`（旧式 Conv，DFL）/ `SegmentU`（新 DWConv）/ `Segment26`（end2end）
> **权重**：`yolo11{n,s,m,l,x}-seg.pth`、`yolov8*-seg.pth`、`yolo26*-seg.pth`
> （ModelScope `ChaoII0987/TorchKiln → pretrained/` 或 `\\tsclient\D\项目资料\ultralytics_models`）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **实例分割头无独立论文**。「掩码系数 × 原型」范式来自 **YOLACT**（*YOLACT: Real-time Instance Segmentation*，arXiv:1904.02689） |
| 机构 | Ultralytics（工程化 YOLOv5/v8/v11/v26 的 seg 头） |
| 官方代码 | https://github.com/ultralytics/ultralytics |
| 官方权重 | `yolo11n-seg.pt` / `yolov8n-seg.pt` / `yolo26n-seg.pt` 等 |
| 本框架实现 | 头：`torchkiln/nn/modules.py::Segment`(`SegmentU`/`Segment26`) / `Proto`(`Proto26`)<br>结构：`torchkiln/cfg/models/{8,11,26}/*-seg.yaml`<br>损失/指标：`torchkiln/seg.py`<br>任务适配：`torchkiln/tasks/segment.py` |
| 移植方式 | YAML 逐层复刻 + 权重逐键对齐（`missing=0`；旧家族仅差函数式 `dfl.conv.weight`）+ 损失/梯度单步对齐 |

### 要解决的问题

- 检测只给矩形框，**像素级轮廓**（人、车、零件、包装）需要实例掩码。
- 直接 decoder 出全图掩码代价高。YOLACT 的思路：**全图共享一组原型 `proto`（nm=32 张基）**，
  每个实例只回归 **`nm` 个系数**，`mask = coeff @ proto` —— 参数与算力开销极小。
- Ultralytics 的关键工程化：**只在 GT 框内**统计 mask BCE，并按实例归一化，否则整图平均会把损失稀释掉。

---

## 2. 网络结构

### 2.1 整体框图（yolo11n-seg / yolo26n-seg）

```
Input (B,3,H,W)
  │ backbone（与 detect 完全相同：Conv/C3k2[/SPPF]/C2PSA）→ P3(8),P4(16),P5(32)
  │ neck（PAN，自顶向下+自底向上）→ T1(64),T2(128),T3(256)
  ▼ Head
  ┌─ 每尺度（nl=3）───────────────────────────────────────────┐
  │  cv2 回归塔 → 4*reg_max  (reg_max=16: DFL；v26: reg_max=1) │
  │  cv3 分类塔 → nc         (旧 family: Conv×3；新 family: DWConv) │
  │  cv4 掩码塔 → nm(32)     ← ★ seg 独有                      │
  └───────────────────────────────────────────────────────────┘
  └─ Proto 分支（仅 P3 输入，stride 8）→ 上采样 ×2 → stride 4 → (B,32,H/4,W/4)
      旧: cv1 Conv3 → ConvTranspose2(×2) → cv2 Conv3 → cv3 Conv1
      v26: + feat_refine(P4,P5) + feat_fuse；训练时另有 semseg 语义头

  每尺度输出 = cat([cv2, cv3, cv4], 1)
  推理掩码  = sigmoid/einsum(coeff(nm=32), proto(nm=32)) → 上采样 → 阈值 → 按框裁剪
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | ultra 名字 |
|---|---|---|---|---|
| `cv2` 回归塔 | `Conv(c,k3)→Conv(c,k3)→Conv2d(1)` | 特征→`4*reg_max` | 框回归（DFL 或 L1） | `cv2` |
| `cv3` 分类塔 | 旧：`Conv×3`；新：`DWConv+Conv` ×2 + `Conv2d` | →`nc` | 分类 | `cv3` |
| **`cv4` 掩码塔** | `Conv(x,c4,3)→Conv(c4,c4,3)→Conv2d(c4,nm,1)`，`c4=max(ch[0]//4, nm)` | →`nm=32` | ★ 掩码系数 | `cv4` |
| **`Proto`** | `Conv(k3)→ConvTranspose2d(k2,s2)`（转置卷积上采样 ×2）→`Conv(k3)`→`Conv(k1)` | P3→`(32,H/4,W/4)` | ★ 共享掩码基 | `Proto` |
| `Proto26` | `Proto` + `feat_refine`(P4/P5 1×1 + 上采样相加) + `feat_fuse` + `semseg`（训练态语义头） | 多尺度→原型（+语义 logits） | v26 原型融合 + 辅助监督 | `Proto26` |
| `Segment26` | `cv2/cv3/cv4` + **`one2one_cv2/cv3/cv4`** 副本，`reg_max=1`、`end2end=True` | 训练 `(o2m,[o2o],proto)` | 端到端分割 | `Segment26` |

> ⚠️ **家族路由（关键）**：`REGISTRY["Segment"]→SegmentU`（新 DWConv 头别名），
> `build_from_arch` 对旧家族（v8/v9/v5/v3 的 yaml）设 `_legacy=True`，
> `parse_model` 把 `Segment` 路由回旧式 Conv 头（`modules.py::Segment`），
> 只有 yolo11/12/26 用新 DWConv 头。见 `docs/models/detect/yolo11.md` §9。

### 2.3 与 detect 头的差异（★ 核心）

| 项 | Detect | **Segment** | 效果 |
|---|---|---|---|
| 输出通道 | `4*reg_max + nc` | **`4*reg_max + nc + nm(32)`** | 多出掩码系数 |
| 多出的塔 | 无 | **`cv4`（掩码系数）** | 每实例 32 维表示 |
| 多出的分支 | 无 | **`Proto`（stride 4 原型）** | 整图共享 32 张基 |
| 损失 | box+cls+dfl | **+ mask BCE（框内归一化）** | 学掩码 |
| 指标 | box mAP | **box mAP + mask mAP** | 双指标 |
| 后处理 | NMS→框 | **NMS→框→`coeff@proto`→上采样→阈值→裁剪** | 出掩码 |

---

## 3. 输出与后处理（★ 重点）

### 3.1 输出张量

| 阶段 | 形状 | 含义 |
|---|---|---|
| 训练（`Segment26` end2end） | `(one2many, one2one, proto)`；`proto` 为 `(p, sem)` 元组 | 三分支 |
| 训练（普通） | `{"feats": [3×`(B, 4*reg_max+nc+nm, H, W)`], "protos": (B,32,H/4,W/4)}` | |
| 推理（`Segment26` end2end=True） | 用 `one2one_*` 分支，**无需 NMS** | |
| 推理（end2end=False） | 用 `one2many` 分支 + NMS | |

### 3.2 掩码解码公式（`torchkiln/seg.py::SegPostProcess`）

```
# 1) 拆头：[reg(4*reg_max), cls(nc), coeff(nm)]
distri, scores, coeffs = split_head(feats, nc, reg_max, "seg")
boxes  = dist2bbox(distri, anchor_points) * stride_tensor      # 像素 xyxy
scores = scores.sigmoid(); conf, labels = scores.max(-1)       # 多标签取最大类

# 2) 逐类 NMS（走 torchvision C++ NMS）
kept = nms(boxes[labels==c], conf[labels==c], iou_thres)       # 每类分别 NMS

# 3) 掩码 = coeff @ proto（logits）
m      = bmm(coeff[kept].unsqueeze(1), proto.unsqueeze(0)).view(-1, h, w)
logits = F.interpolate(m, size=(imgsz_h, imgsz_w), mode="bilinear")  # 上采样到原图
bin_m  = (logits > 0.0).float()          # 在 logits(概率对数) 上阈值 0
masks  = crop_by_box(bin_m, boxes[kept]) # 按预测框裁剪
```

> ⚠️ **阈值口径**：ultralytics `process_mask(upsample=True)` 是**先双线性上采样 logits，
> 再阈值 `>0` 二值化，最后按框裁剪**。掩码 BCE 用的是 `BCEWithLogits`，所以这里用 `>0` 而非 `>0.5`。

### 3.3 NMS / 阈值

| 项 | 值 | 说明 |
|---|---|---|
| NMS | `torchvision.ops.nms`（C++） | 原 Python 循环 10s/图 → **2.5s/50 图** |
| `iou_thres` | 0.7（评估）/ 0.45（推理） | |
| `conf_thres` | 0.001（评估，ultra 口径）/ 0.25（推理） | |
| `max_det` | 300 | |
| `mask_thres` | 0.5（外部保存） | 内部二值用 logits>0 等价 |

---

## 4. 配置与用法

### 4.1 最小配置（yolo26-seg，`configs/yolo/yolo26-seg.yml`）

```yaml
Architecture:
  task: segment
  yaml_file: torchkiln/cfg/models/26/yolo26-seg.yaml
  scale: n
  Head: {num_classes: 2, reg_max: 1, end2end: true}
Loss:  {name: SegLoss, topk: 10, alpha: 0.5, beta: 6.0, cls_gain: 0.5, box_gain: 7.5}
Metric: {name: SegMetric, main_indicator: mask_mAP50-95}
PostProcess: {name: SegPostProcess, conf_thres: 0.001, iou_thres: 0.7, end2end: true}
Train:
  dataset: {name: SegDataset, data_dir: datasets/seg_demo, label_file_list: [.../train.txt],
            transform: {image_size: 256, mask_stride: 4}}
```
> ⚠️ v11/v8 用 `reg_max: 16` + `Segment`（DFL）；v26 用 `reg_max: 1` + `end2end: true` + `Segment26`。

### 4.2 四条链路

```bash
# 训练（从官方预训练微调）
tkiln train -c configs/yolo/yolo11-seg.yml -o Global.pretrained_model=yolo11n-seg

# 评估（掩码 mAP）
tkiln val     -c configs/yolo/yolo11-seg.yml --weights output/yolo11-seg/best_accuracy.pth

# 预测（出掩码）
tkiln predict -c configs/yolo/yolo11-seg.yml --weights ... --input imgs/

# 导出 ONNX / 结构自检
tkiln export  -c configs/yolo/yolo11-seg.yml --weights ... --onnx
tkiln check   -c configs/yolo/yolo26-seg.yml
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ | 权重体积 | 备注 |
|---|---|---|---|
| yolo11n-seg | 2.88 | ~5.8 MB | 头与 v8n-seg 同构 |
| yolo11s-seg | 10.11 | ~20 MB | |
| yolo11m-seg | 22.42 | ~45 MB | |
| yolo11l-seg | 27.68 | ~55 MB | |
| yolo11x-seg | 62.14 | ~124 MB | |
| yolov8n-seg | 3.10 | — | 参数差 ~0.3% |
| yolo26n-seg | 3.13 | — | reg_max=1 + Proto26 |

¹ 取自 `torchkiln/cfg/models/**/*-seg.yaml` 注释里的参数量（n=2,876,848 等）。
**FLOPs 未在本框架侧统计**，本机推理耗时/显存**未统计**（不作编造）。

---

## 6. 公开指标

### 6.1 权重加载对齐（AGENTS 记录）

| 家族 | missing | unexpected | 说明 |
|---|---|---|---|
| yolov8-seg n/s/m/l/x | **0** | 1 | 仅函数式 `dfl.conv.weight` |
| yolo11-seg n/s/m/l/x | **0** | 1 | 仅函数式 dfl；参数差 ~0.2% |
| yolo26-seg n/s/m/l/x | **0** | **0** | `reg_max=1` + 新 `Segment26`/`Proto26` |

> 验证时需设 `Architecture.scale` 对应档位、`num_classes=80`（对齐 COCO 预训练）。

### 6.2 同权重推理（`package-seg`，ultra 微调 30ep 的 `best.pt`）

| 指标 | **本框架** | ultralytics | 差异 |
|---|---|---|---|
| box_mAP50 | 0.929 | 0.922 | |
| box_mAP50-95 | 0.840 | 0.845 | ~0.005（算子级） |
| mask_mAP50 | 0.9228 | 0.9235 | |
| **mask_mAP50-95** | **0.8206** | **0.8214** | **0.0008** |

⇒ 同一权重下 **mask mAP 几乎相等**，证明整个 seg 解码/掩码度量链路与 ultra 一致。

### 6.3 端到端训练（缩放集 100/50，50 轮，同一框架评估器）

| 模型 | 本框架 mask_mAP50-95 | ultralytics | 结论 |
|---|---|---|---|
| yolov8n-seg | **0.650** | 0.648 | 系统性差距已消除 |
| yolo11n-seg | **0.618** | 0.583 | 反超 |
| yolo26n-seg | **0.636** | 0.642 | 持平 |

### 6.4 单步 loss 对齐

同权重同输入下，seg loss 与 ultra 差 **~1.5%**（mask 分量 1.7%，box/dfl <0.5%）。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 通用实例分割（COCO 风格） | **yolo11n-seg / yolo26n-seg** | 权重齐全、头部对齐 |
| 边缘设备实时分割 | **n + reg_max=16** | 参数 2.9M，掩码只多一个 32 通道塔 |
| NMS-free / 低延迟 | **yolo26n-seg（end2end）** | one2one 头，推理无需 NMS |
| 高精度离线 | **11x-seg / 26x-seg** | |
| 只要语义类别不回实例 | 见 `docs/models/task/semantic.md` | SemanticSegment 更轻 |
| 只要矩形框 | 见 `docs/models/detect/yolo11.md` | 更省 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs detect** | 多 `cv4`(32) 塔 + `Proto`（stride 4），掩码由 `coeff@proto` 生成 |
| **vs YOLACT** | 同一「系数 × 原型」范式；ultra 把原型放在 stride 4 并只在框内算 BCE |
| **vs Mask R-CNN** | 两阶段 + RoIAlign，重且慢；本头单阶段、无 RoI，实时 |
| **vs semantic** | segment 出**每实例**掩码并分类；semantic 出**每像素**类别图、不区分实例 |
| **v8 vs v11 vs v26** | v8/v11 用 Conv 头 + `reg_max=16`(DFL)；v26 用 DWConv 头 + `reg_max=1`(L1) + `Proto26`(多尺度融合+语义辅助) + end2end |

---

## 9. 已知问题 / 注意事项

- ⚠️ **评估必须 fp32**：`ptcore/trainers/base.py::_evaluate_loop` 强制
  `torch.autocast(enabled=False)`。此前评估走 fp16 时 yolo26 seg 的 `reg_max=1` 框解码
  与 proto einsum 失真 —— 同权重 mask_mAP50-95 **0.5596(fp16) vs 0.6413(fp32)**。**影响所有任务**。
- ⚠️ **mosaic 裁剪中心**：`data/augment.py::mosaic4` 必须调用
  `_random_crop_mosaic(..., center=(s,s))`（画布中心，对齐 ultra `RandomPerspective`）。
  改前 v8n 0.595，改后 **0.650 = ultra 0.648**。（试过"不裁剪改缩放"更差 0.47，勿重蹈。）
- ⚠️ **SegLoss 的 DFL/L1**：`tgt` 不可 clamp 到 `reg_max-1`，只 clamp `bbox2dist` 的 ltrb 距离；
  `reg_max<=1` 时必须走 **L1 loss**（对齐 ultra `BboxLoss` 无 DFL 分支）。
- ⚠️ **Segment26 one2one bias 必须初始化**（对齐 ultra `bias_init`）：不初始化时 one2one cls 初始
  `p~0.5` → 训练初期 cls loss 爆炸（1495）→ 崩盘。且需**重生成 nc=1 权重**。
- ⚠️ **Segment26 的 one2one 输入特征要 `.detach()`**（不回流骨干），否则骨干梯度 = o2m+o2o，与 ultra 不符。
- ⚠️ **proto 语义分支**：one2one 的 `proto` 必须 detach。
- ⚠️ **e2e 增益调度**：o2m 从 0.8 线性降到 0.1，每 epoch 末调用 `loss.update()`（`base.py::_train_one_epoch`）。
- ⚠️ **`evaluate()` 的 BN eps**：det/seg/pose/obb 保留训练值 **1e-3**（强改 1e-5 会崩），
  只有 yolo_cls 用 1e-5。
- ⚠️ **端到端头评估交换**：`evaluate()` 评估前把含 `one2one_cv2` 的头的 `end2end` 临时置为
  `PostProcess.end2end`（默认 False → 走 one2many+NMS，对齐 ultra 重载模型）。
- ⚠️ 对比必须**同权重 + 同一评估器**（ultra 自报用 `rect=True` 会抬高数值）。
