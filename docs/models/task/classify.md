# 图像分类（Classify）

> **定位**：YOLO 的**分类头** —— 复用检测 backbone（C3k2/C2PSA），末尾用
> `Conv → 1280 → AdaptiveAvgPool → Linear(nc)` 输出整图类别；也可一键切换为**多标签**。
> **任务**：`classify`（别名 `cls`） · **头**：`Classify`
> **权重**：`yolo11{n,s,m,l,x}-cls.pth`、`yolov8*-cls.pth`、`yolo26*-cls.pth`（ImageNet 1000 类）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | Ultralytics 的分类头**无独立论文**；数据集 **ImageNet**（Deng et al., *ImageNet: A Large-Scale Hierarchical Image Database*, CVPR 2009） |
| 机构 | Ultralytics |
| 官方代码 | https://github.com/ultralytics/ultralytics |
| 官方权重 | `yolo11n-cls.pt` / `yolov8n-cls.pt` / `yolo26n-cls.pt` |
| 本框架实现 | 头：`torchkiln/nn/modules.py::Classify`<br>结构：`torchkiln/cfg/models/{8,11,26}/*-cls.yaml`<br>损失/后处理/指标：`torchkiln/tasks/_cls.py`<br>任务：`torchkiln/tasks/classify.py` · 数据：`torchkiln/data/cls.py::ClsDataset` |
| 移植方式 | 权重逐键对齐（15 个权重全 `missing=0/unexpected=0`）+ 前向/损失/梯度单步对齐 |

### 要解决的问题

- 需要**整图分类**（缺陷类型、场景、材质、安全着装）时，检测头是多余的。
- Ultralytics 的做法：**共用同一个 backbone**，只把 neck/head 换成一个
  `Conv(→1280)` + 全局池化 + 一层线性分类器，让分类/检测/分割共享预训练，迁移方便。
- 关键工程细节：`Classify` 头**不做重复堆叠**（`C3k2` 那种 `n>1` 复制逻辑要跳过它）。

---

## 2. 网络结构

### 2.1 整体框图（yolo11n-cls）

```
Input (B,3,224,224)
  │
  ├─[0] Conv(3→32, k3,s2)      ┐
  ├─[1] Conv(32→64, k3,s2)     │  backbone
  ├─[2] C3k2(64→64,  n=1)      │  （与 detect 相同，
  ├─[3] Conv(64→64, k3,s2)     │    但无 SPPF、
  ├─[4] C3k2(64→128, n=2)      │    末尾直接进 C2PSA）
  ├─[5] Conv(128→128,k3,s2)    │
  ├─[6] C3k2(128→128,n=2)      │
  ├─[7] Conv(128→256,k3,s2)    │
  ├─[8] C3k2(256→256,n=2)      │
  └─[9] C2PSA(256→256, n=2)    ┘
       │
       ▼ Head（不等同 detect 的 3 尺度头）
  Classify: Conv(c1→1280,k1) → AdaptiveAvgPool2d(1) → Dropout → Linear(1280, nc)
       输出 (B, nc)  logits
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | ultra 名字 |
|---|---|---|---|---|
| backbone | 与 yolo11 detect 同（Conv/C3k2/C2PSA） | `(B,3,H,W)→(B,256,H/32,W/32)` | 特征提取 | 同 |
| **`Classify`** | `Conv(c1→hidden=1280,k=1)` → `AdaptiveAvgPool2d(1)` → `Dropout(p)` → `Linear(1280,nc)` | `(B,c1,h,w)→(B,nc)` | ★ 分类头 | `Classify` |
| `hidden` | 默认 **1280**（可配 `Head.hidden` 或 yaml 第 2 参） | | 分类前瓶颈 | |

### 2.3 与 detect 头的差异

| 项 | Detect | **Classify** |
|---|---|---|
| 输入 | 3 尺度特征（P3/P4/P5） | **单尺度**（最后一层特征） |
| 空间结构 | 网格上逐位置预测 | **全局平均池化** |
| 输出 | `(B, 4*reg_max+nc, N)` | **`(B, nc)`** |
| 是否堆叠 n 份 | 否 | **否**（`parse_model` 显式跳过 `Classify`） |
| loss | box+cls+dfl | **CrossEntropy**（或 BCE 多标签） |

---

## 3. 输出与后处理（★ 重点）

### 3.1 单标签（默认）

```
logits = model(x)                       # (B, nc)
probs  = softmax(logits, dim=1)         # (B, nc)
topk   = probs.topk(k=5, dim=1)         # 返回 [(cls_id, prob), ...]
metric = AccTop1 / AccTop5              # ClsMetric
loss   = CrossEntropyLoss(logits, label)  # label_smoothing 可配，默认 0
```

### 3.2 多标签（一键开关）

配置里写 `Loss.multi_label: true`（或 `-o Loss.multi_label=true`）即自动切换：

```
scores = sigmoid(logits)                # 每类独立
hit    = [i for i where scores[i] >= PostProcess.threshold]   # 默认 0.5
loss   = BCEWithLogits(loss)            # MultiLabelLoss
metric = AttrMetric → mA / mAP          # main_indicator 自动改为 mAP
```

> `build_cls_loss/metric/post_process`（`tasks/_cls.py`）会根据 `multi_label` 自动选实现，
> 无需另写 name；`tasks/classify.py::build_loss` 还会把 `Metric.main_indicator` 从 `acc` 改成 `mAP`。

### 3.3 阈值 / NMS

| 项 | 值 |
|---|---|
| NMS | 无（分类不涉及框） |
| 单标签 | 取 argmax / topk |
| 多标签 | `PostProcess.threshold`（默认 0.5） |

---

## 4. 配置与用法

### 4.1 最小配置（`configs/yolo/yolo11-cls.yml`）

```yaml
Architecture:
  task: classify
  yaml_file: torchkiln/cfg/models/11/yolo11-cls.yaml
  scale: n
  Head: {num_classes: 3, reg_max: 1}
Loss:  {name: CrossEntropy}                # 多标签：加 -o Loss.multi_label=true
Metric: {name: ClsMetric, main_indicator: acc}
PostProcess: {name: ClsPostProcess}
Train:
  dataset: {name: ClsDataset, data_dir: datasets/cls_demo, label_file_list: [.../train.txt],
            transform: {image_size: 224}}   # int=方图；[h,w] 非方（边长 %32==0）
  loader:  {batch_size_per_card: 32, num_workers: 4}
Eval:
  dataset: {..., transform: {image_size: 224}}   # 必须与 Train 一致
```

### 4.2 四条链路

```bash
tkiln train   -c configs/yolo/yolo11-cls.yml -o Global.pretrained_model=yolo11n-cls
tkiln val     -c configs/yolo/yolo11-cls.yml --weights output/yolo11-cls/best_accuracy.pth
tkiln predict -c configs/yolo/yolo11-cls.yml --weights ... --input imgs/
tkiln export  -c configs/yolo/yolo11-cls.yml --weights ... --onnx
tkiln check   -c configs/yolo/yolo26-cls.yml
```

---

## 5. 规模与速度

| 档位 | 参数(M)¹ | 权重体积 | ImageNet top-1 |
|---|---|---|---|
| yolo11n-cls | 1.63 | ~3.3 MB | 未统计 |
| yolo11s-cls | 5.55 | ~11 MB | 未统计 |
| yolo11m-cls | 10.46 | ~21 MB | 未统计 |
| yolo11l-cls | 12.94 | ~26 MB | 未统计 |
| yolo11x-cls | 28.46 | ~57 MB | 未统计 |
| yolo26n-cls | 2.81 | — | 未统计 |

¹ 取自 `torchkiln/cfg/models/11/yolo11-cls.yaml`（n=1,633,584）。**FLOPs / 本机耗时 / 官方 top-1 未在本框架侧记录，不编造。**

---

## 6. 公开指标

### 6.1 权重加载对齐（**全部 15 个权重**）

对 `\\tsclient\D\项目资料\ultralytics_models\{yolov8,yolo11,yolo26}\*-cls.pt`（n/s/m/l/x 共 15 个）：

| 项 | 结果 |
|---|---|
| `missing` / `unexpected` | **全部 0 / 0** |
| 同输入 (224) 推理 softmax maxdiff | **≤ 0.0000007**（几乎逐位一致） |
| top-1 | **全部一致** |

### 6.2 前向逐层（同权重同输入）

| 项 | maxdiff |
|---|---|
| backbone 各层（Conv/C3k2/C2PSA） | **≈ 0.00000** |
| raw logits | 0.000004 |
| softmax | **0.000000** |
| top-1 | 一致（如 885） |

### 6.3 训练单步（同权重、同 `(2,3,224,224)` 输入、同标签 `[3,885]`）

| 项 | 结果 |
|---|---|
| loss diff（对齐 `v8ClassificationLoss` = `F.cross_entropy`） | **≤ 5.7e-6**（yolo11n=0） |
| 每模型梯度 | **全部层匹配（无漏层）** |
| 梯度 maxdiff | **≤ 0.00017**（最大为首层 `model.0.conv.weight`，cuDNN 反向算子级） |

### 6.4 评估冒烟

`configs/_parity/pkg_cls_demo.yml`（cls_demo 3 类，imgsz=224）评估跑通（top1/top5 正常输出）。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 边缘实时分类 | **yolo11n-cls / yolo26n-cls** | 1.6M，共用检测 backbone |
| 通用 ImageNet 迁移 | **11m / 11l** | |
| 高精度离线 | **11x-cls** | 28M |
| **多标签**（属性/缺陷可共存） | 任意档 + `Loss.multi_label=true` | BCE + 阈值 |
| 行人/车辆属性（专业模型） | 见 `docs/models/task/attribute.md` | PP-LCNet + 26/19 属性 |
| 时序/视频分类 | 见 `video_cls.md` | TSM |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs detect** | 单尺度 + 全局池化，输出 `(B,nc)`；loss 换 CE |
| **vs PP-LCNet（attribute）** | 后者专用轻量 CPU backbone + 多标签头；本头复用 YOLO backbone |
| **vs ResNet/EfficientNet** | 本头与 YOLO backbone 共享权重，检测/分类可互迁 |
| **单标签 vs 多标签** | 单标签 softmax+CE；多标签 sigmoid+BCE+mAP，一个开关切换 |
| **v8/v11/v26** | backbone 差异（va: C2f；v11: C3k2+C2PSA；v26: C3k2(attn)+C2PSA）；头同构 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **BN eps 推理对齐（影响所有任务评估）**：ultra `initialize_weights` 把 BN eps 设为
  **1e-3**（仅训练期），保存的权重不含 eps；ultra **推理**用构造默认 **1e-5**。
  本框架 `_set_bn_ultralytics` 永久设 1e-3，导致推理层 0 起就有 0.005 meanabs 差异。
  已在 `ptcore/trainers/base.py::evaluate()` 评估时**临时恢复 BN eps=1e-5**（评估完还原）。
  **仅 yolo_cls 用 1e-5；det/seg/pose/obb 保留 1e-3（强改 1e-5 会崩）。**
- ⚠️ **`Classify` 不参与 n>1 堆叠**：`parse_model` 里 `not isinstance(layer, Classify)` 显式跳过，
  否则 yaml 里的重复数会被误用。
- ⚠️ **多标签默认不带 `label_ratio` 加权**（`build_cls_loss` 设 `size_sum=False, weight_ratio=False`）；
  需要类平衡时显式开（attribute 任务默认开）。
- ⚠️ **Train/Eval `image_size` 必须一致**。
- ⚠️ **评估 fp32**：`_evaluate_loop` 强制 `autocast(enabled=False)`（fp16 会失真）。
- ⚠️ 对比 ultra 必须**同权重 + 同一评估器**。
