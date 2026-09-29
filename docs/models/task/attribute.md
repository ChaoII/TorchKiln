# 多标签属性识别（PP-LCNet + MultiLabelHead）

> **定位**：**行人/车辆多属性**识别 —— 轻量 CPU 友好的 `PP-LCNet_x1_0` backbone +
> 1280 维瓶颈 + 一层线性输出，**多标签**（每属性独立 sigmoid + 阈值）。
> **任务**：`attribute`（别名 `attr`） · **模型**：`AttributeNet`（`PPLCNetX1_0` + `MultiLabelHead`）
> **权重**：`PP-LCNet_x1_0_pedestrian_attribute.pth`（26 属性）、
> `PP-LCNet_x1_0_vehicle_attribute.pth`（19 属性），ModelScope `ChaoII0987/TorchKiln → pretrained/`

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **PP-LCNet: A Lightweight CPU Convolutional Neural Network**（arXiv:2109.15099，Cui et al.） |
| 机构 | 百度 PaddleClas / PaddleX |
| 官方代码 | https://github.com/PaddlePaddle/PaddleClas（`ppcls/arch/backbone/legendary_models/pp_lcnet.py`） |
| 官方权重 | PaddleX「行人属性 / 车辆属性」模型（已转为本框架 `.pth` 并托管 ModelScope） |
| 本框架实现 | backbone：`torchkiln/nn/attribute.py::PPLCNetX1_0`（移植 PaddleClas NET_CONFIG x1_0）<br>头：`MultiLabelHead`<br>数据集/损失/指标/后处理：`torchkiln/tasks/attribute.py`<br>配置：`configs/attr/{pedestrian,vehicle}_attribute.yml` |
| 移植方式 | **结构逐层照搬 PaddleClas**（`blocks2..6` + `DepthwiseSeparable` + SE）；权重加载见 §6 |

### 要解决的问题

- 安防/交通需要细粒度属性：行人（性别、年龄、背包、帽子、眼镜……26 项）、
  车辆（颜色、车型、朝向、遮挡……19 项）。
- 这些属性**可同时成立**（既戴帽子又戴眼镜），是**多标签**问题，不能用 softmax 归一化。
- **PP-LCNet** 为 CPU 推理优化：大量使用 depthwise separable + `hardswish` + 末段 SE，
  精度接近大模型但算力极低。

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,3,256,192) 或 (B,3,192,256)
  │
  ▼ PPLCNetX1_0 backbone
  │   conv1 (3→16, k3 s2, hardswish)
  │   blocks2: [3,16,32,1]                                      
  │   blocks3: [3,32,64,2], [3,64,64,1]
  │   blocks4: [3,64,128,2], [3,128,128,1]
  │   blocks5: [3,128,256,2], 5×[5,256,256,1]
  │   blocks6: [5,256,512,2,SE], [5,512,512,1,SE]     ← 仅 blocks5/6 用 SE
  │   → (B,512,h,w)
  ▼ MultiLabelHead
  │   AdaptiveAvgPool2d(1) → Conv2d(512→1280, k1) → Hardswish → Dropout(p) → Linear(1280, num_classes)
  输出 (B, num_classes) logits   # 每属性独立，接 sigmoid
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 | PaddleClas 名字 |
|---|---|---|---|---|
| `conv1` | `ConvBNLayer(3→16,k3,s2,hardswish)` | `(B,3,H,W)→(B,16,H/2,W/2)` | stem | `conv1` |
| `DepthwiseSeparable` | `dw(k,s,g=in) → BN → act → [SE] → pw(1×1) → BN → act` | 同维 | 轻量卷积 | `DepthwiseSeparable` |
| `SqueezeExcitation` | `GAP → Conv(ch→ch/4) → ReLU → Conv(→ch) → Hardsigmoid → 乘` | 同维 | 通道注意力（仅 blocks5/6） | `SqueezeExcitation` |
| **`MultiLabelHead`** | `GAP → Conv2d(512→1280,1) → Hardswish → Dropout → Linear(1280,nc)` | →`(B,nc)` | ★ 多标签头 | `MultiLabelHead` |
| `class_expand` | 默认 **1280** | | 分类瓶颈 | `class_expand` |

### 2.3 与图像分类（Classify）的差异

| 项 | classify | **attribute** |
|---|---|---|
| backbone | YOLO backbone | **PP-LCNet_x1_0（CPU 友好）** |
| 头 | Conv→GAP→Linear | **Conv(→1280)→Hardswish→Dropout→Linear** |
| 损失 | CrossEntropy（单标签） | **BCEWithLogits + ratio2weight（多标签）** |
| 后处理 | softmax + topk | **sigmoid + 阈值（0.5）** |
| 指标 | top-1/5 | **mA（每属性准确率）/ mAP** |
| 数据标签 | `path label` | **`path v1 v2 ... vC`（multi-hot）** |

---

## 3. 输出与后处理（★ 重点）

### 3.1 多标签输出

```
logits = model(x)                          # (B, C)，C=属性数
scores = sigmoid(logits)                   # (B, C)，每属性独立概率
hit    = [i for i where scores[i] >= threshold(0.5)]
# MultiLabelThresPostProcess 返回每样本 {"scores", "labels", "attributes"(名字)}
```

### 3.2 数据格式（`AttributeDataset`）

```
# label file（PaddleClas MultiLabelDataset）：
relative/path.jpg  v1 v2 ... vC      # multi-hot 或软标注
```

| 项 | 说明 |
|---|---|
| `label_ratio` | **默认 true**：每个样本额外携带**数据集级正例比例** `ratio`（供损失重加权） |
| 图像尺寸 | `transform.image_size: [W, H]`；行人 **[192,256]**、车辆 **[192,256]（config 默认同）** |
| 预处理 | resize → 训练 `crop_pad`（Padv2）+ RandomCrop / 评估居中；RGB → 除 255 → ImageNet 归一化 |
| 增广 | `flip`（默认开）、`randaugment`（p≈0.8）、`erasing`（p≈0.4） |

### 3.3 损失（`MultiLabelLoss`，对齐 PaddleClas）

```
logits, labels, ratio
cost = BCEWithLogits(logits, target, reduction="none")      # 每元素
if weight_ratio and ratio is not None:
    mask = (labels > 0.5)
    cost = cost * ratio2weight(mask, ratio)                 # exp((1-t)r + t(1-r))
loss = cost.sum(1).mean()  if size_sum else cost.mean()     # 默认 size_sum=True
# epsilon 非空时做标签平滑
```

### 3.4 指标（`AttrMetric`）

| 指标 | 计算 |
|---|---|
| `mA` | 每属性在 **阈值 0.5** 下的准确率，再对属性平均 |
| `mAP` | 每属性 AP（按 score 降序，`Σ prec·gt / Σ gt`）后平均 |
| `acc` | 回落为 `mA`（兼容 classify 的 `main_indicator`） |

---

## 4. 配置与用法

### 4.1 配置（`configs/attr/pedestrian_attribute.yml`）

```yaml
Global:
  pretrained_model: https://www.modelscope.cn/models/ChaoII0987/TorchKiln/resolve/master/pretrained/PP-LCNet_x1_0_pedestrian_attribute.pth
Architecture:
  task: attribute
  algorithm: PP-LCNet_x1_0_pedestrian_attribute
  Backbone: {name: PPLCNet_x1_0, scale: 1.0, dropout_prob: 0.5}   # use_ssld: True
  Head: {num_classes: 26, class_expand: 1280, dropout_prob: 0.5,
         label_list: torchkiln/utils/pedestrian_attribute_label_list.txt}
Loss:  {name: MultiLabelLoss, weight_ratio: true, size_sum: true}
Metric:       {name: AttrMetric, main_indicator: mA, threshold: 0.5}
PostProcess:  {name: MultiLabelThresPostProcess, threshold: 0.5}
Train:
  dataset: {name: AttributeDataset, label_ratio: true,
            transform: {image_size: [192,256], resize: [192,256], crop_pad: [212,276]},
            augment: {flip: true, randaugment: {...}, erasing: {p: 0.4}}}
```

车辆版 `configs/attr/vehicle_attribute.yml`：`num_classes: 19`、`label_list` 改为车辆属性表；其余同。

### 4.2 四条链路

```bash
tkiln train   -c configs/attr/pedestrian_attribute.yml
tkiln val     -c configs/attr/pedestrian_attribute.yml --weights output/.../best_accuracy.pth
tkiln predict -c configs/attr/pedestrian_attribute.yml --weights ... --input imgs/   # 输出命中属性名
tkiln check   -c configs/attr/vehicle_attribute.yml
```

---

## 5. 规模与速度

| 模型 | backbone | class_expand | 属性数 | 输入 (W,H) |
|---|---|---|---|---|
| 行人属性 | `PPLCNet_x1_0`（scale 1.0） | 1280 | **26** | 192×256 |
| 车辆属性 | `PPLCNet_x1_0`（scale 1.0） | 1280 | **19** | 192×256 |

- **参数量 / FLOPs / 本机耗时（CPU/GPU）未在本框架侧统计**（不编造）。
- PP-LCNet_x1_0 设计为 **CPU 实时**；本框架 backbone `out_channels=512`（scale 1.0）。

---

## 6. 公开指标

| 项 | 状态 |
|---|---|
| **权重加载对齐（PaddleClas → 本框架）** | 两个权重均可通过 `Global.pretrained_model` 加载（配置已内置 ModelScope URL） |
| **前向 / 单步 loss / 梯度对拍** | **未记录**（AGENTS 未收录 attribute 的数值对齐记录） |
| **公开基准（PETA/PA100K/VehicleID 等）** | **未统计** |
| 结构 | backbone **逐层照搬** PaddleClas `NET_CONFIG x1_0`；损失 `ratio2weight` 与 PaddleClas 同名同式 |

> ⚠️ **如实说明**：结构与损失公式按 PaddleClas 复刻，权重可直接加载，
> 但**本框架侧未做逐层前向/损失/梯度的数值对拍，也无公开基准成绩**。
> 若要验收，建议按统一对齐流程补做「权重 missing/unexpected + 同输入 logits maxdiff」。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 行人属性（26 项） | **`pedestrian_attribute.yml`** | 官方权重 + 标签表齐全 |
| 车辆属性（19 项） | **`vehicle_attribute.yml`** | 同上 |
| 通用图像多标签（自定义属性） | **classify + `Loss.multi_label=true`** | 复用 YOLO backbone |
| 单标签分类 | classify | 更简单 |
| 边缘 CPU 部署 | **本任务（PP-LCNet）** | 专为 CPU 优化 |

---

## 8. 参考与对比

| 对比 | 说明 |
|---|---|
| **vs classify（多标签）** | 同一 BCE+阈值 范式；attribute 用 PP-LCNet + ratio 重加权 + 专用预处理 |
| **vs 单标签 softmax** | 多标签用 sigmoid（各属性独立），softmax 会互斥，错误 |
| **vs PaddleClas 官方** | backbone/loss 同源；本框架用 PyTorch 重写、权重经转换加载 |
| **行人 vs 车辆** | 仅 `num_classes` 与标签表不同，**同一代码路径** |

---

## 9. 已知问题 / 注意事项

- ⚠️ **对齐状态未验证**：权重可加载，但**未做数值对拍**（见 §6）；报告指标前须注明。
- ⚠️ **多标签必须用 sigmoid，不能用 softmax**（属性之间不互斥）。
- ⚠️ **`label_ratio: true` 与 `Loss.weight_ratio: true` 必须配对**：缺少 `ratio` 时损失自动跳过重加权
  （`batch` 只有 2 项时 `ratio=None`）。
- ⚠️ **图像尺寸非方**：`image_size: [W,H]`（宽在前），`crop_pad` 是 (w,h)；行人/车辆同为 192×256。
- ⚠️ **`use_ssld` 检查点 → `dropout_prob=0.5`**：PaddleX 两个属性模型都设 `use_ssld: True`，
  配置里 dropout 为 0.5（从零训练才是 0.2）。
- ⚠️ **`AttrMetric` 的 `mAP`** 在正例为 0 的属性上跳过；类别极不平衡时 mA 会被少数类拉低。
- ⚠️ **`margin`/`scale` 等 ArcFace 相关配置未使用**（PaddleClas 分类头才需要）。
- ⚠️ **评估 fp32**：`_evaluate_loop` 强制 `autocast(enabled=False)`。
