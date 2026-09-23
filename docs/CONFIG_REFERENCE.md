# 配置项全解

所有配置都是四层结构,`ptcore.config` 加载后是一个嵌套 dict,`-o A.B.C=v` 可覆盖任意字段。

```
Global          全局训练参数
Architecture    模型(决定 model_family / task / 结构)
Optimizer       优化器 + 学习率策略 + 正则
Loss            损失及其超参
Metric          评估指标
PostProcess     推理/评估后处理
Train / Eval    dataset + loader
```

## Global

| 字段 | 默认 | 说明 |
|---|---|---|
| `model_name` | — | 实验名(日志/目录用) |
| `use_gpu` | `true` | 是否用 GPU |
| `gpu` | — | `0,1,2,3` 或 `0` 指定卡;多卡配合 `distributed` |
| `distributed` | `false` | `true` → DDP,`false` 且多卡 → DP |
| `epoch_num` | — | 训练轮数 |
| `print_batch_step` | `10` | 日志打印间隔(step) |
| `eval_batch_step` | `[0, 2000]` | `[起始, 间隔]`,评估频率;`[0,0]` 关闭训练中评估 |
| `save_model_dir` | — | 输出目录(`best_accuracy.pth`/`latest.pth`/`train.log`/`config.yml`) |
| `save_epoch_step` | `10` | 定期保存间隔 |
| `pretrained_model` | `null` | 预训练权重(路径 / URL / 缓存名);按**名字+形状**加载 |
| `checkpoints` | `null` | 断点续训(指向 `latest` 前缀,恢复优化器/epoch/EMA) |
| `seed` | `1024` | 随机种子;OCR 数据管线需 `seed=None` 才可复现(见 FAQ) |
| `use_ema` | `false` | 启用 EMA |
| `ema_decay` | `0.9998` | EMA 衰减 |
| `amp` | `false` | 混合精度;`amp_dtype: fp16/bf16` |
| `freeze` | `null` | 冻结前 N 层,或名字前缀列表 `["conv1","blocks2"]` |
| `accumulate` | `1` | 梯度累积步数 |
| `character_dict_path` | — | OCR rec 字符集路径(v5/v6 可自动选择) |

## Architecture

| 字段 | 说明 |
|---|---|
| `model_family` | `ocr` → `torchkiln.ocr.Trainer`;`yolo` → `torchkiln.Trainer`(车牌/属性也用 `yolo`)|
| `task` | OCR:`det/rec/cls/e2e/sr/table...`;YOLO:`detect/segment/obb/pose/classify/semantic/depth/plate_det/plate_rec/attribute` |
| `algorithm` | 模型名(手写模型构建器按名分派;仅日志/兼容用)|
| `yaml_file` / `yaml_text` | 给定时走 **YAML 图模型**(`torchkiln/cfg/models/**`) |
| `scale` | `n/s/m/l/x`,按 YAML `scales` 自动缩放 |
| `in_channels` | 输入通道(默认 3)|
| `Backbone` | 部分任务的结构块:`name`/`scale`(属性)/`dropout_prob` |
| `Head` | 头参数,见下 |

### `Architecture.Head` 常用字段

| 字段 | 适用 | 说明 |
|---|---|---|
| `num_classes` | 全部 | 类别数 |
| `reg_max` | det/obb/seg/pose | `1` = DFL-free(上游 8.4.x / v11 / v26);`>1` 启用 DFL |
| `end2end` | det/obb/seg/pose | NMS-free 端到端头(v10/v26),训练返回 `(one2many, one2one)` |
| `reg_layout` | obb | `upstream`(=`[reg,cls,angle]`)/ `ltrb_angle`(手写头布局)|
| `kpt_shape` | pose / plate_det | 如 `[17,3]`(可见性)/ `[4,2]`(车牌四角)|
| `kpt_label` | plate_det | 关键点个数(车牌 4)|
| `cfg` / `class_expand` / `dropout_prob` / `label_list` | plate_rec / attribute | 识别网络宽度、多标签头维度、dropout、属性名表 |

> 多尺度变体(`*-p2` / `*-p6`)的 anchor stride 由**模型头部自动推导**(`[4,8,16,32]` / `[8,16,32,64]`),
> 配置里写不写 `strides` 都会被纠正为真实层数。

## Optimizer

```yaml
Optimizer:
  name: AdamW            # Adam / AdamW / SGD / RAdam / NAdam
  momentum: 0.937        # SGD / RAdam 用
  beta1: 0.9             # Adam 系
  beta2: 0.999
  nesterov: false
  lr:
    name: Cosine         # Cosine / Linear / OneCycle / Const
    learning_rate: 0.001
    lrf: 0.01            # 余弦终值系数(缺省按 Paddle 语义衰减到 0)
    warmup_epoch: 1      # 或 warmup_steps
  regularizer:
    name: L2
    factor: 0.0005
```

## Loss / Metric / PostProcess

| 任务 | Loss(`name`) | Metric(`name`, main_indicator)| PostProcess(`name`)|
|---|---|---|---|
| OCR det | `DBLoss` + Shrink/Threshold/Binary | `DetMetric`(hmean)| `DBPostProcess`(Paddle 原版)|
| OCR rec | `CTCLoss` / `MultiLoss` / `SARLoss` | `RecMetric`(acc)| `CTCLabelDecode`|
| YOLO detect | `DetLoss`(TAL) | `DetMetric`(`mAP50-95`)| `DetPostProcess`|
| YOLO obb | `ObbLoss` | `DetMetric`(`box_format: xywhr`)| `DetPostProcess`(`box_type: xywhr`)|
| YOLO segment | `SegLoss` | `SegMetric`(`mask_mAP50-95`)| `SegPostProcess`|
| YOLO pose | `PoseLoss` | `PoseMetric`(`mAP50-95`)| `PosePostProcess`|
| YOLO classify | `CrossEntropy` | `ClsMetric`(`acc`)| `ClsPostProcess`|
| YOLO semantic | `SemLoss` | `SemMetric`(`mIoU`)| `SemPostProcess`|
| YOLO depth | `DepthLoss` | `DepthMetric`(`delta1`)| `DepthPostProcess`|
| 车牌 det | `PlateDetLoss`(v5 anchor + WingLoss)| `PlateDetMetric`(`mAP50-95` + 角点 NME)| `PlateDetPostProcess`|
| 车牌 rec | `PlateRecLoss`(CTC + 颜色 CE)| `PlateRecMetric`(`acc` / `char_acc` / `color_acc`)| `PlateRecPostProcess`|
| 属性 | `MultiLabelLoss` | `AttrMetric`(`mA` + `mAP`)| `MultiLabelThresPostProcess`|

通用可覆盖项:`conf_thres`、`iou_thres`、`max_det`、`topk`、`label_smoothing`、`strides`(自动注入)。

## Train / Eval

```yaml
Train:
  dataset:
    name: DetDataset                 # 逻辑名(实际类由 task 决定)
    data_dir: datasets/det_demo      # 图片根目录
    label_file_list: [datasets/det_demo/train.txt]   # 标签清单
    kpt_shape: [4, 2]                # pose / plate_det
    box_format: xywhr                # obb
    label_ratio: true                # 属性:携带数据集级正例比例
    double_plate: false              # 车牌识别:双层牌上下拼接
    transform:
      image_size: 640                # 或 [192, 256](W,H);属性用
      resize: [192, 256]             # 属性:先 resize
      crop_pad: [212, 276]           # 属性:Padv2 (W,H) 后 RandomCrop
      mask_stride: 4                 # segment
    augment: {...}                   # 见 docs/TRAINING.md §2
  loader:
    batch_size_per_card: 8
    shuffle: true
    drop_last: false
    num_workers: 0                   # Windows 建议 0~4
    prefetch_factor: 4
Eval:
  dataset: {...}                     # 同 Train(通常不开增广)
  loader: {batch_size_per_card: 8, shuffle: false, num_workers: 0}
```

## 数据格式速查

| 任务 | 标签行格式(归一化坐标)|
|---|---|
| OCR det | 四点坐标 `x1 y1 x2 y2 ...`(文本多边形)+ 文本 |
| OCR rec | `路径\t文本` |
| YOLO detect | `cls cx cy w h` |
| YOLO obb | `cls cx cy w h angle`(弧度,le90)|
| YOLO pose / plate_det | `cls cx cy w h px py v ...`(plate_det 用 `px py`,无 v)|
| YOLO segment | 检测行 + 多边形点 / 掩码 |
| YOLO classify | `路径 类别号` |
| YOLO semantic | 掩码 PNG(像素值=类别索引)|
| YOLO depth | 深度图(16bit PNG / npy)|
| 车牌 rec | `路径 c1 c2 ... c7 颜色号`(c 为字符表下标,0=CTC blank)|
| 属性 | `路径 v1 v2 ... vC`(0/1 或软标签)|
