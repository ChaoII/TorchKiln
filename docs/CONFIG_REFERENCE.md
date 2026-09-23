# 配置项全解

所有配置都是四层结构,`ptcore.config` 加载后是一个嵌套 dict,`-o A.B.C=v` 可覆盖任意字段。
完整训练/评估/推理/导出命令见 [`TRAINING.md`](TRAINING.md);数据格式见 [`DATASET_FORMATS.md`](DATASET_FORMATS.md);
**小白向全参数+默认值+引号规则**见 [`USER_GUIDE.md`](USER_GUIDE.md)。

```
Global          全局训练参数
Architecture    模型(决定 model_family / task / 结构)
Optimizer       优化器 + 学习率策略 + 正则
Loss            损失及其超参
Metric          评估指标
PostProcess     推理/评估后处理
Train / Eval    dataset + loader
```

可用 `-o` 段名(写错会 ValueError 列出下表):
`Global, Architecture, Loss, Optimizer, PostProcess, Metric, Train, Eval`(部分旧配置另有 `dataset` 等顶层兼容键)。

## Global

| 字段 | 默认 | 说明 |
|---|---|---|
| `model_name` | — | 实验名(日志/目录用) |
| `use_gpu` | `true` | 无 `device` 时的回退开关 |
| **`device`** | 空 | **多卡/指定卡的真键**:`cpu` / `cuda:0` / `gpu:0` / `gpu:0,1`;**不要**写 `'0'`(会解析到 CPU) |
| `dist_backend` | 自动 | DDP 后端:有 NCCL 用 `nccl`,否则 `gloo`;可强制 |
| `dist_init_method` | `env://` | 进程组初始化;`torchrun` 默认即可 |
| `distributed` | — | **历史死键**:训练循环**不读**,是否 DDP 只看环境变量 `WORLD_SIZE`(`torchrun` 注入) |
| `gpu` | — | **历史死键**:请改用 `Global.device` |
| `epoch_num` | — | 训练轮数 |
| `print_batch_step` | `10` | 日志打印间隔(step) |
| `eval_batch_step` | `[0, 2000]` | `[起始, 间隔]`,评估频率;`[0,0]` 关闭训练中评估 |
| `eval_epoch_step` | — | 按 epoch 评估;慢评估时设大可提速 |
| `save_model_dir` | — | 输出目录(`best_accuracy.pth`/`latest.pth`/`train.log`/`config.yml`) |
| `save_epoch_step` | `10` | 定期保存间隔 |
| `pretrained_model` | `null` | 预训练权重(路径 / URL / 缓存名);按**名字+形状**加载 |
| `checkpoints` | `null` | 断点续训(指向 `latest` 前缀,恢复优化器/epoch/EMA);别名 `Train.resume_path` |
| `seed` | `1024` | 随机种子;OCR 数据管线需 `seed=None` 才可复现(见 FAQ) |
| `use_ema` | `false` | 启用 EMA;**`tkiln val` 强制 false**(评原始权重) |
| `ema_decay` | `0.9998` | EMA 衰减 |
| `ema_decay_type` | threshold | `threshold` / `exponential`(对齐 ultra 用 exponential) |
| `amp` | `false` | 混合精度;`amp_dtype: fp16/bf16`;评估阶段强制 fp32 |
| `freeze` | `null` | 冻结前 N 层,或名字前缀列表 `["conv1","blocks2"]` |
| `accumulate` | `1` | 梯度累积步数 |
| `character_dict_path` | — | OCR rec 字符集路径(v5/v6 可自动选择) |
| `show_eval_progress` | `true` | 评估进度条 |

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
| YOLO classify | `CrossEntropy`(单标签)/ `MultiLabelLoss`(多标签)| `ClsMetric`(`acc`)/ `AttrMetric`(`mAP@0.5`)| `ClsPostProcess` / `MultiLabelThresPostProcess`|
| YOLO semantic | `SemLoss` | `SemMetric`(`mIoU`)| `SemPostProcess`|
| YOLO depth | `DepthLoss` | `DepthMetric`(`delta1`)| `DepthPostProcess`|
| 车牌 det | `PlateDetLoss`(v5 anchor + WingLoss)| `PlateDetMetric`(`mAP50-95` + 角点 NME)| `PlateDetPostProcess`|
| 车牌 rec | `PlateRecLoss`(CTC + 颜色 CE)| `PlateRecMetric`(`acc` / `char_acc` / `color_acc`)| `PlateRecPostProcess`|
| 属性 | `MultiLabelLoss` | `AttrMetric`(`mA` + `mAP`)| `MultiLabelThresPostProcess`|

通用可覆盖项:`conf_thres`、`iou_thres`、`max_det`、`topk`、`label_smoothing`、`strides`(自动注入)。
**classify 多标签一键**:`-o Loss.multi_label=true` → 自动 `MultiLabelLoss` + `AttrMetric`(mAP@0.5) + `MultiLabelThresPostProcess`;

## Train / Eval

数据接入**只有** `data_dir` + `label_file_list` 两个路径字段;OCR 额外用 `name`/`transforms`,
YOLO 用 `transform`/`augment`(**单数 transform,与 OCR 复数 transforms 两套体系**)。

```yaml
Train:
  dataset:
    name: DetDataset                 # 逻辑名(实际类由 task 决定;OCR 常为 SimpleDataSet)
    data_dir: datasets/det_demo      # 图片根目录
    label_file_list: [train.txt]     # 标签清单;相对 data_dir,或已存在的绝对路径
    kpt_shape: [4, 2]                # pose / plate_det(另需写入 Architecture.Head / Loss)
    box_format: xywhr                # obb:标签为 9 字段四角点,内部 poly2rbox
    names: [plate]                   # YOLO 类别名(可 -o 逗号分隔)
    label_ratio: true                # 属性 / classify 多标签:携带数据集级正例比例
    multi_label: true                # classify 多标签:按路径 v1..vC 解析(通常由 Loss.multi_label 自动带上)
    double_plate: false              # 车牌识别:双层牌上下拼接
    ignore_index: 255                # semantic:忽略像素
    depth_scale: 1000                # depth:uint16 → 物理单位除数
    mask_stride: 4                   # segment(常写在 transform 下,见各配置)
    transform:                       # YOLO 系(单数)
      image_size: 640                # 或 [192, 256](W,H);属性用
      resize: [192, 256]             # 属性:先 resize
      crop_pad: [212, 276]           # 属性:Padv2 (W,H) 后 RandomCrop
      mask_stride: 4                 # segment
    transforms: [...]                # 仅 OCR(name=SimpleDataSet):Decode/Resize/...
    augment:                         # 空 dict falsy → 不建增广器;要开须写显式键
      mosaic: 1.0
      hsv: {p: 0.5}
      affine: {degrees: 0.0, translate: 0.1, scale: 0.5}
  loader:
    batch_size_per_card: 8           # 批大小(与 dataset 同级,不是 dataset.loader)
    shuffle: true
    drop_last: false
    num_workers: 4                   # Windows 建议 0~4(Eval 用 0)
    prefetch_factor: 4
Eval:
  dataset: {...}                     # 同 Train(通常不开 augment)
  loader: {batch_size_per_card: 8, shuffle: false, num_workers: 0}
```

路径解析规则:

| `label_file_list` 项 | 行为 |
|---|---|
| 非绝对路径且不是已存在文件 | 拼到 `data_dir` 下 |
| 已存在文件(含绝对路径) | 直接打开 |

> 历史文档中的 `Train.dataset.train_list` / `val_list` **不存在**;batch 写在 `Train.loader`,
> 不是 `Train.dataset.loader`(旧式写法兼容读取,但请用 `Train.loader`)。

### 多卡(真写法)

```powershell
# DDP:是否分布式只看 WORLD_SIZE,只读 Global.device
torchrun --nproc_per_node=4 tools/train.py -c <cfg>.yml -o Global.device=gpu:0,1,2,3

# Windows 单进程 DP:直接把多卡写进 device
python tools/train.py -c <cfg>.yml -o Global.device=gpu:0,1
```

`-o Global.distributed=true -o Global.gpu=0,1` **无效**(死键,仅 warning)。

## 数据格式速查

| 任务 | 标签行格式(归一化坐标)|
|---|---|
| OCR det | 四点坐标 JSON `transcription + points`(或 `###` 忽略) |
| OCR rec | `路径\t文本` |
| YOLO detect | `cls cx cy w h` |
| YOLO obb | **`cls x1 y1 x2 y2 x3 y3 x4 y4`**(4 角点归一化,9 个数;`box_format=xywhr`)|
| YOLO pose / plate_det | `cls cx cy w h px py v ...`(plate_det 用 `px py`,无 v)|
| YOLO segment | 检测行 + 多边形点 / 掩码 |
| YOLO classify | `路径 类别号`(单标签);**多标签** `路径 v1..vC` + `Loss.multi_label=true` |
| YOLO semantic | 掩码 PNG(像素值=类别索引)|
| YOLO depth | 深度图(16bit PNG / npy)|
| 车牌 rec | `路径 c1 c2 ... c7 颜色号`(c 为字符表下标,0=CTC blank)|
| 属性 | `路径 v1 v2 ... vC`(0/1 或软标签)|
