# 训练 / 评估 / 推理 / 导出 —— 完整命令手册

> **新手请先读 [`USER_GUIDE.md`](USER_GUIDE.md)**（同内容更小白向：全参数默认值、引号规则、报错速查、模型对照表）。
> 本文偏「参数真值 + 命令链」速查。

所有产品线（OCR / YOLO / 车牌 / 属性）共用同一套 CLI：读 `Architecture.model_family` 选 Trainer，
再用 `Architecture.task` 选任务适配器。**任务真值在配置里**，命令行 task 只做命名空间与一致性校验。

---

## 0. 快速开始（3 步）

```powershell
conda activate ptocr
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"

# 1) 自检：只建模型/损失/指标，不训练
.\tkiln.bat check -c configs/ocr/det/PP-OCRv4_mobile_det.yml

# 2) 训练（不改 yml，用 -o 指向自己的数据并覆盖超参）
.\tkiln.bat train -c configs/ocr/det/PP-OCRv4_mobile_det.yml `
  -o Train.dataset.data_dir=D:/mydata/det `
  -o Train.dataset.label_file_list=train.txt `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt `
  -o Global.save_model_dir=./output/my_det `
  -o Global.epoch_num=100

# 3) 评估 / 推理 / 导出
.\tkiln.bat val     -c configs/ocr/det/PP-OCRv4_mobile_det.yml --weights output/my_det/best_accuracy.pth `
  -o Eval.dataset.data_dir=D:/mydata/det -o Eval.dataset.label_file_list=val.txt
.\tkiln.bat predict -c configs/ocr/det/PP-OCRv4_mobile_det.yml --weights output/my_det/best_accuracy.pth `
  --input D:/mydata/det/images/val/0001.jpg --output output/det_vis.jpg
.\tkiln.bat export  -c configs/ocr/det/PP-OCRv4_mobile_det.yml --weights output/my_det/best_accuracy.pth `
  --save-dir output/export/my_det --onnx --slim
```

`-o A.B.C=value` 按 yml 原类型自动转换（int/float/bool/list）；可重复；列表可写
`-o Train.dataset.label_file_list=train.txt,val.txt`（逗号分隔，无需中括号）。
**PowerShell** 下含 `=` 的项建议整体加双引号：`-o "Train.dataset.data_dir=D:/my data"`。

---

## 1. CLI 总览

### 1.1 四套等价入口

| 写法 | 说明 |
|---|---|
| `.\tkiln.bat <mode> ...` / `./tkiln <mode> ...` | 仓库根启动器（推荐，免 pip） |
| `tkiln <mode> ...` | `pip install -e .` 后的 console script |
| `python -m torchkiln <mode> ...` | 免安装等价 |
| `python tools/train.py ...` 等 | 底层脚本，参数与上面完全一致 |

解释器选择：`TKILN_PYTHON` → `CONDA_PREFIX` → `ptocr` 环境 → `python`。

### 1.2 mode 一览

| mode | 路由脚本 | 用途 |
|---|---|---|
| `train` | `tools/train.py` | 训练 |
| `val` | `tools/eval.py` | 评估（强制 `epoch_num=0`、`use_ema=false`） |
| `predict` | `tools/infer/predict_{yolo,det,rec}.py` | 单图推理（按 task 自动路由） |
| `export` | `tools/export.py` | 导出 pth / TorchScript / ONNX / slim |
| `check` | CLI 内置 | 只构建 + 打印 model/loss/metric/postprocess 摘要 |
| `data` | `torchkiln/datasets.py` | 数据集检查 / 下载 |

### 1.3 task 别名（可省略）

```text
detect|det  segment|seg  obb  pose  classify|cls  semantic|sem  depth
plate_det|plate-det  plate_rec|plate-rec  attribute|attr
pose_action|pose-action|action  video_cls|video-cls|video
ocr  ocr_det|ocr-det  ocr_rec|ocr-rec
```

显式写法（与省略等价，多做一次配置一致性校验）：

```powershell
.\tkiln.bat detect train -c configs/yolo/yolov8-det.yml -o Global.epoch_num=100
```

predict 路由：YOLO 族 → `predict_yolo.py`；OCR det → `predict_det.py`；OCR rec → `predict_rec.py`。

### 1.4 `tkiln data` 数据集子命令

```powershell
tkiln data list                 # 检查 datasets/manifest.yml：labels / 图片 / url 就绪性
                                # 别名: verify / status（默认子命令）
tkiln data get plate_det_demo   # 从 ModelScope 下载 zip → 解包 datasets/<name>/
tkiln data get --all            # 全部；--force 已就绪仍重下
                                # 别名: download / pull
```

> 实现于 `torchkiln/datasets.py`；**没有** `tools/download_dataset.py`。
> 当前 `manifest.yml` 多数 `url: null`，未填 URL 时 `data get` 会拼 `prefix/<name>.zip` 可能 404——
> 本地自检请用 `python tools/make_demo_data.py --all` 生成占位图。

### 1.5 查看各工具自己的 argparse help

```powershell
python tools/train.py --help
python tools/eval.py --help
python tools/export.py --help
python tools/infer/predict_yolo.py --help
python tools/infer/predict_det.py --help
python tools/infer/predict_rec.py --help
```

> `tkiln --help` 打印的是本 CLI 的 USAGE，**不会**透传到 tools 的 argparse。

---

## 2. `-o` 覆盖语法（详细）

| 写法 | 效果 |
|---|---|
| `-o Global.epoch_num=100` | int |
| `-o Optimizer.lr.learning_rate=0.001` | float |
| `-o Global.use_ema=true` | bool（`true/false`） |
| `-o Global.pretrained_model=null` | None（`null`/`none`） |
| `-o Train.dataset.label_file_list=a.txt,b.txt` | list（逗号拆） |
| `-o Train.dataset.transform.image_size=[640,640]` | yaml list |
| `-o Global.amp` | 裸 key = 置 `True` |
| `-o A.B=1 -o C.D=2` 或 `-o A.B=1 C.D=2` | 可重复 / 一次多项 |

规则：

- 首段必须是配置已有 section（`Global`/`Architecture`/`...`），写错会 `ValueError` 并列出可用 section。
- 未知子键 **warning 但仍写入**；已应用键记入 `config["_overrides"]` 并落盘到 `save_model_dir/config.yml`。
- 配置不存在的 section **不会**被 `-o` 凭空创建。

---

## 3. 数据集接入（如何用命令把数据喂给训练）

### 3.1 字段只有两个（没有 `train_list` / `val_list`）

```yaml
Train:
  dataset:
    data_dir: D:/mydata/det          # 根目录
    label_file_list: [train.txt]     # 相对 data_dir，或绝对路径
Eval:
  dataset:
    data_dir: D:/mydata/det
    label_file_list: [val.txt]
```

路径语义（`torchkiln/data/det.py`）：

- 项**不是绝对路径**且**当前不是已存在文件** → 拼到 `data_dir` 下；
- 项已是存在的文件（含绝对路径）→ 直接用。

因此两种写法都合法：

```powershell
# 相对 data_dir（推荐，配置可移植）
-o Train.dataset.data_dir=D:/mydata/det -o Train.dataset.label_file_list=train.txt

# 绝对路径
-o Train.dataset.label_file_list=D:/mydata/det/train.txt
```

### 3.2 完整：自定义 YOLO 检测集从零训练

```powershell
# 目录准备（详见 docs/DATASET_FORMATS.md §3）
#   D:/mydata/det/
#     images/train/*.jpg  images/val/*.jpg
#     labels/train/*.txt  labels/val/*.txt     # 每行: cls cx cy w h (归一化)
#     train.txt  val.txt                       # 只列相对图片路径

.\tkiln.bat check -c configs/yolo/yolo11-det.yml `
  -o Train.dataset.data_dir=D:/mydata/det `
  -o Architecture.Head.num_classes=1 `
  -o Train.dataset.names=[plate]

.\tkiln.bat train -c configs/yolo/yolo11-det.yml `
  -o Train.dataset.data_dir=D:/mydata/det `
  -o Train.dataset.label_file_list=train.txt `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt `
  -o Architecture.Head.num_classes=1 `
  -o Train.dataset.names=[plate] `
  -o Global.save_model_dir=./output/my_y11 `
  -o Global.epoch_num=100 `
  -o Train.loader.batch_size_per_card=8 `
  -o Global.device=cuda:0
```

### 3.3 其它任务：在 3.2 基础上补的字段

| 任务 | 必须额外对齐的字段 |
|---|---|
| OBB | `Train/Eval.dataset.box_format=xywhr`；标签 **9 字段角点**（见 DATASET_FORMATS §4） |
| pose | `kpt_shape` 写在 **三处**：`Architecture.Head` + `Loss` + `Train/Eval.dataset` |
| seg | `transform.mask_stride`（默认 4）；标签多边形或 `masks/` |
| sem | `ignore_index`（默认 255）；`masks/<split>/x.png` 同名 |
| depth | `depth_scale`（默认 1000 = 毫米）；`depth/<split>/` 或 `.npy` |
| cls | `names` + `Head.num_classes`；清单 `路径 类别号`；**多标签** `-o Loss.multi_label=true` + `路径 v1..vC` |
| plate_det | `kpt_shape: [4,2]`（dataset）+ `Head.kpt_label: 4`、`num_classes: 2` |
| plate_rec | `transform.image_size: [48,168]`、`Head.color_num: 5`；清单 `路径 c1..c7 颜色` |
| attribute | `label_ratio: true` + `Head.label_list`；清单 `路径 v1..vC` |
| OCR det | `name: SimpleDataSet` + **`transforms:` 列表**（不是 `transform:`） |
| OCR rec | 同上 + `Global.character_dict_path` + `max_text_length` |

模板目录：`datasets/_format_examples/<task>/`（含 `README.txt` 配置片段）。
生成：`python tools/make_format_examples.py` 或 `--task det --task pose`。

### 3.4 格式转换（当前仅两向）

```powershell
python tools/convert/dataset_format.py --from yolo_det --to ocr_det --src <src> --dst <dst>
python tools/convert/dataset_format.py --from ocr_det --to yolo_det --src <src> --dst <dst> --nc 1 --names plate
python tools/convert/dataset_format.py --from yolo_det --to yolo_det --src <src> --dst <dst>   # 规整+硬链接
```

`ocr_rec` / `cls` / `seg` 方向**尚未实现** reader/writer（docstring 有写，READERS 表没有）。

---

## 4. 训练

### 4.1 基础

```powershell
# OCR 识别（v5/v6 可自动选 dict；否则显式给 character_dict_path）
.\tkiln.bat train -c configs/ocr/rec/PP-OCRv5_mobile_rec.yml -o Global.epoch_num=100

# YOLO 图模型（configs/yolo/*.yml，53 个可训练）
.\tkiln.bat train -c configs/yolo/yolov8-det.yml -o Global.epoch_num=100

# 端到端变体
.\tkiln.bat train -c configs/yolo/yolo26-p2-det.yml -o Global.epoch_num=200

# 车牌 / 属性（预训练微调）
.\tkiln.bat train -c configs/plate/plate_det.yml -o Global.pretrained_model=_downloads/plate/plate_detect.pth
.\tkiln.bat train -c configs/attr/vehicle_attribute.yml `
  -o Global.pretrained_model=~/.torchkiln/pretrained/PP-LCNet_x1_0_vehicle_attribute.pth
```

常用覆盖：

```powershell
-o Global.epoch_num=100
-o Global.save_model_dir=./output/exp
-o Global.pretrained_model=null                 # 从零
-o Global.pretrained_model=output/pre/best_accuracy.pth
-o Global.use_ema=true -o Global.ema_decay=0.9999 -o Global.ema_decay_type=exponential
-o Global.amp=true -o Global.freeze=8 -o Global.accumulate=2
-o Optimizer.name=SGD -o Optimizer.lr.learning_rate=0.01
-o Train.loader.batch_size_per_card=4 -o Train.loader.num_workers=0
-o Eval.loader.num_workers=0
-o Global.device=cuda:0
```

### 4.2 断点续训 vs 预训练微调

| 用途 | 字段 | 恢复内容 |
|---|---|---|
| **断点续训** | `Global.checkpoints=output/exp/latest.pth`（或前缀 `.../latest`） | 模型 + 优化器 + lr_scheduler + EMA + epoch + global_step + best_metric |
| **仅权重微调** | `Global.pretrained_model=<path\|URL\|缓存名>` | 只有权重，epoch 从 0 |

```powershell
# 续训：把 epoch 调大，否则训满会打印 nothing to do
.\tkiln.bat train -c configs/yolo/yolo11-det.yml `
  -o Global.checkpoints=output/y11/latest.pth `
  -o Global.epoch_num=200
```

> `Train.resume_path` 是 `Global.checkpoints` 的别名（OCR 旧配置兼容）。

### 4.3 训练特性（与 ultralytics 对齐）

```yaml
Global:
  amp: true
  freeze: 8                  # 或 ["conv1", "blocks2"]
  accumulate: 2
  use_ema: true
Optimizer:
  name: AdamW                # Adam / AdamW / SGD / RAdam / NAdam
  lr: {name: Cosine, learning_rate: 0.01, lrf: 0.01, warmup_epoch: 3}
  regularizer: {name: L2, factor: 0.0005}
Train:
  dataset:
    augment:
      mosaic: 1.0
      mixup: 0.1
      close_mosaic: 60
      multi_scale: 0.3
      hsv: {p: 0.5, hgain: 0.015, sgain: 0.7, vgain: 0.4}
      affine: {degrees: 0.0, translate: 0.1, scale: 0.5, shear: 2.0, perspective: 0.0005}
```

几何增广对框 / 旋转框 / 关键点 / 掩码用同一 homography，不错位。
`augment: {}`（空 dict）是 falsy → **不会**建增广器；要开必须写显式键。

### 4.4 多卡（真写法）

实现真值（`ptcore/trainers/base.py`）：

- **DDP**：是否分布式只看环境变量 `WORLD_SIZE`（`torchrun` 会设），**不读** `Global.distributed`；
- **设备**：只读 `Global.device`（如 `gpu:0,1` / `cuda:0`），**不读** `Global.gpu`。

```powershell
# 推荐 DDP（每进程一卡）
torchrun --nproc_per_node=4 tools/train.py -c configs/yolo/yolo11-det.yml `
  -o Global.device=gpu:0,1,2,3 -o Train.loader.batch_size_per_card=4

# Windows 单进程 DP（DataParallel）
python tools/train.py -c configs/yolo/yolo11-det.yml -o Global.device=gpu:0,1
```

> 旧文档中的 `-o Global.distributed=true -o Global.gpu=0,1,2,3` 是**死键**——`-o` 只 warning 不报错，
> 命令能跑但仍是单卡。请改用上面写法。

---

## 5. 评估（`tkiln val`）

```powershell
.\tkiln.bat val -c configs/yolo/yolo11-det.yml `
  --weights output/y11/best_accuracy.pth `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt
```

| 参数 | 说明 |
|---|---|
| `-c` | 必填 |
| `--weights` | 可选，覆盖 `Global.pretrained_model`（路径/URL/缓存名均可） |
| `-o ...` | 同训练 |

内部固定：`Global.epoch_num=0`、**`Global.use_ema=false`**（评原始权重，非 EMA）、
`save_model_dir` 默认 `./output/_eval`。打印全部指标 + `Metric.main_indicator`。

无 Eval 集时打 warning 并返回。多卡/大图注意 `Eval.loader.num_workers=0`。

---

## 6. 推理（`tkiln predict`）

**只支持单张图片**（无目录批推理、无 JSON 落盘批量接口）。

### 6.1 YOLO 族（detect / seg / obb / pose / cls / sem / depth / plate / attr）

```powershell
.\tkiln.bat predict -c configs/yolo/yolo11-det.yml `
  --weights output/y11/best_accuracy.pth `
  --input D:/img.jpg `
  --output output/yolo_result.jpg `
  --device cuda:0
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `-c` | — | 必填 |
| `--weights` | — | 必填 |
| `--input` | — | 必填，**单图路径** |
| `--output` | `output/yolo_result.jpg` | 可视化输出 |
| `--device` | `cuda:0` | 非 cuda 前缀或无 GPU 回退 CPU |
| `-o` | — | 覆盖配置 |

letterbox 到 `Train.dataset.transform.image_size`；按 `Architecture.task` 自动可视化；
类别名取 `Train.dataset.names`。

### 6.2 OCR 文本检测

```powershell
.\tkiln.bat predict -c configs/ocr/det/PP-OCRv4_mobile_det.yml `
  --weights output/.../best_accuracy.pth `
  --input D:/img.jpg --output output/det_result.jpg
```

内部 `_DetResizeForTest(limit_side_len=960, limit_type="max")`。

### 6.3 OCR 文本识别

```powershell
.\tkiln.bat predict -c configs/ocr/rec/PP-OCRv5_mobile_rec.yml `
  --weights output/.../best_accuracy.pth --input D:/crop.jpg
```

- **无 `--output`**：结果只打印到 stdout（`<路径> -> <文本>`）。
- 输入尺寸取 `Global.d2s_train_image_shape`（默认 `[3,48,320]`）。

### 6.4 等价底层调用

```powershell
python tools/infer/predict_yolo.py -c <cfg> --weights <pth> --input img.jpg --output out.jpg
python tools/infer/predict_det.py   -c <cfg> --weights <pth> --input img.jpg --output out.jpg
python tools/infer/predict_rec.py   -c <cfg> --weights <pth> --input img.jpg
```

> 旧文档中的 `--image_dir` **不存在**，参数名是 `--input`。

---

## 7. 导出（`tkiln export`）

```powershell
$env:PYTHONIOENCODING="utf-8"
.\tkiln.bat export -c configs/yolo/yolo11-det.yml `
  --weights output/y11/best_accuracy.pth `
  --save-dir output/export/y11 `
  --fuse --onnx --slim --opset 18
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `-c` | 是 | — | 配置 |
| `--weights` | 是 | — | checkpoint `.pth`（经 resolve，支持 URL/缓存名） |
| `--save-dir` | 是 | — | 输出目录 |
| `-o` | 否 | — | 覆盖 |
| `--fuse` | 否 | false | 融合 rep / Conv+BN |
| `--onnx` | 否 | false | 导出 `model.onnx` |
| `--slim` | 否 | false | 对**已存在的** `model.onnx` 跑 onnxslim → `model_slim.onnx` |
| `--opset` | 否 | YOLO=18 / OCR=11 | ONNX opset |
| `--legacy-exporter` | 否 | false | `dynamo=False` 旧导出器 |
| `--torchscript` | 否 | **true（恒开）** | 导出 `model.pt`（`torch.jit.trace`） |

产物固定顺序：

1. `inference.pth` — state_dict（可先 `--fuse`）
2. `inference.yml` — **原 yml 的拷贝，不含 `-o` 覆盖**（用 `-o` 改过结构再导出要注意）
3. `model.pt` — TorchScript
4. `model.onnx` — 若 `--onnx`（batch 维动态；输入名 `x`）
5. `model_slim.onnx` — 若 `--slim` 且已有 onnx

注意：

- **`--slim` 不会自动触发 `--onnx`**；单独 `--slim` 会在 onnxslim 失败（异常被吞，只打印）。
- 端到端头（`Head.end2end: true`，v10/v26）ONNX **图内无 NMS**；非端到端需部署侧外部 NMS。
- 权重按**名字+形状**过滤 `strict=False` 加载，形状不匹配会静默跳过。

---

## 8. 权重转换（Paddle / ultralytics → 本平台）

```powershell
# OCR：① paddlex 环境导出 pickle ② ptocr 环境映射成 torch
& C:\ProgramData\miniconda3\envs\paddlex\python.exe tools/convert/dump_all.py
python tools/convert/convert_all.py

# 上游 YOLO .pt（ultralytics 环境）→ 纯张量 state_dict
& C:\ProgramData\miniconda3\envs\ultralytics\python.exe tools/convert/dump_ultralytics.py yolo11n.pt --out _downloads/upstream

# 车牌（内置 Unpickler，无需上游代码）
python tools/convert/convert_plate_weights.py --detect <plate_detect.pt> --rec <plate_rec_color.pth> --out _downloads/plate

# 属性（PaddleClas .pdparams）
& C:\ProgramData\miniconda3\envs\paddlex\python.exe tools/convert/dump_attribute_paddle.py --weights <x.pdparams> --out <x.pkl>
python tools/convert/convert_attribute_weights.py --pkl <x.pkl> --num-classes 26 --out <x.pth>
```

预下载官方预训练到缓存（新文件名，无 `_ptocr`/`_state` 后缀）：

```powershell
python tools/download_pretrained.py                    # 全部 17 个 PP-OCR
python tools/download_pretrained.py PP-OCRv6_tiny_det  # 指定
python tools/download_pretrained.py --list             # 只打印解析路径
```

缓存目录：`~/.torchkiln/pretrained/`（**不再**回退 `~/.pytorchocr/`）。
ModelScope：`https://www.modelscope.cn/models/ChaoII0987/TorchKiln/resolve/master/pretrained/<name>.pth`

---

## 9. 自检（改代码后必跑）

```powershell
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"

python tools/smoke_all.py           # 每配置: 建模型 + 1 batch 训练 + 1 评估
# 期望: 76 OK, 0 FAIL

python tools/check_graph_build.py   # 53 个 YAML 图配置建图+前向
# 期望: 53 configs: 53 OK, 0 FAIL

python tools/check_plate_models.py  # 车牌张量/ONNX 对齐
# 期望: 检测 500/500、识别 86/86

.\tkiln.bat check -c <任意配置>.yml   # 单配置快速建图摘要
```

---

## 10. Windows 注意事项

| 项 | 建议 |
|---|---|
| 线程 | `$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"` |
| `num_workers` | Train **0~4**、Eval **0**；≥10 易 `WinError 1455` / OpenBLAS error |
| `device` | 必须写 **`cuda:0`** 或 **`gpu:0,1`**；写 `'0'` 会解析到 CPU |
| 两训练并行 | 16GB 显存勿同时跑两个；串行 |
| 大图 OOM | 优先降 `batch_size_per_card`（如 4），不要先砍 imgsz |
| 进程残留 | 训练「结束即卡住」用 val 子集或加大 `eval_epoch_step`；残留 python 用 `Stop-Process` 清 |

详细坑位见 [`docs/FAQ.md`](FAQ.md)。

---

## 11. 权重与数据外置（仓库只放代码 / 配置 / label）

### 11.1 预训练权重

- 缓存：**`~/.torchkiln/pretrained/<name>.pth`**（Windows 自动隐藏）。
- 配置可直接写 ModelScope URL（自动下载+缓存）：

```yaml
Global:
  pretrained_model: https://www.modelscope.cn/models/ChaoII0987/TorchKiln/resolve/master/pretrained/PP-OCRv6_tiny_det.pth
```

- 加载规则：按**参数名 + 形状**匹配，不匹配跳过并打印 `missing / unexpected`。
- 上传命名（与配置名一致，**无** `_ptocr` / `_state` 后缀）：

| 文件 | 模型 |
|---|---|
| `<config_name>.pth` | OCR 17 个（PP-OCRv3/4/5/6 det/rec） |
| `PP-LCNet_x1_0_pedestrian_attribute.pth` / `..._vehicle_attribute.pth` | 行人 / 车辆属性 |
| `plate_detect.pth` / `plate_rec_color.pth` | 车牌检测 / 识别 |
| `yolo11n.pth` 等 | 上游 YOLO state_dict（已与官方 `.pt` 逐位对齐） |

环境变量（**名字保持 `PYTORCHOCR_*` 不变**）：

| 变量 | 作用 |
|---|---|
| `PYTORCHOCR_HOME` | 缓存根（默认 `~/.torchkiln`） |
| `PYTORCHOCR_PRETRAINED_DIR` | 直接指定权重目录 |
| `PYTORCHOCR_AUTO_DOWNLOAD=0` | 关自动下载 |
| `PYTORCHOCR_ALLOW_LOCAL_REPO=0` | 不复用仓库内 `_downloads/official` |

### 11.2 数据集

```powershell
tkiln data list                 # 就绪性
tkiln data get <name>           # ModelScope → datasets/<name>/
tkiln data get --all --force
python tools/make_demo_data.py --all          # 本地占位图（仅自检）
python tools/make_format_examples.py          # 各任务标签格式模板
```

约定：`datasets/<name>/{train.txt, val.txt, images/}`；YOLO 系标注在 `labels/<split>/`，
sem 在 `masks/`、depth 在 `depth/`。清单与上传：`datasets/manifest.yml`、`datasets/README.md`。
格式权威文档：[`docs/DATASET_FORMATS.md`](DATASET_FORMATS.md)。

### 11.3 评估频率相关字段（避免误判「卡住」）

| 字段 | 含义 |
|---|---|
| `eval_batch_step: [起始, 间隔]` | 按 global_step 评估；`[0,0]` 关闭训练中评估 |
| `eval_epoch_step` | 按 epoch 评估（大图/慢评估时设大） |
| `print_batch_step` | 日志打印间隔 |

训练期不评估（只末轮评一次）可显著提速：把 `eval_epoch_step` / `eval_batch_step` 设大。

---

## 12. 相关文档

| 文档 | 内容 |
|---|---|
| [`USER_GUIDE.md`](USER_GUIDE.md) | **小白完整手册**（推荐先看） |
| [`DATASET_FORMATS.md`](DATASET_FORMATS.md) | 各任务目录/标签格式 + 从零接入步骤 |
| [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) | 配置字段全解 |
| [`MODEL_ZOO.md`](MODEL_ZOO.md) | 模型总表与指标 |
| [`FAQ.md`](FAQ.md) | 环境 / Windows / 权重坑位 |
| [`../datasets/README.md`](../datasets/README.md) | 内置数据集清单与拉取命令 |
| [`../README.md`](../README.md) | 项目总览与 OCR 侧细节 |
