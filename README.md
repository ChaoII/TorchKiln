# TorchKiln —— 统一 PyTorch 训练平台

> 原名 **PytorchOCR**（仓库/包/CLI 已于 2026-09 改为 TorchKiln / `torchkiln` / `tkiln`，详见 [AGENTS.md](AGENTS.md) 改名节）。
> 命令入口：仓库根目录 `tkiln.bat` / `tkiln`，或 `python -m torchkiln`，或 `pip install -e .` 后的 `tkiln`。

一个平台、四条产品线,**不依赖 PaddleOCR / PaddleX / ultralytics 的运行时**(仅用它们的权重做对齐与初始化):

| 产品线 | 内容 | 配置 |
|---|---|---|
| **OCR** | PaddleOCR 复现:PP-OCRv**2/3/4/5/6** 检测 + 识别(17 个模型,与 Paddle 逐层权重对齐) | `configs/ocr/det/*.yml`(9)、`configs/ocr/rec/*.yml`(8) |
| **YOLO 家族** | 自研实现,7 个任务(检测/实例分割/旋转框/关键点/分类/语义分割/深度)+ **上游 YAML 图模型**(v3/v5/v6/v8/v9/v10/v11/v12/v26 共 53 个可训练配置) | `configs/yolo/*.yml`(53) |
| **车牌** | 上游 `we0091234` 项目原版复刻:yolov5-lite + 4 角点检测、CNN+CTC+颜色识别 | `configs/plate/*.yml`(2) |
| **属性识别** | PaddleX 复刻:PP-LCNet_x1_0 多标签(行人 26 属性 / 车辆 19 属性) | `configs/attr/*.yml`(2) |

能力:**训练**(断点续训 / 官方预训练微调 / 多卡 / AMP / EMA / 梯度累积 / 冻结)、
**评估**、**推理**、**导出**(pth / TorchScript / ONNX / onnxslim)、**权重转换**(Paddle `.pdparams`、上游 `.pt` → torch)。
公共平台层 `ptcore/` 统一配置、优化器、精度与训练循环;换数据集/字典/输出目录/超参一律用 `-o` 覆盖,不改 yml。

## 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | **⭐ 小白完整手册**:训练/评估/推理/导出全参数、默认值、引号规则、模型对照、报错速查（**新手先看这篇**） |
| [`docs/STRUCTURE.md`](docs/STRUCTURE.md) | **项目结构**:完整目录树 + 每个模块的职责 + 依赖方向 |
| [`docs/TASK_ARCHITECTURE.md`](docs/TASK_ARCHITECTURE.md) | **任务体系**:两级分派、TaskAdapter 契约、各任务组件落点、统一结构与新增任务流程 |
| [`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md) | **模型总表**:四产品线全部模型/配置/数据格式/指标/对齐证据 |
| [`docs/TRAINING.md`](docs/TRAINING.md) | **训练闭环**:训练/评估/导出/推理命令、训练特性、权重转换、自检 |
| [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) | **配置项全解** + 各任务标签格式速查 |
| [`docs/FAQ.md`](docs/FAQ.md) | **FAQ**:环境、Windows 坑位、复现性、权重加载、车牌/属性细节 |
| 本文件 §4~§13 | OCR 侧完整细节(数据准备、日志格式、断点续训、结构详解...) |
| 本文件 §14 / §15 | YOLO 七任务 / YAML 图模型与 ultralytics 兼容性 |
| 本文件 §16 / §17 | 车牌(检测+识别)/ 属性识别(行人/车辆) |

---

## 快速开始(3 步)

```powershell
conda activate ptocr
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"     # Windows 建议设置

# 0) 统一 CLI（等价于 python -m torchkiln / python tools/*.py）
.\tkiln.bat --help                                             # 或 pip install -e . 后直接用 tkiln

# 1) 用命令行指向你自己的数据集并训练（不改 yml；每个覆盖项一个 -o）
python tools/train.py -c configs/ocr/det/PP-OCRv5_mobile_det.yml -o Train.dataset.data_dir=D:/mydata/det -o Train.dataset.label_file_list=D:/mydata/det/train.txt -o Eval.dataset.data_dir=D:/mydata/det -o Eval.dataset.label_file_list=D:/mydata/det/val.txt -o Global.save_model_dir=./output/my_det

# 2) 评估
python tools/eval.py -c configs/ocr/det/PP-OCRv5_mobile_det.yml --weights output/my_det/best_accuracy.pth -o Eval.dataset.data_dir=D:/mydata/det -o Eval.dataset.label_file_list=D:/mydata/det/val.txt

# 3) 推理
python tools/infer/predict_det.py -c configs/ocr/det/PP-OCRv5_mobile_det.yml --weights output/my_det/best_accuracy.pth --input my.jpg --output out.jpg
```

识别只需多一个字典：`-o Global.character_dict_path=D:/mydata/rec/dict.txt`。详见 [4.4](#44-用命令行指定自己的数据集推荐不用改-yml)。

其它产品线同样用法:

```powershell
python tools/train.py -c configs/yolo/yolov8-det.yml                 # YOLO(53 个配置任选)
python tools/train.py -c configs/plate/plate_det.yml `                 # 车牌检测(上游权重微调)
  -o Global.pretrained_model=_downloads/plate/plate_detect.pth
python tools/train.py -c configs/attr/vehicle_attribute.yml `          # 车辆属性
  -o Global.pretrained_model=~/.torchkiln/pretrained/PP-LCNet_x1_0_vehicle_attribute.pth

# 改完代码先自检(全量:每配置 1 步训练 + 1 步评估)
python tools/smoke_all.py          # 期望 76 OK, 0 FAIL
python tools/check_graph_build.py  # 期望 53 configs: 53 OK, 0 FAIL
```

---

## 目录

1. [支持的模型](#1-支持的模型)（OCR）
2. [环境准备](#2-环境准备)
3. [项目结构](#3-项目结构)（详见 `docs/STRUCTURE.md`）
4. [训练数据怎么准备](#4-训练数据怎么准备)
5. [训练](#5-训练)
6. [断点续训 / 从预训练微调](#6-断点续训--从预训练微调)
7. [评估](#7-评估)
8. [推理](#8-推理)
9. [模型导出](#9-模型导出)
10. [Paddle 权重转换](#10-paddle-权重转换)
11. [配置项全解](#11-配置项全解)（详见 `docs/CONFIG_REFERENCE.md`）
12. [常见问题 FAQ](#12-常见问题-faq)（详见 `docs/FAQ.md`）
13. [附：模型结构详解](#13-附模型结构详解)
14. [YOLO 七任务](#14-yolo-任务torchkiln)
15. [YAML 图模型 + ultralytics 兼容](#15-yaml-图模型--ultralytics-生态兼容)
16. [车牌(检测 + 识别)](#16-车牌检测--识别)
17. [属性识别(行人 / 车辆)](#17-属性识别行人--车辆)
18. [项目结构总览与文档索引](#18-项目结构总览与文档索引)

---

## 1. 支持的模型

共 **17** 个配置（`configs/ocr/det/*.yml` 9 个，`configs/ocr/rec/*.yml` 8 个）。

### 1.1 文本检测 det

| 模型 | Backbone | Neck | Head | 参数量(本项目) | 官方 Hmean(%) | 官方模型大小(MB) |
|---|---|---|---|---|---|---|
| PP-OCRv3_mobile_det | MobileNetV3 (scale0.5, large, 无SE) | RSEFPN (96) | DBHead (k=50) | 0.60M | 2.x 代模型¹ | — |
| PP-OCRv3_server_det | ResNet_vd (50) | LKPAN (256) | DBHead (kernel[7,2,2]) | 31.59M | 2.x 代模型¹ | — |
| PP-OCRv4_mobile_det | PPLCNetV3 (0.75, det) | RSEFPN (96) | DBHead (k=50, fix_nan) | 3.52M | 63.8 | 4.7 |
| PP-OCRv4_server_det | PPHGNet_small | LKPAN (256, intracl) | PFHeadLocal (large, fix_nan) | 28.46M | 69.2 | 109 |
| PP-OCRv5_mobile_det | PPLCNetV3 (0.75, det) | RSEFPN (96) | DBHead (k=50, fix_nan) | 3.52M | 79.0 | 4.7 |
| PP-OCRv5_server_det | PPHGNetV2_B4 | LKPAN (256, intracl) | PFHeadLocal (large) | 26.29M | 83.8 | 84.3 |
| PP-OCRv6_tiny_det | PPLCNetV4 (tiny) | RepLKFPN (64, dilk5) | DBHead (aux=64, fix_nan) | 0.45M | 80.6* | 1.9 |
| PP-OCRv6_small_det | PPLCNetV4 (small) | RepLKFPN (96, dilk7) | DBHead (aux=96, fix_nan) | 2.51M | 84.1* | 9.6 |
| PP-OCRv6_medium_det | PPLCNetV4 (medium) | RepLKPAN (256, intracl) | DBHead (aux=256, fix_nan) | 15.79M | 86.2* | 59.4 |

> ¹ PP-OCRv3 属 PaddleOCR 2.x 时代模型，官方 3.x 文档未列指标。
> `*` PP-OCRv6 指标基于官方内部多场景评估集，与 v4/v5 的通用评估集不同，**不可直接横向比较**。

### 1.2 文本识别 rec

| 模型 | Backbone | Neck/Encoder | Head | 参数量(本项目)² | 官方 Avg Acc(%) | 官方模型大小(MB) |
|---|---|---|---|---|---|---|
| PP-OCRv3_mobile_rec | MobileNetV1Enhance (0.5) | SVTR (dims64) | MultiHead(CTC + SAR) | 27.51M | 2.x 代模型¹ | — |
| PP-OCRv4_mobile_rec | PPLCNetV3 (0.95) | SVTR (dims120, depth2) | MultiHead(CTC + NRTR384) | 21.35M | 78.74 | 10.5 |
| PP-OCRv4_server_rec | PPHGNet_small | SVTR | MultiHead(CTC + NRTR384) | 39.11M | 85.19 | 173 |
| PP-OCRv5_mobile_rec | PPLCNetV3 (0.95) | SVTR | MultiHead(CTC + NRTR384) | 21.35M | 81.29 | 16 |
| PP-OCRv5_server_rec | PPHGNetV2_B4 | SVTR | MultiHead(CTC + NRTR384) | 47.50M | 86.38 | 81 |
| PP-OCRv6_tiny_rec | PPLCNetV4 (tiny) | reshape + mid80 | MultiHead(CTC + NRTR384) | 15.71M | 73.5* | 4.4 |
| PP-OCRv6_small_rec | PPLCNetV4 (small) | LightSVTR (dims120) | MultiHead(CTC + NRTR384) | 18.53M | 81.3* | 20.4 |
| PP-OCRv6_medium_rec | PPLCNetV4 (medium) | LightSVTR (dims192) | MultiHead(CTC + NRTR512) | 40.82M | 83.2* | 73.3 |

> ² 识别模型参数含训练期辅助分支（NRTR/SAR）；实际部署只需 CTC 分支。

---

## 2. 环境准备

本项目使用独立 conda 环境 `ptocr`（Python 3.12 + PyTorch GPU），避免污染其它环境。

```powershell
# 创建环境
conda create -n ptocr python=3.12 -y
conda activate ptocr

# 安装 PyTorch（按你的 CUDA 选择 index-url；示例为 cu132）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 安装其余依赖
pip install opencv-python shapely pyclipper albumentations pyyaml tqdm editdistance rapidfuzz scipy scikit-image six
# 可选：导出 ONNX / 优化用
pip install onnx onnxscript onnxslim onnxruntime
```

验证：
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 重要：Windows 上必须开多进程数据加载，但**不能开太多**

**训练慢、GPU 利用率低，几乎都是因为 `num_workers=0`。** `num_workers=0` 时数据读取和 GPU 计算串行，GPU 有大半时间在等 CPU。（实测见 [5.7](#57-训练太慢gpu-利用率低怎么办)）

但 Windows 下**每个 worker 进程都要加载 torch + CUDA 的 DLL（约 3.5~4 GB 提交内存）**，worker 太多会耗尽**提交内存（RAM + 页面文件）**，报错并卡死：

```
OSError: [WinError 1455] 页面文件太小，无法完成操作
Error loading "...\torch\lib\cufft64_12.dll"
```
或
```
OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
```

**本机实测**：内存 32GB + 页面文件 32GB ⇒ 提交上限 **63.8GB**，而系统基线已占 **37.5GB**，只剩约 **26GB** 给训练。每个 worker 约 3.5~4GB ⇒ **最多约 4 个 worker（含主进程）**。

所以：

- **默认已设成 `Train.num_workers=4`、`Eval.num_workers=0`**（4 个 worker），实测稳定且吞吐与 6/8 个 worker 相同（数据加载已被完全隐藏）。
- `Eval.num_workers=0` 是故意的：评估时如果再去 spawn worker，很容易在这一刻把内存顶爆（表现为进度条刚出现就 `OpenBLAS error`）。
- **想用更多 worker，先加大页面文件**（`系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存`，设成 64GB 或"系统管理"），把提交上限抬上去再加。
- 训练前建议确认没有**残留 python 进程**占内存（崩溃后 worker 可能不会退出）：
  ```powershell
  Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,@{n='RAM_MB';e={[int]($_.WorkingSet64/1MB)}}
  (Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue/1GB   # 当前提交内存
  ```
- 建议顺手设置（避免每个 worker 内部线程过多）：
  ```powershell
  $env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
  ```

---

## 3. 项目结构

> **权威目录树见 [`docs/STRUCTURE.md`](docs/STRUCTURE.md)**(覆盖 OCR / YOLO / 车牌 / 属性 全部模块与职责)。
> 下面只展开 OCR 侧结构。

```
TorchKiln/
├─ configs/
│  ├─ ocr/det/*.yml           # 9 个检测配置（可直接训练）
│  ├─ ocr/rec/*.yml           # 8 个识别配置
│  ├─ yolo/*.yml              # 53 个 YOLO 图模型配置（小写+横线）
│  ├─ plate/ attr/ action/ video/   # 车牌 / 属性 / 动作 / 视频
│  ├─ local/                  # 业务实验（gitignore，不入库）
│  └─ _parity/                # 对照验收 / 消融（gitignore，不入库；debug 在其下）
├─ ptcore/                   # 与任务无关的训练平台（config/optimizer/ema/precision）
│  ├─ task.py                # TaskAdapter 抽象(11 个钩子)
│  └─ trainers/              # 每个任务一个 trainer,共享 base.py 的训练循环
│     ├─ base.py             #   BaseTrainer(训练/评估/EMA/断点续训/多卡/日志/保存)
│     ├─ detect.py segment.py pose.py classify.py obb.py semantic.py depth.py
│     ├─ plate_det.py plate_rec.py attribute.py pose_action.py video_cls.py ocr.py
│     └─ __init__.py         #   TRAINER_REGISTRY / get_trainer() / build_trainer()
├─ torchkiln/              # YOLO 家族（7 任务） + YAML 图模型
│  ├─ nn/modules.py          # 模块库 + 注册表（Conv/C2f/C3k2/Detect/OBB/Segment/... 60+）
│  ├─ nn/graph.py            # YAML 解析器 + GraphModel（负数索引/Concat/缩放/多尺度）
│  ├─ cfg/models/            # 50 个上游 YAML（v3/v5/v6/v8/v9/v10/11/12/26）
│  ├─ data/augment.py        # HSV/Affine/mosaic4/mosaic9/mixup/copy_paste/random_erasing
│  ├─ tasks/                 # 每任务一个文件(classify/detect/obb/segment/pose/semantic/depth/
│  │                         #   plate_det/plate_rec/attribute/pose_action/video_cls + _base/_cls)
│  └─ det/ seg.py pose.py sem.py depth.py task.py trainer.py
├─ torchkiln/ocr/
│  ├─ modeling/              # 网络定义（与 Paddle 权重键对齐）
│  │  ├─ backbones/          # PPLCNetV3/V4, MobileNetV3, ResNet_vd, PPHGNet(_V2), MobileNetV1Enhance...
│  │  ├─ necks/              # RSEFPN, LKPAN, RepLKFPN, RepLKPAN, DBFPN, SVTR/LightSVTR
│  │  ├─ heads/              # DBHead, PFHeadLocal, MultiHead, CTCHead, Transformer(NRTR), SARHead
│  │  └─ architectures/      # BaseModel（按配置组装）
│  ├─ data/                  # 数据管线（从 PaddleOCR 移植）
│  │  ├─ imaug/              # DecodeImage/DetLabelEncode/CopyPaste/IaaAugment/EastRandomCropData/
│  │  │                      #   RandomCrop/ColorJitter/RandomPerspective/MakeBorderMap/MakeShrinkMap/
│  │  │                      #   RecResizeImg/RecAug/RecConAug/MultiLabelEncode...
│  │  ├─ simple_dataset.py   # 数据集
│  │  └─ paddle_batch_sampler.py  # 与 Paddle 一致的乱序采样器（含多卡分片）
│  ├─ losses/                # DBLoss 系列, CTCLoss, MultiLoss, NRTRLoss, SARLoss
│  ├─ metrics/               # DetMetric(hmean), RecMetric(acc)
│  ├─ optimizer/             # Adam + Cosine(warmup) + L2（复刻 Paddle 取 LR 时机）
│  ├─ postprocess/           # DBPostProcess, CTCLabelDecode（Paddle 原版移植）
│  ├─ utils/                 # config(-o 覆盖)、pretrained(权重解析/下载)、ema、precision、logging
│  └─ trainer.py             # 训练/评估/EMA/断点续训/多卡
├─ tools/
│  ├─ train.py               # 训练入口
│  ├─ eval.py                # 独立评估
│  ├─ export.py              # 模型导出（pth/TorchScript/ONNX/onnxslim）
│  ├─ download_pretrained.py # 预下载预训练权重到本地缓存
│  ├─ smoke_all.py           # 一键自检全部产品线配置（76 OK；跳过 _parity/local）
│  ├─ check_graph_build.py   # 只建图 + 前向：校验 53 个 YAML 图配置
│  ├─ gen_configs.py         # 从官方配置生成 configs/
│  ├─ infer/
│  │  ├─ predict_det.py      # 检测推理
│  │  ├─ predict_rec.py      # 识别推理
│  │  └─ predict_yolo.py     # YOLO 七任务推理（可视化）
│  └─ convert/
│     ├─ dump_all.py         # (paddlex 环境) 把 .pdparams 导出为 numpy pickle
│     ├─ dump_paddle_sd.py   # 单个权重导出
│     ├─ dump_ultralytics.py # (ultralytics 环境) 把上游 .pt 导出为纯张量 state_dict
│     └─ convert_all.py      # (ptocr 环境) pickle -> torch .pth
├─ datasets/                 # 示例数据集（det / rec）
├─ ~/.torchkiln/pretrained/      # 官方 .pdparams + 转换后的 *.pth
└─ output/                   # 训练/评估/导出产物
```

---

## 4. 训练数据怎么准备

目录约定：每个数据集一个文件夹，里面放 `train.txt`、`val.txt`（可选 `dict.txt`）以及图片。
**配置字段只有两个**：`Train/Eval.dataset.data_dir` + `label_file_list`（没有 `train_list`/`val_list`）。
完整命令链、路径语义、各任务 `-o` 对照见 [`docs/TRAINING.md`](docs/TRAINING.md) §3 与
[`docs/DATASET_FORMATS.md`](docs/DATASET_FORMATS.md) §0。

### 4.0 全任务一览（目录 + 标签 + 一条训练命令）

| 任务 | 清单/标签 | 必须额外 `-o`（示例） | 训练入口 |
|---|---|---|---|
| OCR det | `路径\tJSON四点` | `dataset.name=SimpleDataSet` + `transforms:` | `configs/ocr/det/*.yml` |
| OCR rec | `路径\t文本` + 字典 | `Global.character_dict_path=...` | `configs/ocr/rec/*.yml` |
| YOLO detect | `labels/`：`cls cx cy w h` | `Head.num_classes` + `names` | `configs/yolo/*.yml` |
| YOLO OBB | **9 字段四角点**（非 6 字段 angle） | `dataset.box_format=xywhr` | 同上 + obb 配置 |
| YOLO seg/pose/cls/sem/depth | 见 DATASET_FORMATS | `kpt_shape`/`mask_stride`/… | 同上 |
| 车牌 det/rec | detect 布局 / `c1..c7 颜色` | `kpt_shape=[4,2]` 等 | `configs/plate/*.yml` |
| 属性 | `路径 v1..vC` | `label_ratio` + `label_list` | `configs/attr/*.yml` |

```powershell
# 数据就绪检查 / 下载 / 格式模板
tkiln data list
tkiln data get plate_det_demo
python tools/make_format_examples.py          # 各任务 README.txt + train.txt 样板
python tools/make_demo_data.py --all          # 占位图(仅 smoke)

# 接到训练：只改 -o，不改 yml（YOLO 单类示例）
python tools/train.py -c configs/yolo/yolo11-det.yml `
  -o Train.dataset.data_dir=D:/mydata/det `
  -o Train.dataset.label_file_list=train.txt `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt `
  -o Architecture.Head.num_classes=1 `
  -o Train.dataset.names=[plate] `
  -o Global.save_model_dir=./output/my_y11
```

OBB：加载器 `box_format=xywhr` 时**每行必须 `cls` + 4 角点（9 个数）**，`<9` 整行丢弃；
6 字段 `cls cx cy w h angle` **不被接受**（`torchkiln/data/det.py`）。

### 4.1 检测（OCR det）数据集

**标注文件格式**：每行
```
图片相对路径<TAB>标注
```
标注是 **JSON 数组**（也支持 PaddleOCR 的 `x1,y1,x2,y2,...` 或 `###` 形式；示例数据集用 JSON）。每个文本框含 `transcription` 与 `points`（4 个点，顺时针或逆时针都行，代码内部会校正）。

```
images/train_1.jpg	[{"transcription": "Hello", "points": [[10,10],[100,10],[100,30],[10,30]]}, {"transcription": "###", "points": [[110,10],[150,10],[150,30],[110,30]]}]
```
- `transcription == "###"` 表示**忽略**该框（不参与 loss 与指标，通常用于模糊/无法辨认的文本）。
- `points` 至少 4 个点（支持多边形）。

目录示例：
```
datasets/ocr_det_dataset_examples/
├─ images/
│  ├─ train_1.jpg
│  └─ val_1.jpg
├─ train.txt
└─ val.txt
```
在配置里：
```yaml
Train:
  dataset:
    name: SimpleDataSet            # OCR 数据集逻辑名
    data_dir: datasets/ocr_det_dataset_examples
    label_file_list: [train.txt]
    transforms: [...]              # OCR 用 transforms 列表，不是 YOLO 的 transform
```
> `data_dir` + 标注中的相对路径 拼成最终图片路径；`label_file_list` 项已是绝对路径/已存在文件时直接用。

### 4.2 识别（rec）数据集

**标注文件格式**：每行
```
图片相对路径<TAB>文本
```
```
images/word_1.png	Genaxis Theatre
images/word_2.png	[06]
```
**字典文件**（字符集）：每行一个字符。本项目官方模型配套字典放在
`torchkiln/ocr/utils/ppocr_keys_v1.txt`（6623 字符）、`torchkiln/ocr/utils/dict/ppocrv5_dict.txt`、`ppocrv6_dict.txt`、`ppocrv6_tiny_dict.txt`。

配置里：
```yaml
Global:
  character_dict_path: datasets/ocr_rec_dataset_examples/dict.txt   # 你的字典
  use_space_char: true        # 是否把空格当作一个字符
  max_text_length: 25         # 最大文本长度（超过会被截断）
Train:
  dataset:
    name: SimpleDataSet
    data_dir: datasets/ocr_rec_dataset_examples
    label_file_list: [train.txt]
```

> **字典必须与模型输出维度一致**：`模型类别数 = len(字典) + 1(blank) + (1 若 use_space_char)`。
> 例如加载官方 v5 rec 权重，就必须用 `ppocrv5_dict.txt`（18383），否则加载会报 size 不匹配。

### 4.3 用官方示例数据集快速上手

项目内已解压好示例数据：
```
datasets/ocr_det_dataset_examples/    # ICDAR2015 风格，train 200 + val 50
datasets/ocr_rec_dataset_examples/    # 词条图，train ~4400 + val，字典=ppocr_keys_v1
```
YOLO/车牌/属性等占位与标签模板：
```
tkiln data list
python tools/make_demo_data.py --all
python tools/make_format_examples.py   # datasets/_format_examples/<task>/
```

### 4.4 用命令行指定自己的数据集（推荐，**不用改 yml**）

配置里的数据路径就是普通字段，可以被 `-o` 覆盖。日常换数据集、换字典、换输出目录，**都在命令行写，不要改 `configs/*.yml`**。

**OCR 检测：**
```powershell
python tools/train.py -c configs/ocr/det/PP-OCRv5_mobile_det.yml -o Train.dataset.data_dir=D:/mydata/det -o Train.dataset.label_file_list=D:/mydata/det/train.txt -o Eval.dataset.data_dir=D:/mydata/det -o Eval.dataset.label_file_list=D:/mydata/det/val.txt -o Global.save_model_dir=./output/my_det
```

**OCR 识别（多一个字典）：**
```powershell
python tools/train.py -c configs/ocr/rec/PP-OCRv5_mobile_rec.yml -o Global.character_dict_path=D:/mydata/rec/dict.txt -o Train.dataset.data_dir=D:/mydata/rec -o Train.dataset.label_file_list=D:/mydata/rec/train.txt -o Eval.dataset.data_dir=D:/mydata/rec -o Eval.dataset.label_file_list=D:/mydata/rec/val.txt -o Global.save_model_dir=./output/my_rec
```

**YOLO / 车牌 / 属性** 同一写法，只换 `-c` 与任务字段（见 [4.0](#40-全任务一览目录--标签--一条训练命令)）；相对路径写法：
`-o Train.dataset.label_file_list=train.txt`（相对 `data_dir`）。

要点：
- **一个覆盖项一个 `-o`**，不用引号、不用方括号（与 PaddleOCR 一致）。单个标注文件直接给路径即可；多个用逗号分隔：
  `-o Train.dataset.label_file_list=D:/a.txt,D:/b.txt`
- 路径不要含空格；若真含空格，则在 cmd/终端里给整个 `键=值` 加双引号：
  `"Train.dataset.data_dir=D:/my data/det"`
- 训练/评估启动时会打印实际生效路径，可用来核对：
  `Train dataset: data_dir=... label_file_list=[...]`
- `-o` 里写错**配置段名**（如 `Trainn.` 或带上了引号 `'Train.`）会**直接报错**并提示可用段名，不会再静默失效。
- 断点/续训/微调时，**数据路径同样这样覆盖**，不用动 yml。
- 其它所有可覆盖字段见 [5.2](#52--o-覆盖语法) 与 [11. 配置项全解](#11-配置项全解)。
- **四阶段完整命令**（check → train → val → predict → export）见 [`docs/TRAINING.md`](docs/TRAINING.md)。

---

## 5. 训练

### 5.1 基本命令

```powershell
# 检测（默认示例数据 + 指定输出目录）
python tools/train.py -c configs/ocr/det/PP-OCRv4_mobile_det.yml -o Global.save_model_dir=./output/my_det

# 识别
python tools/train.py -c configs/ocr/rec/PP-OCRv4_mobile_rec.yml -o Global.save_model_dir=./output/my_rec
```

`-c` 指定配置文件；`-o` 用于**覆盖**配置里的任意字段（见 5.2）。
**换成你自己的数据集不用改 yml**，直接 `-o` 覆盖数据路径，见 [4.4](#44-用命令行指定自己的数据集推荐不用改-yml)。

### 5.2 `-o` 覆盖语法

- 形式：**一个覆盖项写一个 `-o`**（和 PaddleOCR 一样，不用加引号）：
  `-o Global.epoch_num=20 -o Global.save_model_dir=./output/quick`
  也支持一次给多个（`-o a=1 b=2`），两种写法可以混用、都会生效。
- 键路径用 `.` 表示层级：`Global.save_model_dir`、`Train.dataset.data_dir`、`Train.loader.batch_size_per_card`。
- 值类型自动识别：`true/false`→bool；纯数字→int/float；`[..]`/`{..}`→list/dict；`null/none`→None；其它→字符串。
- **列表字段可以直接给单个值，不用方括号**：`-o Train.dataset.label_file_list=D:/train.txt`
  多个用逗号：`-o Train.dataset.label_file_list=D:/a.txt,D:/b.txt`
- **写错配置段名会直接报错**（不再静默失效），例如误写成 `Trainn.xxx` 或误加引号 `'Train.xxx`：
  ```
  ValueError: Invalid -o override: unknown config section(s):
    - 'Trainn...'  (did you mean 'Train..'?)
  Available sections: Global, Architecture, Loss, Optimizer, PostProcess, Metric, Train, Eval
  ```
- 只有**值里含空格**时才需要引号，且用双引号包住整个 `键=值`：`"Train.dataset.data_dir=D:/my data/det"`。

```powershell
python tools/train.py -c configs/ocr/det/PP-OCRv4_mobile_det.yml -o Global.epoch_num=20 -o Global.save_model_dir=./output/quick -o Global.device=gpu:0 -o Train.dataset.data_dir=D:/mydata/det -o Train.dataset.label_file_list=D:/mydata/det/train.txt -o Train.loader.batch_size_per_card=8 -o Optimizer.lr.learning_rate=0.001 -o Global.eval_batch_step=[0,100]
```

> 上面的写法在 **cmd / PowerShell / Bash 都能直接跑**（无引号最省事）。
> 想换行的话：cmd 用 `^`，PowerShell 用反引号 `` ` ``，Bash 用 `\`。

### 5.3 训练时会发生什么

每个 epoch：
1. 按 `Train.loader.batch_size_per_card` 取批；数据经过 `Train.dataset.transforms` 增广后送入模型。
2. 计算 loss → 反向 → 优化器更新 → LR 调度器更新（若开 EMA 则更新 EMA）。
3. 按 `Global.eval_batch_step` 的节奏在验证集评估（`[起始step, 间隔]`）；指标变好就保存 `best_accuracy.pth`。
4. 每 `Global.save_epoch_step` 个 epoch 存一次 `epoch_N.pth`；每轮结束存 `latest.pth`（含优化器/EMA 状态，供续训）。

产物（`Global.save_model_dir` 下）：
```
best_accuracy.pth     # 验证指标最好的模型（开 EMA 时是 EMA 权重）
latest.pth            # 最新状态（模型+优化器+调度器+EMA），用于续训
epoch_N.pth / final.pth
train.log             # 训练日志（控制台输出的完整记录）
config.yml            # ★ 本次实际生效的完整配置（含所有 -o 覆盖），可直接查看/复现
```

> 想确认"这次到底用了什么参数"，直接看 `save_model_dir/config.yml`，或在 `train.log` 里搜 `Config:` 开头的摘要行（模型/epoch/step/batch/lr/字典/后处理阈值/预训练等都会打印）。

### 5.4 常用“减负”覆盖（本机/小显存）

```powershell
# 小显存 / 冒烟测试
-o Global.epoch_num=1 -o Train.loader.batch_size_per_card=4
# 关 EMA（与 Paddle 默认一致）
-o Global.use_ema=false
```

### 5.5 多 GPU 训练

支持两种方式，通过 `Global.device` 与启动命令区分。

**方式一：DDP（官方推荐，命令行 `torchrun` 启动）**

让每个进程独占一张卡，进程组自动初始化（rank/world_size 由 `torchrun` 注入）：

```powershell
# Linux 单机 4 卡（NCCL 后端，效率最高）
torchrun --nproc_per_node=4 tools/train.py -c configs/ocr/det/PP-OCRv5_server_det.yml `
  -o Global.save_model_dir=./output/ddp_v5

# 指定用哪几张卡（每个 rank 一个）：gpu:0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/ocr/det/PP-OCRv5_server_det.yml `
  -o Global.device=gpu:0,1,2,3 Global.save_model_dir=./output/ddp_v5
```

**方式二：DataParallel（单进程多卡，Windows 友好）**

不启动 torchrun，直接把多张卡写进 `Global.device`：

```powershell
python tools/train.py -c configs/ocr/det/PP-OCRv5_mobile_det.yml `
  -o Global.device=gpu:0,1 Global.save_model_dir=./output/dp_v5
```

**多卡相关配置项**

| 配置 | 默认 | 说明 |
|---|---|---|
| `Global.device` | 空（用 `use_gpu`） | `cpu` / `gpu:0` / `gpu:0,1`（多卡）/ `cuda:0` |
| `Global.dist_backend` | 自动 | `nccl`（有则用）/ `gloo`；可用它强制指定 |
| `Global.dist_init_method` | `env://` | 进程组初始化方式；torchrun 用默认即可 |

**注意事项**

- **DDP 每卡的 `batch_size_per_card` 是“单卡批大小”**：总 batch = `batch_size_per_card × GPU 数`。想保持总 batch 不变就等比分小。
- **学习率**：训练步数按“单卡步数”计算（与 Paddle 一致），`warmup_epoch`/Cosine 的总步数也会相应变化；多卡放大总 batch 后，通常需要按比例调大 `Optimizer.lr.learning_rate`。
- **只有 rank0 写日志/存权重/评估**；`save_model_dir` 产物与单卡一致，checkpoint 无 `module.` 前缀，可直接用于评估/导出。
- **Windows 提示**：官方 Windows 版 PyTorch **不含 NCCL**，DDP 会退回 `gloo`（GPU 梯度经 CPU 通信，较慢）；且该构建可能缺少 libuv，此时 `torchrun` 会报 `use_libuv ... run with USE_LIBUV=0`。若遇到，可用方式二（DataParallel），或用 `-o Global.dist_init_method=file:///<路径> Global.device=cpu Global.dist_backend=gloo` 做功能验证。**要发挥多卡性能，建议在 Linux + NCCL 下用 DDP。**
- 本机目前只有 1 张卡，因此仓库内未做真实多卡速度实测；DDP 的接线、分片采样、单 rank 保存/评估已用 2 进程 + gloo 验证通过。

### 5.6 日志怎么看（与 PaddleOCR 同格式）

训练每 `print_batch_step` 步打一行：

```
epoch: [1/20], global_step: 10, lr: 0.000200, loss: 18.397704,
loss_shrink_maps: 5.580157, loss_threshold_maps: 1.602375, loss_binary_maps: 2.235097,
loss_cbn: 0.000000, loss_aux_maps_p4: 9.768128, loss_aux_maps_p3: 9.125267, loss_aux_maps_p2: 10.722172,
avg_reader_cost: 0.95422 s, avg_batch_cost: 1.13092 s, avg_samples: 8.00000,
ips: 7.07388 samples/s, eta: 0:00:16, max_mem_reserved: 5510 MB, max_mem_allocated: 4917 MB
```

| 字段 | 含义 |
|---|---|
| `global_step` | 已训练的总步数 |
| `lr` | 当前学习率（warmup 阶段会从 0 线性升到 `learning_rate`） |
| `loss` + 各分项 | 平滑窗口（`Global.log_smooth_window`，默认 20）内的平均值；det 会列出 `loss_shrink_maps/loss_threshold_maps/loss_binary_maps` 和 v6 的 `loss_aux_maps_*` |
| `avg_reader_cost` | 平均**数据读取**耗时；理想值接近 0（多进程预取生效）。若它和 `avg_batch_cost` 差不多大，说明 GPU 在等数据，见 [5.7](#57-训练太慢gpu-利用率低怎么办) |
| `avg_batch_cost` | 平均**单步计算**耗时 |
| `avg_samples` | 平均每批样本数（增广丢样本时会略小于 `batch_size`） |
| `ips` | 每秒样本数（吞吐），= `avg_samples / avg_batch_cost` |
| `eta` | 预计剩余训练时间 |
| `max_mem_reserved` / `max_mem_allocated` | 显存峰值：预留 / 实际占用（MB） |

评估时会先打印进度条，再打两行指标：

```
eval model::: 100%|██████████| 169/169 [00:16<00:00,  9.99img/s]
cur metric, precision: 0.47305389221556887, recall: 0.39303482587064675, hmean: 0.42934782608695654, fps: 19.108542229596594
best metric, hmean: 0.42934782608695654, is_float16: False, precision: 0.47305389221556887, recall: 0.39303482587064675, fps: 19.108542229596594, best_epoch: 2
```

- 进度条可用 `-o Global.show_eval_progress=false` 关掉。
- `cur metric` 是**本次**评估结果；`best metric` 是**历史最好**（每个 eval 都会打印，即使本次没刷新）。
- `best_epoch` 是取得最好指标时的 epoch，`is_float16` 表示模型是否为半精度。

启动时还会打印本次生效的参数摘要（同样写进 `train.log`）：

```
Config saved to <save_model_dir>/config.yml
Config: model=PP-OCRv6_tiny_det type=det algorithm=DB
Config: epochs=2 steps/epoch=25 batch=8 eval_batch=1 workers=0/0 seed=1024 device=cuda:0 use_ema=True distributed=False
Config: optimizer=Adam lr=0.001 warmup_epoch=2 l2=1e-06
Config: postprocess thresh=0.2 box_thresh=0.4 unclip=1.4 max_candidates=3000
Config: head=DBHead backbone=PPLCNetV4 neck=RepLKFPN
Config: pretrained=... checkpoints=None save_dir=output/cfg_test
```
（识别模型还会多一行 `Config: dict=... num_classes=... max_text_length=... use_space_char=...`）

### 5.7 训练太慢、GPU 利用率低怎么办

**先看日志里这两个值**：`avg_reader_cost`（等数据）和 `avg_batch_cost`（GPU 计算）。
- 若 `avg_reader_cost` 和 `avg_batch_cost` 差不多大 → **GPU 有一半时间在等数据**，就是这个问题。
- 若 `avg_reader_cost` 很小（毫秒级）而 `avg_batch_cost` 很大 → 瓶颈在 GPU 计算本身，见下面"其它可调项"。

**最有效的三招**

1. **开多进程数据加载**（默认已经是 `num_workers=4`）。`num_workers=0` 时读取和计算完全串行。
2. **`persistent_workers=True`**（已默认开启）：worker 进程只在训练开始时创建一次，不再每个 epoch 重新 spawn。
3. **控制 worker 数量**：本机提交内存只剩约 26GB，4 个 worker 是安全上限；更多就要先加大页面文件（见 [第 2 节](#重要windows-上必须开多进程数据加载但不能开太多)）。

**实测**（PP-OCRv6_tiny_det，200 张图，`batch_size=8`，RTX 4060 Ti）：

| 配置 | 稳态 `avg_reader_cost` | 稳态 `avg_batch_cost` | GPU 利用率 | epoch 耗时 |
|---|---|---|---|---|
| `num_workers=0` | 1.18 s | 0.67 s | ~36% | 44 s |
| `num_workers=4`（persistent） | ~0.001 s | 0.67 s | **99~100%** | **17.4 s** |

> 提升 **2.5 倍**。注意第一个 epoch 会慢一些（要 spawn worker + 预热），从第二个 epoch 起是稳态。

**和 PaddleX 同数据对比**（你自己那份 `det_checkpoint/1/train.log`，674 张 2560×1440 的图）：

| | `avg_reader_cost` | `avg_batch_cost` | ips |
|---|---|---|---|
| PaddleX | 1.73 s | 2.02 s | 3.97 samples/s |
| 本项目 | 0.001 s | 0.673 s | 11.88 samples/s |

本项目约快 **5 倍**（数据加载被 worker 完全隐藏，GPU 打满）。

**其它可调项**

| 手段 | 怎么做 | 说明 |
|---|---|---|
| 增大 batch | `-o Train.loader.batch_size_per_card=16` | 提高 GPU 占用；注意显存（v6 det 在 640×640 下 batch16 约需 11GB） |
| cuDNN 自动调优 | `$env:PYTORCHOCR_CUDNN_BENCHMARK="1"` | 对固定输入尺寸可能更快；会略微改变数值，默认关闭 |
| 增加预取 | `-o Train.loader.prefetch_factor=8` | 每个 worker 预取的批数，默认 4 |
| 关 EMA | `-o Global.use_ema=false` | EMA 每步都要遍历全部参数，能省一点 |
| 关评估进度条 | `-o Global.show_eval_progress=false` | 影响很小，只是少点 IO |
| 换输入尺寸 | 改 `Train.dataset.transforms` 的 crop 大小 | det 默认 640×640；DB 头输出分辨率与它平方相关，是主要计算量 |

---


## 6. 断点续训 / 从预训练微调

### 6.1 预训练权重：写 URL，首次自动下载并缓存

**和 PaddleOCR 一样**，配置里直接写权重 URL；加载时若本地没有就下载到用户目录的隐藏缓存，之后一直复用（不会重复下载）。

17 个 yml 的 `pretrained_model` 默认就是对应的 ModelScope URL，例如：

```yaml
Global:
  pretrained_model: https://www.modelscope.cn/models/ChaoII0987/TorchKiln/resolve/master/pretrained/PP-OCRv6_tiny_det.pth
```

首次训练时：

```
Downloading pretrained weights to C:\Users\<你>\.torchkiln\pretrained\PP-OCRv6_tiny_det.pth
Saved pretrained weights to C:\Users\<你>\.torchkiln\pretrained\PP-OCRv6_tiny_det.pth
Pretrained source: https://.../PP-OCRv6_tiny_det.pth -> C:\Users\<你>\.torchkiln\pretrained\PP-OCRv6_tiny_det.pth
Loaded pretrained: ... (missing=0 unexpected=0)
```

第二次及以后：

```
Pretrained weights already cached: C:\Users\<你>\.torchkiln\pretrained\PP-OCRv6_tiny_det.pth
```

> 参考实现：PaddleOCR 的 `ppocr/utils/network.py::maybe_download_params()` 也是把 URL 下到 `~/.paddleocr/models/`（带进度条、3 次重试）。本项目等价实现见 `torchkiln/ocr/utils/pretrained.py`，缓存目录为 `~/.torchkiln/ocr/pretrained/`（Windows 下自动隐藏）。多卡时只由 rank0 下载，其余 rank 等待文件出现。

`pretrained_model` 支持的写法：

| 写法 | 行为 |
|---|---|
| `https://.../xx.pth` | **推荐**。没有则下载到缓存，有则直接用 |
| `E:/dx_ocr/det_checkpoint/2/best_accuracy.pth` | 用本地文件（自己训练的权重，见下） |
| `PP-OCRv6_tiny_det` | 裸名字：查缓存 → 没有就按官方 URL 下载 |
| `null` | 从零训练 |

常用操作：

```powershell
# 从零训练（不要预训练）
python tools/train.py -c configs/ocr/det/PP-OCRv6_tiny_det.yml -o Global.pretrained_model=null

# 提前把权重全部下载好（或指定几个）
python tools/download_pretrained.py
python tools/download_pretrained.py PP-OCRv6_tiny_det PP-OCRv5_mobile_rec
python tools/download_pretrained.py --list        # 查看缓存与其实际命中的路径

# 评估 / 导出 / 推理的 --weights 同样支持 URL、路径、名字
python tools/eval.py -c configs/ocr/det/PP-OCRv6_tiny_det.yml --weights PP-OCRv6_tiny_det
python tools/export.py -c configs/ocr/det/PP-OCRv6_tiny_det.yml --weights PP-OCRv6_tiny_det --save-dir output/exp
python tools/infer/predict_rec.py -c configs/ocr/rec/PP-OCRv5_mobile_rec.yml --weights PP-OCRv5_mobile_rec --input xx.jpg
```

**用自己训练出来的权重**（路径形式，最常用）

`pretrained_model` 也可以直接给**路径**——例如"拿上次训练最好的模型再训"：

```powershell
python tools/train.py -c configs/ocr/det/PP-OCRv6_tiny_det.yml -o Global.pretrained_model=E:/dx_ocr/det_checkpoint/2/best_accuracy.pth -o Global.save_model_dir=E:/dx_ocr/det_checkpoint/3
```

- `best_accuracy.pth` / `final.pth` / `epoch_N.pth`（纯权重）和 **`latest.pth`（含优化器/EMA，会自动取其中的 `model`）** 都能这么用 —— 实测四种来源都是 `missing=0 unexpected=0`。
- 想接着上次的**完整状态**继续（含优化器、学习率进度、epoch 数）→ 用 `Global.checkpoints=<.../latest.pth>`，见 [6.2 断点续训](#62-断点续训)。
- 注意**路径写错会直接报错**，不会静默变成从零训练：
  ```
  FileNotFoundError: Pretrained weights file not found: E:/dx_ocr/nope/best.pth
    Pass an existing .pth path, or just the model name (e.g. PP-OCRv6_tiny_det) ...
  ```
- 架构必须和配置一致（同模型间才能完整加载）；不同模型间只会加载形状相同的部分并打印 `missing/unexpected`。

**缓存位置 / 环境变量**

| 变量 | 作用 |
|---|---|
| `PYTORCHOCR_HOME` | 改缓存根目录（默认 `~/.torchkiln`） |
| `PYTORCHOCR_PRETRAINED_DIR` | 直接指定权重目录（默认 `<home>/pretrained`） |
| `PYTORCHOCR_AUTO_DOWNLOAD=0` | 关掉自动下载（只查本地，找不到就从零训练） |
| `PYTORCHOCR_ALLOW_LOCAL_REPO=0` | 不复用仓库内 `~/.torchkiln/pretrained/` 的同名权重 |

- Windows 下 `~/.torchkiln` 会被**自动设为隐藏文件夹**（资源管理器里默认看不到）。
- **缓存为空 ≠ 出错**：配置里虽然写的是 URL，但若 `~/.torchkiln/ocr/pretrained/` **或仓库的 `~/.torchkiln/pretrained/`** 里已有同名 `.pth`，就直接用它、不重复下载。这就是为什么在开发目录里跑训练时缓存可能是空的：
  ```
  Using local pretrained weights: E:\TorchKiln\_downloads\official\PP-OCRv6_tiny_rec.pth
  ```
- **想把缓存真正填满**（例如想脱离仓库运行、或验证下载）：
  ```powershell
  python tools/download_pretrained.py                       # 全部下载进缓存
  python tools/download_pretrained.py PP-OCRv6_tiny_det     # 只下一个
  python tools/download_pretrained.py --list                # 看每个名字实际解析到哪个文件
  ```
  该工具**总是下载到缓存**，不会复用 `~/.torchkiln/pretrained` 里的副本。
- 也可以直接禁掉本地复用：`set PYTORCHOCR_ALLOW_LOCAL_REPO=0`，或 `set PYTORCHOCR_HOME=<空目录>`。

**识别模型注意字典**（权重和字典必须匹配）；例如：
```powershell
python tools/train.py -c configs/ocr/rec/PP-OCRv5_mobile_rec.yml -o Global.character_dict_path=./torchkiln/ocr/utils/dict/ppocrv5_dict.txt -o Global.save_model_dir=./output/finetune_rec
```

### 6.2 断点续训

用 `Global.checkpoints` 指向**上次训练目录里的 `latest.pth`**：

```powershell
python tools/train.py -c configs/ocr/det/PP-OCRv4_mobile_det.yml -o Global.checkpoints=./output/my_det/latest.pth -o Global.save_model_dir=./output/my_det
```

会恢复 **模型 + 优化器 + 学习率调度 + EMA + global_step + best_metric/best_epoch + epoch_num**，并**从下一个 epoch 继续**（不会重跑已训过的 epoch）：

```
Restored epoch_num=2 from latest.pth (pass -o Global.epoch_num=... to override).
Resumed from ./output/my_det/latest.pth (epoch=4, global_step=196, best_hmean=0.73000 @epoch 4)
Continue training from epoch 5 to 6.
```

**`epoch_num` 会自动跟随 checkpoint**：当初用 `-o Global.epoch_num=100` 训的，续训时**不写 `epoch_num` 也会恢复成 100**（不会变回 yml 里的默认值）。
只有你**显式**传 `-o Global.epoch_num=...` 时才以命令行为准。

**想多训几轮**：把 `Global.epoch_num` 调大即可（余弦退火会用**新的总 epoch 数**重新拉长曲线，与 Paddle 一致）：

```powershell
# 之前训了 20 轮，现在想训到 40 轮
python tools/train.py -c configs/ocr/det/PP-OCRv4_mobile_det.yml -o Global.epoch_num=40 -o Global.checkpoints=./output/my_det/latest.pth -o Global.save_model_dir=./output/my_det
```

若 checkpoint 已完成（`epoch >= epoch_num`）且你没调大 `epoch_num`，会直接提示而不重复训练：

```
Checkpoint already trained 2/2 epochs; nothing to do. Pass -o Global.epoch_num=<larger> to train more.
```

> 注意区分：
> - **续训**用 `Global.checkpoints=<.../latest.pth>`（带优化器/EMA/轮数 等完整状态）。
> - 只想拿某个权重**微调/重新开训**，用 `Global.pretrained_model=<.../best_accuracy.pth>`（只加载权重，优化器和步数从 0 开始）。


---

## 7. 评估

```powershell
python tools/eval.py -c configs/ocr/det/PP-OCRv4_mobile_det.yml `
  --weights ./~/.torchkiln/pretrained/PP-OCRv4_mobile_det.pth
```
- `-c` 配置（决定验证集与后处理/指标）
- `--weights` 要评估的 `.pth`（不填则用配置里的 `Global.pretrained_model`）
- `-o` 同样可覆盖任意字段（包括验证集路径）

指标：
- 检测：`precision / recall / hmean`（`hmean` 为主指标）
- 识别：`acc / norm_edit_dis`

> 说明：评估用 `Eval.dataset.transforms`（检测走 `DetResizeForTest`，识别走 `RecResizeImg`），保证与训练时的预处理一致。

---

## 8. 推理

### 8.1 检测（出框 + 可视化）

```powershell
python tools/infer/predict_det.py `
  -c configs/ocr/det/PP-OCRv4_mobile_det.yml `
  --weights ./~/.torchkiln/pretrained/PP-OCRv4_mobile_det.pth `
  --input datasets/ocr_det_dataset_examples/images/val_img_61.jpg `
  --output output/vis.jpg `
  --device cuda:0
```
参数：

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `-c/--config` | 是 | — | 配置 yaml |
| `--weights` | 是 | — | `.pth` 权重 |
| `--input` | 是 | — | 图片路径 |
| `--output` | 否 | `output/det_result.jpg` | 可视化结果保存路径 |
| `--device` | 否 | `cuda:0` | 设备（`cpu`/`cuda:0`） |
| `-o` | 否 | `[]` | 覆盖配置 |

### 8.2 识别（出文本）

```powershell
python tools/infer/predict_rec.py `
  -c configs/ocr/rec/PP-OCRv4_mobile_rec.yml `
  -o Global.character_dict_path=./torchkiln/ocr/utils/ppocr_keys_v1.txt `
  --weights ./~/.torchkiln/pretrained/PP-OCRv4_mobile_rec.pth `
  --input datasets/ocr_rec_dataset_examples/images/val_word_1.png `
  --device cuda:0
```
输出形如：`... -> JOINT`

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `-c/--config` | 是 | — | 配置 yaml |
| `--weights` | 是 | — | `.pth` 权重 |
| `--input` | 是 | — | 图片路径 |
| `--device` | 否 | `cuda:0` | 设备 |
| `-o` | 否 | `[]` | 覆盖（识别**必须**指定 `Global.character_dict_path`，且与权重匹配） |

---

## 9. 模型导出

```powershell
$env:PYTHONIOENCODING="utf-8"   # 本机 GBK 控制台建议设置，避免 torch.onnx 打印报编码 warning

python tools/export.py -c configs/ocr/det/PP-OCRv4_mobile_det.yml `
  --weights ./~/.torchkiln/pretrained/PP-OCRv4_mobile_det.pth `
  --save-dir output/export/v4_det `
  --fuse --onnx --slim
```

产出：
```
save_dir/
├─ inference.pth      # 推理权重（state_dict，可选已融合）
├─ inference.yml      # 推理配置（配置副本）
├─ model.pt           # TorchScript（torch.jit.load 可推理）
├─ model.onnx         # ONNX（可能带 model.onnx.data 外部权重）
└─ model_slim.onnx    # onnxslim 优化后的单文件 ONNX（加 --slim）
```

参数：

| 参数 | 默认 | 影响 |
|---|---|---|
| `-c/--config` | — | 配置（决定网络结构与输入尺寸） |
| `--weights` | — | 训练产物 `.pth` |
| `--save-dir` | — | 输出目录 |
| `--fuse` | 关 | 调用各模块 `rep()`：把 `PPLCNetV3/V4`、`RepLKFPN/RepLKPAN` 的**多分支 + BN 融合为单卷积**。加速、减算子；实测融合前后输出差 ~1e-5 |
| `--onnx` | 关 | 额外导出 ONNX（输入名 `x`，输出名 `output`，`batch` 维度动态） |
| `--slim` | 关 | 用 **onnxslim** 优化 ONNX：融合 BN/胶水算子、合并外部权重为**单文件** |
| `--opset` | 11 | ONNX opset 版本 |
| `--legacy-exporter` | 关 | 改用旧版 TorchScript 导出器（`dynamo=False`）；图更干净，onnxslim 可再砍节点（示例：511→301，BN 全融），但该导出器已被官方标记弃用 |

**关于 dynamo vs legacy / 是否需要 onnxslim：**
- torch 2.9+ 默认用 **dynamo** 导出器（官方主流）；`legacy` 已弃用。
- **节点数多 ≠ 推理慢**：ONNX Runtime / TensorRT 加载时会自行做图优化。实测 511 vs 301 节点，CPU 延迟都在 ~26–28ms（噪声级）。
- `--slim` 的实用价值主要是：① 生成**单文件**（免 `.onnx.data` 分离；否则必须两个文件一起分发）；② 对优化能力弱的边缘端 runtime 更友好。
- 推荐：**默认（dynamo）+ `--fuse --onnx --slim`**；若某后端对 dynamo 图兼容差，再加 `--legacy-exporter`。

---

## 10. Paddle 权重转换

把 Paddle 的 `.pdparams` 转成 PyTorch `.pth`，需要**两个环境各跑一步**（Paddle 在 `paddlex` 环境、torch 在 `ptocr` 环境）：

```powershell
# 1) paddlex 环境：把官方 .pdparams 导出为 numpy pickle
conda activate paddlex
python tools/convert/dump_all.py          # 转换 ~/.torchkiln/pretrained 下所有 .pdparams -> .pkl

# 2) ptocr 环境：映射成 torch .pth
conda activate ptocr
python tools/convert/convert_all.py       # 生成 ~/.torchkiln/pretrained/<模型名>.pth
```

单独转换一个：
```powershell
# paddlex 环境
python tools/convert/dump_paddle_sd.py <xxx.pdparams> <xxx.pkl>
# ptocr 环境
python tools/convert/convert_det.py -c <cfg> --paddle-pkl <xxx.pkl> --output <xxx.pth>
```

> 转换规则：BN 的 `._mean/._variance → running_mean/running_var`；**`nn.Linear` 权重按 `[in,out]→[out,in]` 转置**（含方阵！）；`nn.Embedding` 不转置；Paddle 的 `stageN` / `bb_s_i` 命名映射到本项目对应命名。

---

## 11. 配置项全解

配置是 `-o` 可覆盖的普通 YAML。下面逐段说明**含义 / 默认 / 影响**。

### 11.1 `Global`

| 字段 | 默认 | 含义 / 影响 |
|---|---|---|
| `model_name` | — | 仅标识作用 |
| `use_gpu` | true | 是否用 GPU（`device` 为空时生效） |
| `epoch_num` | det 500 / rec 200 | 训练总轮数；同时决定 Cosine 的 `T_max` 与 warmup 步数。**续训时会自动沿用 checkpoint 里记录的值**，除非用 `-o Global.epoch_num=...` 显式指定 |
| `log_smooth_window` | 20 | 日志平滑窗口（仅影响打印） |
| `print_batch_step` | 10~100 | 每多少 step 打印一次 loss/lr |
| `save_model_dir` | ./output/... | 产物目录 |
| `save_epoch_step` | 10 | 每多少 epoch 存一次 `epoch_N.pth` |
| `eval_batch_step` | det [0,1500] / rec [0,2000] | `[起始step, 间隔]`：训练中途评估节奏 |
| `pretrained_model` | ModelScope URL | 预训练权重。**默认写成 ModelScope 的 URL**（首次自动下载并缓存到 `~/.torchkiln/ocr/pretrained`）；也支持本地路径或模型名；`null` 表示从零训练。见 [6.1](#61-预训练权重写-url首次自动下载并缓存) |
| `checkpoints` | null | 续训用 `latest.pth` |
| `use_ema` | false | 是否用 EMA 权重评估/保存（Paddle 默认关） |
| `ema_decay` / `ema_decay_type` | 0.9998 / threshold | EMA 衰减与调度 |
| `character_dict_path` | rec 必填 | 识别字典 |
| `max_text_length` | 25 | 识别最大文本长度 |
| `use_space_char` | true | 空格是否作为字符（影响类别数） |
| `d2s_train_image_shape` | det [3,640,640] / rec [3,48,320] | 导出/推理默认输入尺寸 |
| `device` | 空 | 设备：`cpu` / `gpu:0` / `gpu:0,1`（多卡）/ `cuda:0`；空则看 `use_gpu` |
| `distributed` | false | 分布式开关（一般由 `torchrun` 自动生效，无需手改） |
| `dist_backend` | 自动 | `nccl`/`gloo`，不填自动选 |
| `dist_init_method` | `env://` | 进程组初始化方式，torchrun 用默认 |
| `show_eval_progress` | true | 评估时是否打印 tqdm 进度条 |

### 11.2 `Architecture`

| 字段 | 含义 |
|---|---|
| `model_type` | `det` 或 `rec` |
| `algorithm` | 标识（`DB` / `SVTR_LCNet` / `SVTR_HGNet`…） |
| `Backbone` | `{name, ...}`，如 `PPLCNetV3{scale:0.75,det:true}` |
| `Neck` | det 常用（`RSEFPN`/`LKPAN`/`RepLKFPN`/`RepLKPAN`）；rec 常为 null（encoder 在 Head 里） |
| `Head` | det：`DBHead`/`PFHeadLocal`；rec：`MultiHead`（内含 CTCHead+NRTRHead/SARHead） |

> 一般**不要改** Architecture，否则与官方权重不匹配。

### 11.3 `Loss`

- 检测：`DBLoss{balance_loss, main_loss_type, alpha, beta, ohem_ratio}`（v6 用 `main_loss_type: DiceFocalLoss`，并带 aux 权重）。
- 识别：`MultiLoss{loss_config_list:[CTCLoss, NRTRLoss/SARLoss]}`（v3 用 SAR，v4+ 用 NRTR）。
- 影响：直接决定优化目标；改 `alpha/beta` 会改变 shrink/threshold/二值三项的权重比例。

### 11.4 `Optimizer`

| 字段 | 默认 | 影响 |
|---|---|---|
| `name` | Adam | 优化器 |
| `beta1/beta2` | 0.9/0.999 | Adam 动量 |
| `lr.name` | Cosine | 学习率调度（`Cosine`/`Const`） |
| `lr.learning_rate` | 0.001 | 基础学习率；**最常调的超参**，太大易发散、太小收敛慢 |
| `lr.warmup_epoch` | det 2 / rec 5 | 线性 warmup 轮数 |
| `regularizer.name` | L2 | 权重衰减类型 |
| `regularizer.factor` | det 5e-5 / rec 3e-5 | L2 系数 |

> 本项目复刻了 Paddle 的 LR 取用时机：warmup 从 0 线性升到 base，之后余弦衰减；与 Paddle 逐步一致。

### 11.5 `PostProcess` / `Metric`

- 检测 `DBPostProcess{thresh, box_thresh, max_candidates, unclip_ratio}`：
  - `thresh`：概率图二值化阈值；调大→框更少。
  - `box_thresh`：框得分阈值；调大→滤掉低分框（提 precision、降 recall）。
  - `unclip_ratio`：框扩展比例；调大→框更大。
- 识别 `CTCLabelDecode`：CTC 贪心解码 + 字典映射。
- `Metric`：`DetMetric{main_indicator:hmean}` / `RecMetric{main_indicator:acc}`。

### 11.6 `Train` / `Eval` 的 `dataset` 与 `loader`

```yaml
Train:
  dataset:
    data_dir: ...                  # 图片根目录（可 -o 覆盖）
    label_file_list: [...]         # 标注文件列表（可多个，会拼接）
    ratio_list: [1.0]              # 每份标注的采样比例（<1 会做子采样，并触发每 epoch 重采样）
    transforms: [...]              # 增广/预处理流水线
  loader:
    batch_size_per_card: 8         # 批大小；显存不足就调小
    shuffle: true                  # 训练打乱
    drop_last: false
    num_workers: 4                 # 多进程读取；本机安全上限（eval 用 0，见第 2 节）
    prefetch_factor: 4             # 每个 worker 预取的批数
Eval:
  dataset: {data_dir, label_file_list, transforms}
  loader: {batch_size_per_card, shuffle:false, num_workers}
```

**检测增广流水线（默认，v3~v5）**：
`DecodeImage → DetLabelEncode → CopyPaste → IaaAugment(Fliplr/Affine/Resize) → EastRandomCropData(640,50,keep_ratio) → MakeBorderMap → MakeShrinkMap → NormalizeImage → ToCHWImage → KeepKeys[image,threshold_map,threshold_mask,shrink_map,shrink_mask]`

**检测增广（v6）**：`... CopyPaste → ColorJitter → IaaAugment → RandomCrop → ...`（medium 还含 `RandomPerspective`）

**检测评估**：`DecodeImage → DetLabelEncode → DetResizeForTest → NormalizeImage → ToCHWImage → KeepKeys[image,shape,polys,ignore_tags]`

**识别增广（默认）**：`DecodeImage → RecConAug → RecAug → MultiLabelEncode → RecResizeImg([3,48,320]) → KeepKeys[image,label_ctc,label_gtc,length,valid_ratio]`

**识别评估**：`DecodeImage → MultiLabelEncode → RecResizeImg → KeepKeys[...]`

---

## 12. 常见问题 FAQ

**Q1. 报 `DependencyError: paddlepaddle is not available`？**
用错环境了。请用 `ptocr` 环境（本项目是纯 PyTorch，不需要 paddle）；报这个通常是跑到了 base/别的环境。

**Q2. 训练/评估报 `OpenBLAS error: Memory allocation still failed` 或 `WinError 1455 页面文件太小`？**
Windows 的**提交内存（RAM+页面文件）被 worker 进程耗尽**了（每个 worker 要加载 torch+CUDA，约 3.5~4 GB）。本机基线已占 37.5GB、上限 63.8GB，只剩约 26GB。处理：
1. 先确认没有崩溃后残留的 python 进程占着内存（`Get-Process python`），有就杀掉；
2. 减小 worker：`-o Train.loader.num_workers=2 -o Eval.loader.num_workers=0`;
3. 或**增大虚拟内存**到 64GB（推荐，这样才能开更多 worker）。
同时建议：
```powershell
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
```
（不要因此把 `num_workers` 长期设成 0——那会让训练慢 2.5 倍以上，见 [5.7](#57-训练太慢gpu-利用率低怎么办)）

**Q2b. 训练很慢、`nvidia-smi` 里 GPU 利用率只有 30~40%？**
就是 `num_workers` 太小/为 0。见 [5.7](#57-训练太慢gpu-利用率低怎么办)。

**Q3. CUDA out of memory？**
调小 `Train.loader.batch_size_per_card`（如 4/8），或先跑小 epoch 验证。识别模型激活较大，显存占用随 batch 线性增长。

**Q4. 识别加载官方权重报 `size mismatch` / 日志出现 `pretrained param(s) skipped because of shape mismatch`？**
说明**字符集（`Global.character_dict_path`）与预训练权重不一致**（`num_classes` 不同）。两种情况：

- **想用官方字符集微调** → 用匹配的字典：v5 用 `torchkiln/ocr/utils/dict/ppocrv5_dict.txt`；v6 small/medium 用 `ppocrv6_dict.txt`；v6 tiny 用 `ppocrv6_tiny_dict.txt`；v3/v4 用 `ppocr_keys_v1.txt`。
- **想用自己的字符集（比如只要 0-9 数字）** → 这**是正常的**：分类头形状不同会被自动跳过并重新初始化，其他权重照常加载。日志会警告但**不会中断训练**（和 PaddleOCR 行为一致）。记得必须显式指定：
  ```powershell
  -o Global.character_dict_path=E:/dx_ocr/rec/dict.txt
  ```
  漏掉这句会用到配置里的默认字典（示例数据的字典），既不匹配你的数据也不匹配预训练头。

> 实测：自己的 0-9 字典（`num_classes=12`）加载 v6_tiny_rec 预训练 → `4 pretrained param(s) skipped`，`missing=4 unexpected=0`，训练正常进行。

**Q5. `seed` 有什么作用？会不会影响收敛？**
`Global.seed`（默认 1024）用于模型初始化与部分增强的可复现。**注意**：数据集构建必须用 `seed=None`（本项目已按 Paddle 的做法处理），这样任意 seed 都能正常收敛。若你自己改数据集构造代码并传了固定 seed，可能触发退化裁剪——请勿那样做。

**Q6. 导出的 ONNX 有 `model.onnx` + `model.onnx.data` 两个文件？**
这是 ONNX 的 external data 机制（权重单独存）。分发时两个文件必须放一起。想合并成单文件用 `--slim`。

**Q7. ONNX 导出时打印 `'gbk' codec can't encode`？**
只是控制台编码问题，不影响导出。先 `$env:PYTHONIOENCODING="utf-8"`。

**Q8. 训练指标和官方文档差很多？**
先确认：① 用的是同一套数据与字典；② 从官方预训练微调 vs 从零训练；③ 训练轮数是否足够（官方是几百 epoch）。文档里的官方指标是在其内部数据集上测的，和你的数据不可直接比。

**Q9. 换数据集必须改 yml 吗？**
不用。所有数据相关字段都能用 `-o` 覆盖，见 [4.4](#44-用命令行指定自己的数据集推荐不用改-yml)。推荐把 yml 当“模型结构模板”，数据/超参全在命令行给。

**Q10. 多 GPU 怎么用？**
见 [5.5 多 GPU 训练](#55-多-gpu-训练)。Linux 用 `torchrun --nproc_per_node=N tools/train.py -c ... -o Global.device=gpu:0,1,...`；Windows 不想折腾就 `Global.device=gpu:0,1`（DataParallel）。

**Q11. 训练了好几轮，`precision`/`hmean` 还是 0，loss 也降得慢？**
先看这几点，按影响从大到小：

1. **有没有用官方预训练？** 这是最关键的。官方配置里的指标都是**从官方预训练微调**得到的。从零训练（日志里 `No pretrained weights specified (training from scratch).`）在几百张图的小数据集上，前几十个 epoch 检测不出东西是很常见的。**17 个 yml 默认已带模型名**，不写就是微调；别用 `-o Global.pretrained_model=null`。也可显式写：
   `-o Global.pretrained_model=https://www.modelscope.cn/models/ChaoII0987/TorchKiln/resolve/master/pretrained/PP-OCRv6_tiny_det.pth`（或直接写模型名 `PP-OCRv6_tiny_det`；默认就是官方 URL，见 [6.1](#61-预训练权重写-url首次自动下载并缓存)）
   实测：v6_tiny_det 用同一份数据，**从零训 1 epoch → hmean 0.0**；**加载预训练训 1 epoch → hmean 0.39**。
2. **学习率是否与总 batch 匹配。** 官方 yml 里 `learning_rate: 0.001 #(8*8c)` 的意思是：这个 lr 按**总 batch = 8 卡 × 8 = 64** 设计。你是单卡 batch 8（总 batch 8），PaddleX 配方用的是 **0.004**：
   `-o Optimizer.lr.learning_rate=0.004`
   同理，多卡放大总 batch 后 lr 也要按比例调大。
3. **epoch 是否够。** 官方是 500（det）/200（rec）；对照的 Paddle 那次是 100 epoch、`best_epoch=72` 才到 0.86。
4. **`hmean=0` 的具体表现**：如果你看到 `recall=0`（不是 `recall=1, precision` 很低），通常就是模型什么都没检出来，多半是 1/2/3 没做对。

> 加载预训练时应看到 `missing=0 unexpected=0`。若出现 `missing>0`，看日志里 `Some weights were NOT loaded (kept random): ...` 那一行：
> - 全是 `head.aux_binarize_*` / `head.aux_thresh_*`（v6 det 的**训练期辅助监督分支**）或 `*.num_batches_tracked`（BN 计数器）→ 无害，推理/导出只用主分支；
> - 出现别的名字 → 说明权重与配置不匹配（常见原因：识别字典不对，或用错了模型的权重），需要处理。
>
> 另外转换时若看到 `unused=...`：`v3_mobile_rec` 的 16 个是 Paddle `nn.LSTM` 内部 `{i}.cell.*` 与 `weight_ih_l*` 的**同一份数据的重复暴露**；`v4_server_det` 的 3 个是 PPHGNet 的 **ImageNet 分类头**（检测不用）——都无害。

---

## 13. 附：模型结构详解

- **Backbone**
  - `PPLCNetV3`：轻量级骨干，含可学习仿射、可重参数化卷积块（训练多分支、推理可融合为单卷积）；det 用 `det:true`。
  - `PPLCNetV4`：v6 骨干，按 `model_size`（tiny/small/medium）缩放，并带更大感受野的块。
  - `MobileNetV3` / `MobileNetV1Enhance` / `ResNet_vd` / `PPHGNet(_V2)`：其它版本骨干。
- **Neck（det）**
  - `RSEFPN`：残差+SE 的 FPN（v3/v4/v5 mobile）。
  - `LKPAN`：大核 PAN（v3/v4/v5 server）。
  - `RepLKFPN / RepLKPAN`：v6 使用，用 `DilatedReparamBlock`（多膨胀大核 DW，可重参数化）替代大核卷积；推理可融合。
- **Head（det）**
  - `DBHead`：Differentiable Binarization，输出 shrink/threshold/binary 三图（训练）；v4+ 带 `fix_nan`。
  - `PFHeadLocal`：DB 的改进版，带 CBN 分支（server）。
  - v6 `DBHead` 支持 `aux_in_channels`：颈返回多层次特征做**辅助监督**（提升收敛/精度）。
- **Head（rec）** `MultiHead`
  - `CTCHead`：主分支，CTC 解码，支持 `use_guide`（U-DML 引导）。
  - `NRTRHead`（v4+）/`SARHead`（v3）：训练期辅助分支（teacher/attention），提升精度；推理只用 CTC。
- **损失**：det 用 DB 的 shrink(Dice/BCE+OHEM) + threshold(L1) + binary(Dice)；v6 有些用 Dice+Focal，并加辅助损失。rec 用 CTC + NRTR/SAR 的 MultiLoss。

---

## 快速自检（推荐第一步）

一键验证全部模型的“模型+数据+1步训练+1步评估”是否正常：

```powershell
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python tools/smoke_all.py
# 期望输出结尾：76 OK, 0 FAIL（17 OCR + 53 YOLO + 2 plate + 2 attr + 1 action + 1 video）
```

只想快速确认“上游 YAML 是否都能建图 + 前向”（不需要数据集、秒级）：

```powershell
python tools/check_graph_build.py
# 期望输出结尾：53 configs: 53 OK, 0 FAIL
```

---

## 14. YOLO 任务（`torchkiln/`）

在同一个训练平台（`ptcore`）上，用**图模型**（`configs/yolo/*.yml`，小写+横线）覆盖 7 个任务。
**不依赖也不嵌入 `ultralytics`**（其代码与权重均为 AGPL-3.0），模型按公开架构描述从零编写。

| 任务 | `Architecture.task` | 头 / 损失 / 指标 | 配置文件 |
|---|---|---|---|
| 分类 | `classify` | `Classify` + CE（多标签 `-o Loss.multi_label=true`） | `configs/yolo/yolo11-cls.yml` |
| 检测 | `detect` | `Detect` + TAL + CIoU/DFL | `configs/yolo/yolo11-det.yml` |
| 旋转框 | `obb` | `OBB` + ProbIoU | `configs/yolo/yolo11-obb.yml` |
| 语义分割 | `semantic` | `SemanticSegment` | `configs/yolo/yolo26-sem.yml` |
| 实例分割 | `segment` | `Segment` + mask 原型 | `configs/yolo/yolo11-seg.yml` |
| 深度估计 | `depth` | `Depth` | `configs/yolo/yolo26-depth.yml` |

**用法与 OCR 完全一致**（业务 yml 放 `configs/local/`，仓库只留模板）：

```powershell
python tools/train.py -c configs/yolo/yolo11-det.yml -o Global.epoch_num=100 -o Train.dataset.data_dir=D:/mydata/det
python tools/eval.py  -c configs/yolo/yolo11-det.yml --weights output/yolo11_g/best_accuracy.pth

# 导出（inference.pth / inference.yml / model.pt / model.onnx / model_slim.onnx）
python tools/export.py -c configs/yolo/yolo11-det.yml --weights output/yolo11_g/best_accuracy.pth --save-dir output/exp --onnx --slim

# 推理（可视化）：cls 打印 top-5；det/obb 画框；seg 画框+掩码叠加；sem 上色；depth 伪彩色
python tools/infer/predict_yolo.py -c configs/yolo/yolo11-det.yml --weights output/yolo11_g/best_accuracy.pth --input img.jpg --output vis.jpg
```

导出说明：YOLO 任务默认 **opset 18**（OCR 仍是 11），因为主干含 `Resize/Interpolate`。
det/obb 导出为**多个输出**（每个尺度一个），seg 额外输出 mask 原型；已用 onnxruntime 验证与 PyTorch 数值一致（max diff ~2e-6）。

**数据格式**

| 任务 | 布局 | 标注 |
|---|---|---|
| cls | `data_dir/images/...` | `相对路径<TAB>类别号` |
| det | `images/` + `labels/` | `cls cx cy w h`（归一化） |
| obb | 同上 | `cls x1 y1 x2 y2 x3 y3 x4 y4`（归一化角点） |
| sem | `images/` + `masks/` | 掩码 PNG，像素为类别号 |
| segment | `images/` + `labels/` | `cls x1 y1 ... xn yn`（归一化多边形） |
| depth | `images/` + `depth/` | uint16 PNG（毫米，`depth_scale=1000`），0=无效 |

`label_file_list` 是若干 `.txt`，每行一个图片相对路径；除 cls 外，掩码/标注按 `images → labels|masks|depth` 替换路径、扩展名换成 `.txt|.png` 查找。

**实测（合成 demo 数据）**

| 任务 | 指标 |
|---|---|
| det | mAP50 **0.894** / mAP75 0.610 / mAP50-95 0.742 |
| obb | mAP50 **0.769** / mAP50-95 0.393 |
| sem | mIoU **0.958** / acc 0.992 |
| depth | δ1 **0.946** / AbsRel 0.104 / RMSE 0.498 |
| seg | box_mAP50 **0.901** / mask_mAP50 **0.800** / mask_mAP50-95 0.493 |

> 许可提示：以上均为独立实现，未使用任何 AGPL 代码或权重；若要发布模型，请用**自己的数据从零训练**。
> 生成 demo 数据的脚本见 `datasets/*_demo`（合成数据，仅用于自检与演示）。

---

## 15. YAML 图模型 + ultralytics 生态兼容

`torchkiln/nn/` 是一个**模块库 + 注册表 + YAML 图解析器**：模型用上游同款的 layer-list
格式描述（`[-1, 1, Conv, [64, 3, 2]]`、`-1` 指上一层、列表 `from` 做 Concat、
`scales: {n: [depth, width, max_channels]}` 自动缩放），**不用 import ultralytics**。

### 15.1 已支持的模型家族（上游 YAML 全量搬运）

上游 `ultralytics/cfg/models` 里**所有可训练的 YAML** 都搬到了 `torchkiln/cfg/models/`，
并在 `configs/yolo/` 生成了可直接训练的配置：

| 家族 | YAML 目录 | 变体（= 生成的 `*.yml`） | 任务 |
|---|---|---|---|
| v3 | `v3/` | `yolov3` `yolov3-spp` `yolov3-tiny` | det |
| v5 | `v5/` | `yolov5` `yolov5-p6` | det |
| v6 | `v6/` | `yolov6` | det |
| v8 | `v8/` | `yolov8` `-p2` `-p6` `-ghost` `-ghost-p2` `-ghost-p6` `-seg` `-seg-p6` `-obb` `-pose` `-pose-p6` `-cls` | det/seg/obb/pose/cls |
| v9 | `v9/` | `yolov9c` `yolov9e` `yolov9m` `yolov9s` `yolov9t` `-seg`(c/e) | det/seg |
| v10 | `v10/` | `yolov10` `b` `l` `m` `n` `s` `x` | det（NMS-free） |
| v11 | `11/` | `yolo11` `-seg` `-obb` `-pose` `-cls` | det/seg/obb/pose/cls |
| v12 | `12/` | `yolo12` `-seg` `-obb` `-pose` `-cls`（`AreaAttention`） | det/seg/obb/pose/cls |
| v26 | `26/` | `yolo26` `-p2` `-p6` `-seg` `-obb` `-pose` `-cls` `-sem` `-depth` | det/seg/obb/pose/cls/sem/depth |

合计 **50 个上游 YAML + 3 个手写配置**（`yolov8n_det`、`yolov9`、`yolov10`），
自检脚本 `tools/check_graph_build.py` 逐个建图 + 前向：**53 configs, 0 FAIL**。

> 端到端头（v10/v26）与 p2/p6 多尺度分支都能直接跑：`Head.end2end` 与
> **anchor stride 由模型头部自动推导**（`[4,8,16,32]` / `[8,16,32,64]`），
> 配置里写不写 `strides` 都会被纠正为真实层数，避免锚点错位。

**模块库**（`torchkiln/nn/modules.py`，YAML 里可直接引用名字）：

```
Conv DWConv ConvTranspose Bottleneck C1 C2 C2f C2fCIB C3 C3x C3TR C3Ghost C3k C3k2
PSA C2PSA C2fAttn Attention ABlock AreaAttention A2C2f SPP SPPF SPPELAN Concat
Proto Classify Detect Detect26 Detect10 v10Detect OBB OBBU Segment SegmentU Pose PoseU
SemanticSegment Depth ADown AConv GhostConv GhostBottleneck CIB ELAN1 SCDown
CBLinear CBFuse RepConv RepConvN RepNCSP RepNCSPELAN4 ResNetLayer TorchVision
```

命名映射：YAML 里的 `Detect`/`Segment`/`OBB`/`Pose` 分别落到 YOLO26 风格的
`Detect26`/`SegmentU`/`OBBU`/`PoseU`（与上游 8.4.x 权重结构一致），`*DFL` 后缀是保留的
DFL 变体，`v10Detect` → `Detect10`。

### 15.2 开关

| 配置 | 位置 | 作用 |
|---|---|---|
| `yaml_file` / `yaml_text` | `Architecture` | 用 YAML 建图模型（否则走手写实现） |
| `scale` | `Architecture` | `n/s/m/l/x`，按 `scales` 自动缩放 |
| `Head.reg_max` | `Architecture.Head` | `>1` 启用 **DFL** 可微回归（v8 默认 16）；`1` 为 DFL-free（上游 8.4.x 的 `Detect` 就是 DFL-free，故图配置一律写 `1`） |
| `Head.end2end` | `Architecture.Head` | **NMS-free 端到端**（配 `Detect10` 头） |
| `Head.reg_layout` | `Architecture.Head` | OBB 布局：`upstream`（`[reg, cls, angle]`）/ `ltrb_angle`（我们手写头的布局） |
| `Head.kpt_shape` | `Architecture.Head` | 关键点数，如 `[17, 3]` / `[4, 3]` |

> 端到端模型导出的 ONNX **图内没有 NMS 节点**（已实测：opset 18、3 个输出、无 NMS），部署时不需要再写后处理。

### 15.3 训练特性（与 ultralytics 对齐的那部分）

```yaml
Global:
  amp: true              # 混合精度（fp16/bf16，amp_dtype 可选）
  freeze: 8              # 冻结前 8 层（也可给名字前缀列表）
  accumulate: 2          # 梯度累积
Optimizer:
  name: AdamW            # Adam / AdamW / SGD / RAdam / NAdam
  momentum: 0.937        # SGD
  lr:
    name: Cosine         # Cosine / Linear / OneCycle / Const
    learning_rate: 0.01
    lrf: 0.01            # 余弦终值系数（不给则按 Paddle 语义衰减到 0）
    warmup_epoch: 3
Loss:
  label_smoothing: 0.05  # det（软目标）/ cls / sem
Train:
  dataset:
    augment:
      mosaic: 1.0        # 4 图拼图
      mosaic9: 0.0       # 9 图拼图（3x3，中心放本图，目标更小、更多截断）
      mixup: 0.1
      copy_paste: 0.0    # 实例级拷贝粘贴（需要掩码 → seg）
      close_mosaic: 60   # 多少 epoch 后自动关掉 mosaic/mosaic9
      multi_scale: 0.3   # 每个 epoch 随机尺寸 ±30%
      hsv: {p: 0.5, hgain: 0.015, sgain: 0.7, vgain: 0.4}
      affine: {degrees: 0.0, translate: 0.1, scale: 0.5, shear: 2.0, perspective: 0.0005}
```

分类（`cls`）另有 **random_erasing**（上游只用在分类上，不破坏框/掩码标注）：

```yaml
Train:
  dataset:
    augment:
      erasing: {p: 0.25, sl: 0.02, sh: 0.4, min_aspect: 0.3, max_count: 1}
```

增广对 **检测框（xyxy / 旋转框 xywhr）、关键点、实例掩码** 做同一套几何变换
（`torchkiln/data/augment.py`），已接入 det/obb/seg/pose/sem/depth 六个数据集；
`mosaic9`/`copy_paste` 同样会一起搬移框、关键点与掩码（`copy_paste` 只粘贴与目标图
同类别的实例，且与已有目标 `IoU > 0.3` 时跳过，避免无效遮挡）。

实测收益：v8-det `mAP50-95 0.690 → 0.717`；v8-obb `mAP50 0.940 → 0.990`。

### 15.4 直接使用上游 `.pt` 权重

上游 `.pt` 里 pickle 了整个模型对象，纯 PyTorch 环境读不了，所以分两步：

```powershell
# ① 在有 ultralytics 的环境里导出为纯张量
python tools/convert/dump_ultralytics.py yolo11n.pt --out _downloads/upstream

# ② 在 ptocr 环境里微调（我们的 loader 会按名字+形状自动匹配）
python tools/train.py -c configs/yolo/yolo11-det.yml ^
  -o Global.pretrained_model=_downloads/upstream/yolo11n.pth ^
  -o Global.epoch_num=100
```
实测：一个上游 yolo11n 检测权重 **708 个张量中 546 个（77.1%）可直接加载，0 形状冲突**，
剩余的是分类头（类别数不同，需重训）和少量结构差异块；用它初始化微调我们 demo 数据 →
`mAP50 0.915`。

> 若配置里 `Head.reg_max` 与权重不符（例如上游是 DFL-free 的 `4` 通道回归而配置写了 16），
> 回归头会被跳过。v11/v26 的配置已按上游设为 `reg_max: 1`。

### 15.5 上游 YAML 兼容性 + 自检

<!-- -->

```powershell
# ① 逐个建图 + 前向（不需要数据，秒级）：53 configs, 0 FAIL
python tools/check_graph_build.py            # 只看结论
python tools/check_graph_build.py --verbose  # 打印每个配置的 task / 输出通道
python tools/check_graph_build.py --family yolo26

# ② 端到端（数据 + 1 步训练 + 1 步评估）：76 OK, 0 FAIL
python tools/smoke_all.py
```

* **不需要改 YAML 就能对齐上游**：`scales`/`depth_multiple` 自动缩放、`-p2/-p6` 多尺度、
  `CBLinear/CBFuse`（v9 的 GELAN 辅助分支）、`AreaAttention`（v12）、
  `RepConv/RepNCSPELAN4/SPPELAN/ELAN1/AConv`（v9）、`C3Ghost/GhostConv`（v8-ghost）都已实现。
* **配置必须带 `Optimizer` 段**（`configs/yolo/*.yml` 已自动生成 `Adam + Cosine`），
  否则训练器会以 `KeyError: 'Optimizer'` 报错——这是有意的（避免静默用错学习率）。
* **上游没有的/不做的**：`rt-detr`/`yolov8-rtdetr`（RT-DETR）、`yoloe-*`、`yolov8-world*`、
  `sam*` 以及 `*-cls-resnet*`（TorchVision/ResNet 主干）。这些要么不支持训练、
  要么不在本项目范围内（迁移脚本会自动跳过）。
* 上游 YAML 里 `nn.ConvTranspose2d`、`nn.Upsample`、`nn.Identity` 这类原始模块名也能解析；
  v3 的**负数相对索引**（`-2` = 上上层）与上游一致。

---

## 16. 车牌(检测 + 识别)

来源:[`we0091234/Chinese_license_plate_detection_recognition`](https://github.com/we0091234/Chinese_license_plate_detection_recognition)
(检测)+ 子仓库 `crnn_plate_recognition`(识别)。**不 import 上游代码**,按源码复刻并在**权重级**对齐。

| 配置 | 模型 | 结构要点 | 指标 |
|---|---|---|---|
| `configs/plate/plate_det.yml` | 检测(单层/双层 2 类 + **4 角点**) | yolov5-lite(ShuffleNetV2:`StemBlock`/`ShuffleV2Block`)+ yolov5-face 头 `PlateDetect`(逐 anchor `[xy,wh,obj,kpt(8),cls(2)]`,`no=15`) | 框 mAP + 角点 NME/kpt_px |
| `configs/plate/plate_rec.yml` | 识别 + 颜色(5 类) | `PlateRecNet`(`myNet_ocr_color`):CNN + CTC(78 类)+ 颜色头;输入 48×168,mean/std = 0.588/0.193,**无 RNN** | 整牌 / 字符 / 颜色 acc |

```powershell
# ① 转换上游权重(读 pickled 模型对象,无需上游代码)
python tools/convert/convert_plate_weights.py `
  --detect _downloads/Chinese_license_plate_detection_recognition/weights/plate_detect.pt `
  --rec    _downloads/Chinese_license_plate_detection_recognition/weights/plate_rec_color.pth `
  --out    _downloads/plate

# ② 训练 / 评估(用官方权重微调)
python tools/train.py -c configs/plate/plate_det.yml -o Global.pretrained_model=_downloads/plate/plate_detect.pth
python tools/train.py -c configs/plate/plate_rec.yml -o Global.pretrained_model=_downloads/plate/plate_rec_color.pth
python tools/eval.py  -c configs/plate/plate_det.yml --weights output/plate_det/best_accuracy.pth

# ③ 对齐校验
python tools/check_plate_models.py   # 检测 500/500、识别 86/86 张量;识别 ONNX 数值 max|diff| ~1e-5
```

对齐证据:
* 检测 `plate_detect.pt` → **500/500 张量,0 冲突**,与上游 `models/yolo.py::Detect` 逐元素 **max|diff| = 0.000e+00**;
* 识别 `plate_rec_color.pth` → **86/86 张量**,与出厂 ONNX 一致(CTC 6.1e-05 / 颜色 6.7e-06);
* ⚠ 用户 `test_data` 里的 `yolov5plate.onnx` 是**更早版权重**(首层卷积即不同),不能作为对齐基准;
* 车牌识别**不是 LPRNet**:仓库虽含 `LPRNet.py`,但权重来自 `plateNet.py/colorNet.py` 的 CNN+CTC(无 RNN)。

数据格式:`路径 c1 c2 ... c7 颜色号`(字符表见 `torchkiln/nn/plate.py::PLATE_CHARSET`,
颜色 `['黑色','蓝色','绿色','白色','黄色']`);检测标签为 `cls cx cy w h p1x p1y ... p4x p4y`(`kpt_shape: [4,2]`)。

---

## 17. 属性识别(行人 / 车辆)

来源:`E:\PaddleX` 的 `multilabel_classification` 模块(PaddleClas 后端)。**一个模型,两份数据**,仅 `num_classes` 与输入方向不同。

| 配置 | 类别 | 输入 | 属性名表 |
|---|---|---|---|
| `configs/attr/pedestrian_attribute.yml` | **26** | 256×192(竖版) | `torchkiln/utils/pedestrian_attribute_label_list.txt` |
| `configs/attr/vehicle_attribute.yml` | **19**(9 颜色 + 10 车型) | 192×256(横版) | `torchkiln/utils/vehicle_attribute_label_list.txt` |

* 网络 `AttributeNet` = `PPLCNetX1_0`(真实 `NET_CONFIG`:blocks5 无 SE、blocks6 仅 2 单元)+ 多标签头
  (`1×1 last_conv 512→1280 → hardswish → dropout(0.5, use_ssld) → Linear`),**1,707,386 参数**;
* 损失 `MultiLabelLoss`:BCE + `ratio2weight = exp((1-t)·r + t·(1-r))`,`size_sum=True`(逐类求和、逐样本平均);
* 指标 `AttrMetric`:阈值 0.5 的逐属性准确率均值 **mA** + **mAP**;
* 增广对齐 `TimmAutoAugment(rand-m9-mstd0.5-inc1, p=0.8)`(见 `torchkiln/attr_randaug.py`)+ RandomErasing + Flip。

```powershell
# 官方权重转换(PaddleClas .pdparams → torch .pth)
& C:\ProgramData\miniconda3\envs\paddlex\python.exe tools/convert/dump_attribute_paddle.py `
  --weights ~/.torchkiln/pretrained/PP-LCNet_x1_0_pedestrian_attribute_pretrained.pdparams --out ~/.torchkiln/pretrained/ped_attr_paddle.pkl
python tools/convert/convert_attribute_weights.py --pkl ~/.torchkiln/pretrained/ped_attr_paddle.pkl --num-classes 26 `
  --out ~/.torchkiln/pretrained/PP-LCNet_x1_0_pedestrian_attribute.pth

# 训练 / 评估
python tools/train.py -c configs/attr/pedestrian_attribute.yml `
  -o Global.pretrained_model=~/.torchkiln/pretrained/PP-LCNet_x1_0_pedestrian_attribute.pth
python tools/train.py -c configs/attr/vehicle_attribute.yml `
  -o Global.pretrained_model=~/.torchkiln/pretrained/PP-LCNet_x1_0_vehicle_attribute.pth
python tools/eval.py  -c configs/attr/vehicle_attribute.yml --weights output/PP-LCNet_x1_0_vehicle_attribute/best_accuracy.pth
```

对齐状态:官方权重 **146/146 张量、0 冲突**,`load_state_dict` missing=0/unexpected=0,
同一输入下 top-5 属性与 PaddleClas **4/5 一致**;logits 仍有 ~0.8 偏差,已定位到 `blocks3.0.pw_conv`
的裸 1×1 卷积(权重 bit 相同)— 属待收尾项。

数据格式:`路径 v1 v2 ... vC`(0/1 或软标签;`label_ratio: true` 时数据集会统计每个属性的**全局正例比例**参与损失加权)。

---

## 18. 项目结构总览与文档索引

```
E:\TorchKiln
├─ README.md            本文件(全平台说明)
├─ docs/                STRUCTURE / MODEL_ZOO / TRAINING / CONFIG_REFERENCE / FAQ
├─ ptcore/              平台层:config / pretrained / optimizer / precision / ema / batch_sampler /
│                       task(TaskAdapter)/ trainers(每任务一个 trainer + base.BaseTrainer)
│                       / trainer.py(兼容 shim)/ factory
├─ torchkiln/ocr/          OCR 产品线:modeling(backbones/necks/heads/transforms)、data/imaug、
│                       losses、metrics、postprocess、utils/dict、task、trainer
├─ torchkiln/         YOLO + 车牌 + 属性:nn(modules/graph/plate/attribute)、models、det、
│                       data(augment/det/seg/pose/sem/depth/plate)、tasks/(每任务一个文件:
│                       classify/detect/obb/segment/pose/semantic/depth/plate_det/plate_rec/
│                       attribute/pose_action/video_cls + _base/_cls)、cfg/models(上游 YAML 53 个)
├─ configs/             ocr/det(9) ocr/rec(8) yolo(53) plate(2) attr(2) action(1) video(1) + 本地 local/_parity(不入库)
├─ tools/               train / eval / export / smoke_all / check_graph_build / check_plate_models /
│                       gen_configs / download_pretrained / infer(predict_{det,rec,yolo})
│  └─ convert/          dump_*(paddlex/ultralytics 环境) + convert_*(ptocr 环境) + parity_*(数值对比)
├─ datasets/            OCR 官方示例 + YOLO 七任务 demo + 车牌/属性合成样例
├─ _downloads/          official(权重) / upstream(上游 YOLO) / plate(车牌) / 上游仓库与参考实现
└─ output/ wheels/      训练产物 / 离线依赖
```

详细目录树与每个模块的职责见 [`docs/STRUCTURE.md`](docs/STRUCTURE.md);模型总表见 [`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md)。

**自检基线(改代码/加模型后必跑)**

| 命令 | 期望 |
|---|---|
| `python tools/smoke_all.py` | `76 OK, 0 FAIL`(每配置 1 步训练 + 1 步评估;跳过 `_parity`/`local`)|
| `python tools/check_graph_build.py` | `53 configs: 53 OK, 0 FAIL` |
| `python tools/check_plate_models.py` | 检测 500/500、识别 86/86,识别 ONNX `max|diff| ≈ 1e-5` |

**依赖方向**:`configs → ptcore.config → ptcore.factory → ptcore.trainer.BaseTrainer ← {torchkiln.ocr.TaskAdapter, torchkiln.TaskAdapter}`;
`ptcore` 不反向依赖任何产品线;`torchkiln/ocr/utils/*`、`torchkiln/ocr/optimizer/` 是兼容 shim(真实现在 `ptcore/`)。


