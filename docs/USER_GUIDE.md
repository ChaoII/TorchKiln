# TorchKiln 小白完整手册（训练 / 评估 / 推理 / 导出）

> 看完这一篇就能跑起来。所有参数均来自**代码真实 argparse / 配置真值**（2026-09）。
> 命令在**仓库根目录** `E:\TorchKiln` 执行；环境：`conda activate ptocr`。
> 统一 CLI：`tkiln` / `tkiln.bat` / `python -m torchkiln` / `python tools/xxx.py` **四套完全等价**。

---

## 0. 五分钟跑通（复制粘贴）

```powershell
conda activate ptocr
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"

# ① 自检：只建模型，不训练（秒级）
.\tkiln.bat check -c configs/yolo/yolo11-det.yml

# ② 训练：yolo11n、用第 2 张卡、100 轮、每 10 步打日志
.\tkiln.bat train -c configs/yolo/yolo11-det.yml `
  -o Architecture.scale=n `
  -o Global.device=cuda:2 `
  -o Global.epoch_num=100 `
  -o Global.print_batch_step=10 `
  -o Global.save_model_dir=./output/y11n `
  -o Train.dataset.data_dir=D:/mydata/det `
  -o Train.dataset.label_file_list=train.txt `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt `
  -o Architecture.Head.num_classes=1 `
  -o Train.dataset.names=[plate]

# ③ 评估
.\tkiln.bat val -c configs/yolo/yolo11-det.yml `
  --weights output/y11n/best_accuracy.pth `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt

# ④ 推理（单张图）
.\tkiln.bat predict -c configs/yolo/yolo11-det.yml `
  --weights output/y11n/best_accuracy.pth `
  --input D:/img.jpg --output output/result.jpg --device cuda:2

# ⑤ 导出 ONNX
.\tkiln.bat export -c configs/yolo/yolo11-det.yml `
  --weights output/y11n/best_accuracy.pth `
  --save-dir output/export/y11n --onnx --slim
```

没有数据集时先用占位数据冒烟：

```powershell
python tools/make_demo_data.py --all
.\tkiln.bat train -c configs/yolo/yolo11-det.yml -o Global.epoch_num=1 -o Global.device=cuda:2
```

---

## 1. CLI 总览

### 1.1 四套入口（任选其一）

| 写法 | 说明 |
|---|---|
| `.\tkiln.bat <mode> ...` | 仓库根启动器（**推荐**，免 pip） |
| `tkiln <mode> ...` | `pip install -e .` 之后 |
| `python -m torchkiln <mode> ...` | 免安装等价 |
| `python tools/train.py ...` | 底层脚本，参数与上面完全一样 |

解释器优先级：`TKILN_PYTHON` → `CONDA_PREFIX` → `ptocr` → `python`。

### 1.2 mode 一览（5 个）

| mode | 实际脚本 | 干什么 |
|---|---|---|
| `train` | `tools/train.py` | 训练 |
| `val` | `tools/eval.py` | 评估 |
| `predict` | `tools/infer/predict_{yolo,det,rec}.py` | 单图推理（自动路由） |
| `export` | `tools/export.py` | 导出 pth / TorchScript / ONNX |
| `check` | CLI 内置 | 只建模型打印摘要，**不训练** |

另有数据子命令：

```powershell
tkiln data list                 # 看数据集缺什么（别名 verify/status，缺省就是 list）
tkiln data get plate_det_demo   # 下载并解包（别名 download/pull）
tkiln data get --all            # 全部；--force 强制重下
```

### 1.3 task 别名（可以省略，一般都省略）

```text
detect|det   segment|seg   obb   pose   classify|cls   semantic|sem   depth
plate_det|plate-det        plate_rec|plate-rec        attribute|attr
pose_action|pose-action|action        video_cls|video-cls|video
ocr   ocr_det|ocr-det   ocr_rec|ocr-rec
```

省略 task 时任务由配置 `Architecture.task` 决定（**以配置为准**）。  
显式写法只多做一致性校验：

```powershell
.\tkiln.bat detect train -c configs/yolo/yolo11-det.yml
```

**predict 按任务自动换脚本**：

| 配置任务族 | 实际脚本 |
|---|---|
| YOLO 系（detect/seg/obb/pose/cls/sem/depth/plate/attr...） | `predict_yolo.py` |
| OCR 检测 | `predict_det.py` |
| OCR 识别 | `predict_rec.py` |

### 1.4 透传规则

`tkiln` 后面除可选 `<task>`、`<mode>` 外，**其余参数原样丢给 tools/*.py**。  
所以本文所有 `tools/xxx` 的参数，都能原样写在 `tkiln` 后面。

```powershell
tkiln --help          # 打印本 CLI 的 USAGE，不是 tools 的 argparse help
python tools/train.py --help    # 真正的 train 参数
```

---

## 2. `-o` 配置覆盖（所有命令通用）

### 2.1 语法

```text
-o 键路径=值
```

- 键用点分层：`Global.epoch_num`、`Train.dataset.data_dir`、`Optimizer.lr.learning_rate`
- 可重复：`-o a=1 -o b=2`，也可一次多个：`-o a=1 b=2`
- 值类型按 yml 原类型自动转：`int/float/bool/list/dict/null`

| 值写法 | 含义 |
|---|---|
| `100` | int |
| `0.001` | float |
| `true` / `false` | bool |
| `null` / `none` | None |
| `a.txt,b.txt` | list（逗号拆） |
| `[640,640]` | yaml list |
| 裸键 `-o Global.amp` | 置 True |

### 2.2 PowerShell 什么时候加引号（小白背这两条）

**规则 A（最常用）**：值里没有 **空格、逗号、`[]`** → **不用引号**。

```powershell
-o Global.epoch_num=100
-o Optimizer.lr.learning_rate=0.001
-o Train.dataset.augment.hsv.p=0.5          # 点不是问题
-o Train.dataset.augment.affine.scale=0.5   # 点不是问题
```

**规则 B**：出现 **空格 / 逗号 / `[]` / `{}`** → **整段 `-o 键=值` 用双引号包住**。

```powershell
-o "Train.dataset.label_file_list=a.txt,b.txt"
-o "Train.dataset.data_dir=D:/my data/det"
-o "Train.dataset.transform.image_size=[640,640]"
```

**记不住就全部加引号**，永远正确：

```powershell
"-o Global.epoch_num=100"
"-o Train.dataset.augment.hsv.p=0.5"
```

### 2.3 常见报错

| 现象 | 原因 |
|---|---|
| `unknown config section(s): Trainn...` | 段名写错（会提示可用段） |
| 未知子键只 warning | 仍会写入，并记到 `save_model_dir/config.yml` 的 `_overrides` |
| 段不存在 | `-o` **不会**凭空创建顶层 section |

可用段名（典型）：`Global, Architecture, Loss, Optimizer, PostProcess, Metric, Train, Eval`。

---

## 3. 训练 `tkiln train`（完整参数）

### 3.1 命令行参数（argparse 真值）

| 参数 | 必填 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| `-c` / `--config` | **是** | str | — | 配置 yml 路径 |
| `-o` / `--opt` | 否 | list[str] | None | 覆盖配置，可重复 |

**train 命令行只有这两个**；其余全是 **`-o` 改配置字段**。

### 3.2 Global（全局）——按使用频率

| `-o` 键 | yml 默认（YOLO 系典型） | 含义 | 小白注意 |
|---|---|---|---|
| `Global.epoch_num` | YOLO `100`；OCR det `500` | 训练轮数 | 必改项之一 |
| `Global.device` | 往往**没写**（空） | **用哪张卡** | 写 **`cuda:2`** 或 **`gpu:0,1`**；写 `2` 会落到 CPU |
| `Global.use_gpu` | `true` | 无 device 时是否用 GPU | 一般不动 |
| `Global.print_batch_step` | YOLO `10`；OCR det `100` | **每 N step 打一行训练日志** | 只管日志，不管评估 |
| `Global.eval_batch_step` | YOLO `[0,2000]`；OCR `[0,1500]` | `[起始step, 间隔]` 按 step 评估 | `[0,0]` = 关训练中评估 |
| `Global.eval_epoch_step` | 常无 | **每 N epoch 评估一次** | 与 print_batch_step **无关**，可同时写 |
| `Global.save_model_dir` | 如 `./output/yolo11_g` | 输出目录 | 里面：`best_accuracy.pth` / `latest.pth` / `train.log` / `config.yml` |
| `Global.save_epoch_step` | `10` | 每 N epoch 存 `epoch_N.pth` | |
| `Global.pretrained_model` | YOLO 常 `null`；OCR 常 ModelScope URL | 预训练权重 | 路径 / URL / 缓存名；**只加载权重**，epoch 从 0 |
| `Global.checkpoints` | `null` | **断点续训** | 指向 `latest` 前缀；恢复优化器/EMA/epoch；**与 pretrained 语义不同** |
| `Global.use_ema` | YOLO `true`；OCR `false` | EMA | 评估 best 时用 EMA 权重 |
| `Global.ema_decay` | YOLO `0.9999` | EMA 衰减 | |
| `Global.ema_decay_type` | YOLO `exponential`；OCR `threshold` | EMA 类型 | 对齐 ultralytics 用 exponential |
| `Global.amp` | 常 `false` | 混合精度 | `amp_dtype: fp16/bf16`；**评估强制 fp32** |
| `Global.freeze` | `null` | 冻结 | 数字=前 N 层；或 `["conv1","blocks2"]` |
| `Global.accumulate` | `1` | 梯度累积 | 有效 batch ≈ `batch × accumulate` |
| `Global.seed` | `1024` | 随机种子 | OCR 管线需 `None` 才好复现 |
| `Global.character_dict_path` | OCR rec 用 | 字符集 | 必须与模型输出维一致 |
| `Global.show_eval_progress` | `true` | 评估进度条 | `false` 可少点 IO |
| `Global.dist_backend` | 自动 | DDP 后端 | NCCL / gloo |
| `Global.dist_init_method` | `env://` | 进程组 | torchrun 默认即可 |
| `Global.distributed` / `Global.gpu` | 历史值 | **死键，训练不读！** | 多卡见 §3.6 |

### 3.3 Architecture（模型）

| `-o` 键 | 说明 | 小白注意 |
|---|---|---|
| `Architecture.scale` | `n/s/m/l/x` | yolo11n → `n` |
| `Architecture.Head.num_classes` | 类别数 | 自定义数据集**必改** |
| `Architecture.task` | 任务 | 一般 yml 已写好 |
| `Architecture.model_family` | `yolo` / `ocr` | 一般不动 |
| `Architecture.yaml_file` | YAML 图结构 | 一般不动 |
| `Architecture.Head.reg_max` | `1`=无 DFL；`>1`=DFL | yolo11/v8 常见 yml 已是 `1`；与权重不一致会跳过回归头 |
| `Architecture.Head.end2end` | 端到端头 | v10/v26 等 |
| `Architecture.Head.kpt_shape` | 关键点形状 | pose/plate |

### 3.4 Optimizer（优化器 + 学习率）

**YOLO `yolo11-det.yml` 默认（真值）**：

| 键 | 默认 |
|---|---|
| `Optimizer.name` | **SGD** |
| `Optimizer.momentum` | 0.937 |
| `Optimizer.nbs` | 64（名义 batch，用于等比缩 lr） |
| `Optimizer.lr.name` | **Linear** |
| `Optimizer.lr.learning_rate` | **0.01** |
| `Optimizer.lr.lrf` | 0.01（终值系数） |
| `Optimizer.lr.warmup_epoch` | 3 |
| `Optimizer.lr.warmup_bias_lr` | 0.1 |
| `Optimizer.regularizer.name` | L2 |
| `Optimizer.regularizer.factor` | 0.0005 |

**OCR det `PP-OCRv5_mobile_det.yml` 默认**：`Adam` + `lr=0.001` + `Cosine` + `warmup_epoch=2` + L2 `5e-5`。

| `-o` 键 | 取值 | 说明 |
|---|---|---|
| `Optimizer.name` | Adam / AdamW / SGD / RAdam / NAdam | OCR 的 Momentum≡SGD |
| `Optimizer.momentum` | 0.937 等 | SGD/RAdam |
| `Optimizer.beta1` / `beta2` | 0.9 / 0.999 | Adam 系 |
| `Optimizer.nesterov` | false | |
| `Optimizer.lr.name` | Cosine / Linear / OneCycle / Const | |
| `Optimizer.lr.learning_rate` | 0.01 等 | **基础学习率** |
| `Optimizer.lr.lrf` | 0.01 | 余弦终值比例；缺省可能衰到 0 |
| `Optimizer.lr.warmup_epoch` | 3 | 或 `warmup_steps` |
| `Optimizer.regularizer.factor` | 0.0005 | 权重衰减 |

日志里的 `lr:` 就是当前学习率。

### 3.5 数据与加载

| `-o` 键 | 默认（yolo11-det） | 说明 |
|---|---|---|
| `Train.dataset.data_dir` | `datasets/det_demo` | 图片根目录 |
| `Train.dataset.label_file_list` | `[datasets/det_demo/train.txt]` | 相对 data_dir 或绝对路径 |
| `Eval.dataset.data_dir` | `datasets/det_demo` | |
| `Eval.dataset.label_file_list` | `[...,/val.txt]` | |
| `Train.dataset.transform.image_size` | YOLO `320`（demo） | 常改成 640 |
| `Train.loader.batch_size_per_card` | **8** | 单卡 batch |
| `Train.loader.shuffle` | true | |
| `Train.loader.num_workers` | **4** | Windows 建议 0~4 |
| `Train.loader.prefetch_factor` | 4 | |
| `Eval.loader.batch_size_per_card` | 8 | |
| `Eval.loader.num_workers` | **0** | 评估别开大 |
| `Train.dataset.names` | 常无 | 类别名，YOLO 可 `[cat,dog]` |
| `Train.dataset.box_format` | obb 为 `xywhr` | **标签 9 字段角点** |
| `Train.dataset.kpt_shape` | null | pose/plate |

**路径语义**：`label_file_list` 项不是绝对路径且不是已存在文件 → 拼到 `data_dir`；已是存在的文件 → 直接打开。  
**没有** `train_list` / `val_list` 字段。

### 3.6 数据增强 `Train.dataset.augment`

| 键 | 含义 | 常见值 |
|---|---|---|
| `mosaic` | 4 图拼接概率 | det `1.0`；小数据微调可 `0` |
| `mosaic9` | 9 图拼接 | `0` |
| `mixup` | mixup | `0~0.1` |
| `copy_paste` | 粘贴（seg 需掩码） | `0` |
| `close_mosaic` | 最后 N epoch 关 mosaic | `10~15` |
| `multi_scale` | 随机尺度幅度 | `0` 或 `0.3` |
| `fliplr` | 水平翻转 | `0.5` |
| `hsv.p/hgain/sgain/vgain` | 颜色抖动 | 0.5 / 0.015 / 0.7 / 0.4 |
| `affine.degrees/translate/scale/shear/perspective` | 仿射 | 0 / 0.1 / 0.5 / 2.0 / 0.0005 |
| `erasing.p` | 擦除 | 分类/属性 |
| `randaugment.*` | RandAugment | 属性 |

**`augment: {}` 空字典 = falsy = 整段不生效**；要开必须写显式键。  
增广只影响 Train，Eval 默认无。OCR 用 **`transforms` 列表**（另一套），不是这个字典。

```powershell
-o "Train.dataset.augment.mosaic=1.0"
-o "Train.dataset.augment.close_mosaic=10"
-o "Train.dataset.augment.hsv.p=0.5"
```

### 3.7 Loss / Metric / PostProcess（常用覆盖）

| 任务 | Loss 默认要点 | Metric | PostProcess 要点 |
|---|---|---|---|
| YOLO detect | `DetLoss` topk=10 alpha=0.5 cls_gain=0.5 box_gain=7.5 | `mAP50-95` | `conf_thres=0.001` `iou_thres=0.7` |
| YOLO obb | `ObbLoss` + `box_format=xywhr` | 同上 | `box_type=xywhr` |
| YOLO classify 单标签 | `CrossEntropy` | `acc`(top1) | softmax topk |
| YOLO classify **多标签** | **`-o Loss.multi_label=true`** → `MultiLabelLoss`(BCE,mean) | **`mAP`**@0.5 | sigmoid + `threshold=0.5` |
| OCR det | `DBLoss` | `hmean` | `thresh=0.3` `box_thresh=0.6` `unclip_ratio=1.5` |
| OCR rec | `CTCLoss`/`MultiLoss` | `acc` | `CTCLabelDecode` |

通用：`PostProcess.conf_thres`、`iou_thres`、`max_det`。

**多标签分类一条命令**（清单 `路径 1 0 1 0`；仓库只留模板，业务 yml 放 `configs/local/`）：

```powershell
.\tkiln.bat train -c configs/yolo/yolo11-cls.yml `
  -o Loss.multi_label=true `
  -o Architecture.Head.num_classes=4 `
  -o Train.dataset.data_dir=datasets/cls_ml_demo `
  -o Train.dataset.label_file_list=[datasets/cls_ml_demo/train.txt] `
  -o Eval.dataset.data_dir=datasets/cls_ml_demo `
  -o Eval.dataset.label_file_list=[datasets/cls_ml_demo/val.txt] `
  -o Train.dataset.multi_label=true `
  -o Eval.dataset.multi_label=true `
  -o Metric.main_indicator=mAP `
  -o PostProcess.threshold=0.5 `
  -o Global.device=cuda:0
# 也可整份复制到 configs/local/my_ml.yml 再改；尺寸覆盖例：
#   -o "Train.dataset.transform.image_size=[320,160]" `
#   -o "Eval.dataset.transform.image_size=[320,160]"
```

### 3.8 完整训练示例（yolo11n + GPU2 + 100 epoch + 学习率 + 增广）

```powershell
.\tkiln.bat train -c configs/yolo/yolo11-det.yml `
  -o Architecture.scale=n `
  -o Global.device=cuda:2 `
  -o Global.epoch_num=100 `
  -o Global.print_batch_step=10 `
  -o Global.eval_epoch_step=5 `
  -o Global.save_model_dir=./output/y11n `
  -o Global.pretrained_model=yolo11n `
  -o Optimizer.name=SGD `
  -o Optimizer.lr.learning_rate=0.01 `
  -o Optimizer.lr.name=Linear `
  -o Optimizer.lr.warmup_epoch=3 `
  -o Train.loader.batch_size_per_card=16 `
  -o Train.loader.num_workers=4 `
  -o Train.dataset.data_dir=D:/mydata/det `
  -o Train.dataset.label_file_list=train.txt `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt `
  -o Architecture.Head.num_classes=1 `
  -o Train.dataset.names=[plate] `
  -o "Train.dataset.augment.mosaic=1.0" `
  -o "Train.dataset.augment.close_mosaic=10" `
  -o "Train.dataset.augment.fliplr=0.5" `
  -o Global.amp=false `
  -o Global.use_ema=true
```

### 3.9 预训练 vs 断点续训（别搞混）

| 用途 | 参数 | 恢复什么 |
|---|---|---|
| **从预训练微调** | `Global.pretrained_model=<path\|URL\|名字>` | 只有权重；epoch 从 0 |
| **断点续训** | `Global.checkpoints=output/xxx/latest`（或 `.../latest.pth`） | 模型+优化器+调度器+EMA+epoch+step |

预训练权重：

```powershell
# 下载 OCR 预训练到缓存 ~/.torchkiln/pretrained/
python tools/download_pretrained.py
python tools/download_pretrained.py PP-OCRv6_tiny_det
python tools/download_pretrained.py --list

# 训练时直接写缓存名 / 本地路径 / ModelScope URL
-o Global.pretrained_model=PP-OCRv5_mobile_det
-o Global.pretrained_model=D:/w/best.pth
-o Global.pretrained_model=https://www.modelscope.cn/models/ChaoII0987/TorchKiln/resolve/master/pretrained/PP-OCRv5_mobile_det.pth
```

缓存目录：`~/.torchkiln/pretrained/`（**不再**找旧的 `~/.pytorchocr/`）。  
环境变量：`PYTORCHOCR_HOME` / `PYTORCHOCR_PRETRAINED_DIR` / `PYTORCHOCR_AUTO_DOWNLOAD=0`。

### 3.10 多卡（真写法）

```powershell
# DDP（推荐）：是否分布式只看 WORLD_SIZE，设备只读 Global.device
torchrun --nproc_per_node=4 tools/train.py -c configs/yolo/yolo11-det.yml `
  -o Global.device=gpu:0,1,2,3 -o Train.loader.batch_size_per_card=4

# Windows 单进程 DP
python tools/train.py -c configs/yolo/yolo11-det.yml -o Global.device=gpu:0,1
```

**`-o Global.distributed=true -o Global.gpu=0,1` 无效（死键）**。  
`device` 必须写 `cuda:0` / `gpu:0`，写 `0` 会当字符串解析到 CPU。

### 3.11 训练产物（`save_model_dir` 下）

| 文件 | 含义 |
|---|---|
| `best_accuracy.pth` | 验证最好的权重（开 EMA 时是 EMA） |
| `latest.pth` | 最新完整状态（续训用） |
| `epoch_N.pth` / `final.pth` | 定期 / 最终 |
| `train.log` | 日志 |
| `config.yml` | **本次实际生效配置（含所有 -o）** |

### 3.12 训练日志怎么看

每 `print_batch_step` 步一行：

```text
epoch: [1/100], global_step: 10, lr: 0.010000, loss: 12.3, ... ips: 50, eta: 0:01:00
```

| 字段 | 含义 |
|---|---|
| `lr` | 当前学习率 |
| `loss` | 平滑窗口内平均 |
| `avg_reader_cost` | 等数据时间；≈`avg_batch_cost` 说明 **GPU 在等数据** |
| `ips` | 每秒样本数 |

评估打印 `cur metric`（本次）和 `best metric`（历史最好）。

---

## 4. 评估 `tkiln val`（完整参数）

### 4.1 命令行参数

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `-c` / `--config` | **是** | — | 配置 |
| `-o` / `--opt` | 否 | None | 覆盖 |
| `--weights` | 否 | None | 要评的权重；**写入** `Global.pretrained_model` |

### 4.2 内部强制行为（小白必知）

| 强制项 | 值 | 原因 |
|---|---|---|
| `epoch_num` | **0** | 只评估不训练 |
| `use_ema` | **false** | 评**原始权重**，不评 EMA |
| `save_model_dir` | 缺省 `./output/_eval` | |

无 Eval 数据集 → warning + 直接结束。

### 4.3 完整示例

```powershell
.\tkiln.bat val -c configs/yolo/yolo11-det.yml `
  --weights output/y11n/best_accuracy.pth `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt `
  -o Eval.loader.batch_size_per_card=8 `
  -o Eval.loader.num_workers=0 `
  -o Global.device=cuda:2
```

OCR det：

```powershell
.\tkiln.bat val -c configs/ocr/det/PP-OCRv5_mobile_det.yml `
  --weights output/my_det/best_accuracy.pth `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt
```

输出：`cur metric, ... fps` + `main indicator (mAP50-95): ...`。

### 4.4 注意

- 大图/慢评估：`Eval.loader.num_workers=0`，避免 Windows 内存爆。
- 想和别人比 mAP：**用同一评估器评双方权重**（把对方 `.pth` 丢进本命令）。
- `--weights` 支持本地路径 / 缓存名 / URL（经 resolve）。

---

## 5. 推理 `tkiln predict`（完整参数）

**只支持单张图片**。没有 `--image_dir`，没有目录批推理。

### 5.1 YOLO 族（detect / seg / obb / pose / cls / sem / depth / 车牌 / 属性）

脚本：`tools/infer/predict_yolo.py`

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `-c` / `--config` | **是** | — | 配置 |
| `-o` / `--opt` | 否 | None | 覆盖 |
| `--weights` | **是** | — | checkpoint `.pth`（路径/名/URL） |
| `--input` | **是** | — | **单张图路径**（不是目录） |
| `--output` | 否 | `output/yolo_result.jpg` | 可视化输出路径 |
| `--device` | 否 | **`cuda:0`** | `cuda:2` / `cpu`；非 cuda 前缀或无 GPU → CPU |

```powershell
.\tkiln.bat predict -c configs/yolo/yolo11-det.yml `
  --weights output/y11n/best_accuracy.pth `
  --input D:/img.jpg `
  --output output/yolo_result.jpg `
  --device cuda:2 `
  -o PostProcess.conf_thres=0.25 `
  -o PostProcess.iou_thres=0.45
```

行为：

- 图按 `Train.dataset.transform.image_size` letterbox，结果映射回原图；
- 按 `Architecture.task` 自动画框/掩码/关键点/伪彩；
- 类别名取 `Train.dataset.names`；
- 权重按名字+形状过滤加载（`strict=False`）。

### 5.2 OCR 文本检测

脚本：`tools/infer/predict_det.py`

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `-c` | 是 | — | |
| `-o` | 否 | None | |
| `--weights` | 是 | — | `.pth` |
| `--input` | 是 | — | 单图 |
| `--output` | 否 | **`output/det_result.jpg`** | 画框图 |
| `--device` | 否 | `cuda:0` | |

内部：限长 **960**（`limit_type=max`）、ImageNet 归一化。  
```powershell
.\tkiln.bat predict -c configs/ocr/det/PP-OCRv5_mobile_det.yml `
  --weights output/my_det/best_accuracy.pth `
  --input D:/img.jpg --output output/det_vis.jpg --device cuda:2
```

### 5.3 OCR 文本识别

脚本：`tools/infer/predict_rec.py`

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `-c` | 是 | — | |
| `-o` | 否 | None | |
| `--weights` | 是 | — | |
| `--input` | 是 | — | 单图（切好的文字条） |
| `--device` | 否 | `cuda:0` | |
| ~~`--output`~~ | — | — | **没有这个参数**；结果只打印 stdout |

输入尺寸：`Global.d2s_train_image_shape`（OCR rec 常见 `[3,48,320]`）。

```powershell
.\tkiln.bat predict -c configs/ocr/rec/PP-OCRv5_mobile_rec.yml `
  --weights output/my_rec/best_accuracy.pth --input D:/crop.jpg --device cuda:2
# 打印: <路径> -> 识别文本
```

### 5.4 等价底层调用

```powershell
python tools/infer/predict_yolo.py -c <cfg> --weights <pth> --input img.jpg --output out.jpg --device cuda:2
python tools/infer/predict_det.py   -c <cfg> --weights <pth> --input img.jpg --output out.jpg
python tools/infer/predict_rec.py   -c <cfg> --weights <pth> --input img.jpg
```

---

## 6. 导出 `tkiln export`（完整参数）

### 6.1 命令行参数（argparse 真值）

| 参数 | 必填 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| `-c` / `--config` | **是** | str | — | 配置 |
| `-o` / `--opt` | 否 | list | None | 覆盖 |
| `--weights` | **是** | str | — | 训练得到的 `.pth`（可 resolve URL/缓存名） |
| `--save-dir` | **是** | str | — | 输出目录（自动创建） |
| `--fuse` | 否 | store_true | **False** | 融合 rep / Conv+BN |
| `--onnx` | 否 | store_true | **False** | 额外导出 `model.onnx` |
| `--slim` | 否 | store_true | **False** | 对**已存在的** `model.onnx` 跑 onnxslim → `model_slim.onnx` |
| `--opset` | 否 | int | None→YOLO **18** / OCR **11** | ONNX opset |
| `--legacy-exporter` | 否 | store_true | False | `dynamo=False` 旧导出器 |
| `--torchscript` | 否 | store_true | **True（恒开）** | 导出 `model.pt`；**没有关闭开关** |

### 6.2 产物固定顺序

1. **`inference.pth`** — state_dict（若 `--fuse` 则先融合再存）
2. **`inference.yml`** — **原 yml 的拷贝，不合并 `-o`！**
3. **`model.pt`** — TorchScript（`torch.jit.trace`）
4. **`model.onnx`** — 仅当 `--onnx`；输入名 `x`，batch 维动态
5. **`model_slim.onnx`** — 仅当 `--slim` 且已有 onnx

### 6.3 完整示例

```powershell
$env:PYTHONIOENCODING="utf-8"

.\tkiln.bat export -c configs/yolo/yolo11-det.yml `
  --weights output/y11n/best_accuracy.pth `
  --save-dir output/export/y11n `
  --fuse --onnx --slim --opset 18 `
  -o Architecture.Head.num_classes=1
```

### 6.4 注意（必看）

| 坑 | 说明 |
|---|---|
| **`--slim` 不会自动 `--onnx`** | 只给 `--slim` 会 onnxslim 失败（异常常被吞）。要 **`--onnx --slim` 一起** |
| **`inference.yml` 不含 `-o`** | 若用 `-o` 改了类别数等，部署配置要自己同步，或先写进一份实验 yml 再导 |
| **端到端无 NMS** | `Head.end2end: true`（v10/v26）的 ONNX **图内没有 NMS**；非端到端要部署侧自己 NMS |
| 权重形状不匹配 | 按名字+形状过滤，`strict=False`，形状不对会**静默跳过** |
| opset | 不写：YOLO=18、OCR=11 |
| 输入尺寸 | dummy 取 `Train.dataset.transform.image_size`（YOLO）或 `d2s_train_image_shape`（OCR rec） |

---

## 7. check / data / 自检

### 7.1 `tkiln check`（只建模型）

```powershell
.\tkiln.bat check -c configs/yolo/yolo11-det.yml
```

- 需要 `-c`；**不用** `--weights`（会把 pretrained 置 None）
- 打印：config / model_family / task / model / loss / metric / postprocess + summary  
- **不训练、不读数据集图片**（快速验证 yml 和 `-c` 是否有效）

### 7.2 `tkiln data`

```powershell
tkiln data list
tkiln data get <name> [--force]
tkiln data get --all
```

实现：`torchkiln/datasets.py`。**没有** `tools/download_dataset.py`。  
`manifest.yml` 里 `url: null` 时会拼 `prefix/<name>.zip`，没上传就 404 → 本地用 `make_demo_data.py`。

### 7.3 改代码后自检

```powershell
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"

python tools/smoke_all.py           # 期望 76 OK, 0 FAIL
python tools/check_graph_build.py   # 期望 53 configs: 53 OK, 0 FAIL
python tools/check_plate_models.py  # 车牌：检测 500/500、识别 86/86
```

### 7.4 下载预训练

```powershell
python tools/download_pretrained.py                 # 全部 17 个 PP-OCR
python tools/download_pretrained.py PP-OCRv6_tiny_det PP-OCRv5_mobile_rec
python tools/download_pretrained.py --list          # 只看解析到哪
```

---

## 8. 模型配置对照表（选哪个 yml）

### 8.1 YOLO 系（`configs/yolo/*.yml`）

| 任务 | 配置示例 | scale |
|---|---|---|
| detect | `yolo11-det.yml` / `yolov8-det.yml` / `yolo12-det.yml` / `yolo26-det.yml` | n/s/m/l/x |
| segment | `yolo11-seg.yml` / `yolo26-seg.yml` | 同上 |
| obb | `yolo11-obb.yml` / `yolo26-obb.yml` | 同上 |
| pose | `yolo11-pose.yml` | 同上 |
| classify | `yolo11-cls.yml` | 同上 |
| **classify 多标签** | 任意 cls 配置 + **`-o Loss.multi_label=true`**（见 §3.7） | 同上 |
| semantic | `yolo26-sem.yml` | |
| depth | `yolo26-depth.yml` | |

命名：`configs/yolo/` **仅** `*.yml`（小写+横线）；业务实验放 `configs/local/`（gitignore，不入库）。
档位不进文件名，用 `-o Architecture.scale=n|s|m|l|x`。

YOLO 系典型默认（`yolo11-det`）：SGD lr=0.01 Linear、batch=8、image_size=320、EMA on、demo 数据 `datasets/det_demo`、`Head.num_classes=2`、`reg_max=1`。

### 8.2 OCR（`configs/ocr/det` 9 + `configs/ocr/rec` 8）

| 类型 | 配置 | 典型默认 |
|---|---|---|
| det | `PP-OCRv{3,4,5,6}_*_det.yml` | epoch 500、Adam lr=0.001 Cosine、EMA off、预训练 URL、`print_batch_step=100` |
| rec | `PP-OCRv{3,4,5,6}_*_rec.yml` | 需字符集（v5/v6 常自动选） |

配置名**没有** `ch_` 前缀（如 `PP-OCRv4_mobile_det.yml`，不是 `ch_PP-OCRv4_det_mobile.yml`）。

### 8.3 车牌 / 属性

| 任务 | 配置 | 额外字段 |
|---|---|---|
| plate_det | `configs/plate/plate_det.yml` | `kpt_shape=[4,2]`、`kpt_label=4`、`num_classes=2` |
| plate_rec | `configs/plate/plate_rec.yml` | `image_size=[48,168]`、`color_num=5` |
| 行人属性 | `configs/attr/pedestrian_attribute.yml` | `label_ratio`、26 维 |
| 车辆属性 | `configs/attr/vehicle_attribute.yml` | 19 维 |

### 8.4 各任务数据字段速查

| 任务 | 清单/标签 | 必须额外 `-o` |
|---|---|---|
| YOLO detect | `labels/`：`cls cx cy w h` | `Head.num_classes` + `names` |
| YOLO **obb** | **`cls` + 4 角点 = 9 个数** | `dataset.box_format=xywhr`（**6 字段 angle 会整行丢弃**） |
| YOLO pose | `cls cx cy w h px py v...` | `kpt_shape` 写三处 |
| YOLO seg | 多边形 | `transform.mask_stride=4` |
| OCR det | `路径\tJSON四点` | `dataset.name=SimpleDataSet` + `transforms:` |
| OCR rec | `路径\t文本` | `Global.character_dict_path` |
| YOLO classify | `路径 类别号`（单标签）；**多标签** `路径 v1..vC` | `-o Loss.multi_label=true` + `Head.num_classes` |
| YOLO classify | `路径 类别号`（单标签）；**多标签** `路径 v1..vC` | `-o Loss.multi_label=true` + `Head.num_classes` |
| YOLO classify | `路径 类别号`（单标签）；**多标签** `路径 v1..vC` | `-o Loss.multi_label=true` + `Head.num_classes` |
| 属性 | `路径 v1..vC` | `label_ratio` + `label_list` |

模板：`python tools/make_format_examples.py` → `datasets/_format_examples/`。  
格式权威：`docs/DATASET_FORMATS.md`。

---

## 9. Windows / 显存注意（不看会卡死）

| 项 | 建议 |
|---|---|
| 线程 | `$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"` |
| `num_workers` | Train **0~4**、Eval **0**；≥10 → `WinError 1455` / OpenBLAS error |
| `device` | **`cuda:2`** 不是 `2`；预测默认 `cuda:0` |
| 显存 16GB | `batch_size_per_card` 别开太大（占 ≤1/4 显存较稳）；OOM 先降 batch |
| 两训练 | 不要同时占两张训练进程同一 GPU |
| 残留进程 | 结束卡住/崩溃后 `Get-Process python` 检查，必要时 `Stop-Process` |
| PowerShell 长命令 | 用反引号 `` ` `` 续行；或写进 `.ps1` |

---

## 10. 常见错误速查

| 报错 / 现象 | 原因与解法 |
|---|---|
| `device` 到了 CPU | 写了 `2` → 改 **`cuda:2`** |
| 预测 `unrecognized arguments: --image_dir` | 参数是 **`--input` 单图** |
| `Global.distributed` 多卡没生效 | 死键 → **`torchrun` + `Global.device`** |
| OBB 框数为 0 | 标签必须 **9 字段角点**，不是 `cls cx cy w h angle` |
| `--slim` 没有 slim 产物 | 先 **`--onnx`**，二者一起给 |
| `export` 后 yml 不对 | `inference.yml` **不含 -o** |
| 找不到 `ch_PP-OCR...yml` | 实为 `PP-OCRv4_mobile_det.yml` 等 |
| `label_file_list` 找不到 | 相对 `data_dir`，或已存在绝对路径；**没有 train_list 字段** |
| 增广没生效 | `augment: {}` 是空 falsy；要写显式键 |
| `KeyError: Optimizer` | 配置必须带 `Optimizer` 段（YOLO graph 已有） |
| 权重 `missing/unexpected≠0` | 分类头类别不同会跳过；`reg_max` 不一致回归头整体跳过 |
| 加载 `.pt` 报 unpickling 错 | 先 `dump_ultralytics.py` 转 state_dict |
| `data get` 404 | manifest url 未填 → `make_demo_data.py` |
| OCR 不可复现 | `seed` 需为 `None`（Paddle 语义） |

---

## 11. 四阶段最小命令备忘（可撕下来贴显示器）

```powershell
# 1) 训练
.\tkiln.bat train -c <配置.yml> `
  -o Global.device=cuda:2 -o Global.epoch_num=100 -o Global.print_batch_step=10 `
  -o Global.save_model_dir=./output/exp `
  -o Train.dataset.data_dir=D:/data -o Train.dataset.label_file_list=train.txt `
  -o Eval.dataset.data_dir=D:/data -o Eval.dataset.label_file_list=val.txt

# 2) 评估（强制不用 EMA）
.\tkiln.bat val -c <配置.yml> --weights output/exp/best_accuracy.pth `
  -o Eval.dataset.data_dir=D:/data -o Eval.dataset.label_file_list=val.txt

# 3) 推理（单图）
.\tkiln.bat predict -c <配置.yml> --weights output/exp/best_accuracy.pth `
  --input D:/a.jpg --output output/r.jpg --device cuda:2

# 4) 导出
.\tkiln.bat export -c <配置.yml> --weights output/exp/best_accuracy.pth `
  --save-dir output/export/exp --onnx --slim
```

---

## 12. 相关文档

| 文档 | 内容 |
|---|---|
| [`TRAINING.md`](TRAINING.md) | 命令手册（与本文互补，结构相同） |
| [`DATASET_FORMATS.md`](DATASET_FORMATS.md) | 各任务标签格式 + 从零接入 §0 |
| [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) | 配置字段全表 |
| [`MODEL_ZOO.md`](MODEL_ZOO.md) | 模型总表与指标 |
| [`FAQ.md`](FAQ.md) | 坑位 FAQ |
| [`../datasets/README.md`](../datasets/README.md) | 数据集清单与 `tkiln data` |
| [`../README.md`](../README.md) | 项目总览（OCR 细节 §4~§9） |
