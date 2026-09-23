# 任务(task)体系:现在怎么组织的 / 怎么调整

平台有 4 条产品线、12 个任务(OCR det/rec、YOLO 7 任务、车牌 2、属性 1)。它们**不是各写一套训练循环**,
而是统一收敛到「一个抽象 + 两级分派 + 每任务 6 个组件」。

---

## 1. 两级分派:先选 Trainer,再选 TaskAdapter

```
configs/xxx.yml
  └─ Architecture.model_family / task ──► ptcore.trainers.build_trainer()
        ├─ "ocr"(或无 family/task) ──► ptcore.trainers.ocr.OcrTrainer
        │                                  └─ torchkiln.ocr.task.OcrTask(det/rec 内部分派)
        └─ task=detect/segment/... ──► ptcore.trainers.<task>.XxxTrainer
                                           └─ torchkiln.tasks.get_task(task) ──► TaskAdapter 实例
```

`ptcore/trainers/base.py::BaseTrainer` 负责**与任务无关的一切**:设备/种子、按 family 建模型、
加载预训练、AMP、优化器/调度器、EMA、断点续训、训练循环、评估循环、日志与保存、多卡(DP/DDP)。
任务差异全部被 `TaskAdapter` 的 11 个钩子吸收;每个任务有**自己的 trainer 文件**
(`ptcore/trainers/<task>.py`),共享 `BaseTrainer`,注册表在 `ptcore/trainers/__init__.py`。

`ptcore/trainer.py` 仅保留为兼容 shim(重导出 `BaseTrainer` / `get_logger`)。
`task` 的取值见 §3 的表。

## 2. TaskAdapter 契约(实现这 11 个方法即可接入平台)

`ptcore/task.py` 定义;Trainer 的调用时机见右列。

| 钩子 | 何时调用 | 职责 |
|---|---|---|
| `build_post_process(config)` | 建模型**之前** | 造后处理器(有些头需要它决定输出通道)|
| `build_model(config, post_process)` | 初始化时 | 建模型(图模型走 `build_arch_model`,或专用构建器)|
| `build_loss(config, model)` | 建模型**之后** | 造损失(可以读模型头参数:`task_head(model)` 取真实头)|
| `build_metric(config)` | 建模型之后 | 造指标 |
| `build_datasets(config, logger)` | 之后 | 造 Train / Eval 数据集 |
| `train_collate(batch)` / `eval_collate(batch)` | DataLoader | 批次拼装(不同任务张量结构不同)|
| `forward_train(model, images, batch)` | 每步训练 | 前向(有些任务要把 batch 传进去)|
| `eval_step(model, batch, post_process, metric, device)` | 评估 | `metric(post_process(model(images)), batch)` |
| `sample_count(batch)` | 日志 | 该 batch 的样本数 |
| `summary_lines(config, global_config, post_process)` | 启动时 | 打印一行任务摘要 |

> 关键约定:任务内**不要**自己写 epoch 循环、AMP、EMA、保存逻辑 —— 那是平台层的事;
> 反过来,平台层**不认识**任何具体任务(不 import 任何 task 模块)。

## 3. 现状:每个任务的组件放在哪(`yolo` 系)

| task | TaskAdapter 类 | 数据 | 损失/指标/后处理 | 模型 |
|---|---|---|---|---|
| `classify` | `task.py::YoloClsTask` | `data/__init__.py::ClsDataset` | 内置(CE/cls 指标) | `models/__init__.py::build_model` |
| `detect` | `task.py::YoloDetTask` | `data/det.py::DetDataset` | `det/{loss,metric,postprocess}.py` | `models/det.py` 或 YAML 图 |
| `obb` | `task.py::YoloObbTask` | `data/det.py`(box_format=xywhr)| `det/loss.py::ObbLoss` + `det/rbox.py` + `det/metric.py` | 手写/YAML 图 |
| `segment` | `task.py::YoloSegTask` | `data/seg.py::SegDataset` | `seg.py::build_seg_{loss,postprocess,metric}` | `models/seg.py` 或图 |
| `pose` | `task.py::YoloPoseTask` | `data/pose.py::PoseDataset` | `pose.py::build_pose_*` | 图模型为主 |
| `semantic` | `task.py::YoloSemTask` | `data/sem.py::SemDataset` | `sem.py::build_sem_*` | `models/sem.py` |
| `depth` | `task.py::YoloDepthTask` | `data/depth.py::DepthDataset` | `depth.py::build_depth_*` | `models/depth.py` |
| `plate_det` | `plate_det.py::PlateDetTask` | `data/pose.py`(`kpt_shape:[4,2]`)| `plate_det.py` 内的 `PlateDetLoss/PostProcess/Metric` | `cfg/models/plate/*.yaml` 图 |
| `plate_rec` | `plate_rec.py::PlateRecTask` | `data/plate.py::PlateRecDataset` | `plate_rec.py` 内的 `PlateRecLoss/PostProcess/Metric` | `models/plate.py::build_rec_model` |
| `attribute` | `attr.py::AttributeTask` | `attr.py::AttributeDataset` | `attr.py` 内的 `MultiLabelLoss/AttrMetric/PostProcess` | `nn/attribute.py::AttributeNet` |

OCR 侧是**单适配器多任务**:`torchkiln/ocr/task.py::OcrTask` 内部再按 `task`(det/rec/cls/e2e/sr/table...)
分派到 `modeling/*`、`losses/*`、`metrics/*`、`postprocess/*`,因为 OCR 各任务共用同一套数据管线与训练细节。

## 4. 现状的三个不一致(建议调整的点)

1. **适配器位置不统一**:6 个任务在 `torchkiln/task.py`,车牌/属性在各自模块
   (`plate_det.py` / `plate_rec.py` / `attr.py`)。
2. **组件粒度不统一**:det/obb 把损失/指标/后处理放在 `det/` 子包;**其他任务**把它们和被适配器放在同一个文件,
   或拆成 `seg.py / pose.py / sem.py / depth.py` 三个 `build_*` 函数。
3. **命名不统一**:`YoloClsTask` vs `PlateDetTask` vs `AttributeTask`;`build_*_loss` 有的返回类实例、
   有的返回 dict(如 attribute 走 `Loss` 段直接构造)。

功能上这些不影响正确性(全部通过自检),但**找代码/加任务时需要在多处跳转**。

## 5. 建议的统一结构(改动是纯搬迁 + 重导出,零功能变化)

```
torchkiln/
├─ tasks/
│  ├─ __init__.py        # TASK_REGISTRY = {"detect": YoloDetTask, ...} + get_task(name)
│  ├─ _base.py           # 任务共用工具:_class_weights_from_config + 重导出 num_classes_of
│  ├─ _cls.py            # 分类公共实现(ClsLoss/ClsPostProcess/ClsMetric + build_*/num_classes_of)
│  ├─ classify.py        # 每个任务一个文件,自包含:Dataset(引用)+ Loss + PostProcess + Metric + Task
│  ├─ detect.py  obb.py  segment.py  pose.py  semantic.py  depth.py
│  ├─ plate_det.py       # 车牌检测(原 plate_det.py)
│  ├─ plate_rec.py       # 车牌识别(原 plate_rec.py)
│  └─ attribute.py       # 属性识别(原 attr.py + attr_randaug.py)
├─ det/                  # 仍然保留:检测族的**公共**算子/损失/指标(det、obb、seg、pose 都用)
├─ data/                 # 各任务数据集(位置不变)
├─ nn/ models/           # 各任务模型(位置不变)
└─ task.py               # 变成兼容层:`from torchkiln.tasks import *` + 旧名字别名
```

迁移映射(纯移动,类名不变):

| 现在 | 调整后 |
|---|---|
| `torchkiln/task.py::YoloClsTask/YoloDetTask/YoloObbTask/YoloSegTask/YoloPoseTask/YoloSemTask/YoloDepthTask` | `torchkiln/tasks/{classify,detect,obb,segment,pose,semantic,depth}.py::<同名>Task` |
| `torchkiln/plate_det.py` | `torchkiln/tasks/plate_det.py` |
| `torchkiln/plate_rec.py` | `torchkiln/tasks/plate_rec.py` |
| `torchkiln/attr.py` + `attr_randaug.py` | `torchkiln/tasks/attribute.py`(+ 内部 import RandAugment) |
| `torchkiln/trainer.py::build_task()` 的长 if-链 | `tasks/__init__.py::get_task(task)` 查表 |
| `torchkiln/seg.py / pose.py / sem.py / depth.py` | 保留(作为公共 Loss/Metric/PostProcess 实现),被 `tasks/*.py` 引用 |

对外接口保持不变:`configs/*.yml` 不动、`Architecture.task` 取值不动、`torchkiln.task` 仍可 import。

## 6. 新增一个任务的标准流程(以 "动作识别" 为例)

1. **定契约**:输入是什么、输出几个张量、指标是什么(`docs/MODEL_ZOO.md` 的表里加一行)。
2. **数据**:`torchkiln/data/action.py::ActionDataset` + `train_collate/eval_collate`
   (`__getitem__` 返回 `[img, ...targets]`;单样本失败返回 `[]`,平台会自动跳过)。
3. **模型**:`torchkiln/models/action.py::build_model(arch)`;若是 YAML 图模型,把 yaml 放
   `torchkiln/cfg/models/<family>/` 并在 `nn/modules.py` 注册新模块(或复用现有头)。
4. **损失/指标/后处理**:`torchkiln/action.py`(或并入 `tasks/action.py`)实现三个类,
   并提供 `build_action_{loss,metric,postprocess}`。
5. **适配器**:`torchkiln/tasks/action.py::ActionTask(TaskAdapter)`,实现 §2 的 11 个钩子
   (最省事的模板:照抄 `tasks/attribute.py`,它同时覆盖了"图像 + 多标签"和"自定义数据集"两种形态)。
6. **注册**:`tasks/__init__.py::TASK_REGISTRY["action"] = ActionTask`(一行)。
7. **配置 + 自检**:`configs/action/xxx.yml`(照抄同族配置)→ `python tools/smoke_all.py`
   应保持全绿(它会自动把新配置纳入)。

> 平台对任务的唯一要求就是这 4 样:**数据集 → 模型 → 损失/指标/后处理 → 适配器**,
> 其余(AMP/EMA/续训/多卡/日志/导出)全部继承。

---

## 7. 按任务分家(ultralytics 形态)—— 目标结构与命令行

现状(§3)是"适配器散在多处",目标是把**每个任务做成一个自包含子包**,命令行也按
`yolo <task> <mode>` 暴露。**任务列表与目录一一对应**:

### 7.1 命令行(已实现:`python -m torchkiln <task> <mode> ...`)

```powershell
python -m torchkiln detect    train  -c configs/yolo/yolov8_graph.yml -o Global.epoch_num=100
python -m torchkiln obb       val    -c configs/yolo/yolov8-obb_graph.yml --weights output/x/best_accuracy.pth
python -m torchkiln segment   export -c configs/yolo/yolov8-seg_graph.yml  --weights ... --onnx
python -m torchkiln pose      predict -c configs/yolo/yolov8-pose_graph.yml --weights ... --input imgs
python -m torchkiln classify  train  -c configs/yolo/yolo11-cls_graph.yml
python -m torchkiln semantic  train  -c configs/yolo/yolo26-sem_graph.yml
python -m torchkiln depth     train  -c configs/yolo/yolo26-depth_graph.yml
python -m torchkiln plate_det train  -c configs/plate/plate_det.yml
python -m torchkiln plate_rec train  -c configs/plate/plate_rec.yml
python -m torchkiln attribute train  -c configs/attr/vehicle_attribute.yml
python -m torchkiln ocr_det   train  -c configs/det/PP-OCRv5_mobile_det.yml
python -m torchkiln ocr_rec   train  -c configs/rec/PP-OCRv5_mobile_rec.yml

python -m torchkiln detect    check  -c configs/yolo/yolov8_graph.yml   # 只构建 + 打印摘要(不训练)
python -m torchkiln --help
```

* `mode ∈ {train, val, export, predict, check}`;`<task>` 只做**命名空间与一致性校验**
  (真正的任务由配置里 `Architecture.task` 决定,不一致时会打印提示),其余参数**原样透传**给
  `tools/{train,eval,export}.py` / `tools/infer/predict_{yolo,det,rec}.py`。
* 实现:`torchkiln/cli.py` + `torchkiln/__main__.py`(别名表 `TASK_ALIASES`、`MODEL_FAMILY`、`MODES`)。
* 想做 `yolo` 这样的短命令,给 `setup.py`/`pyproject.toml` 加
  `console_scripts = yolo = torchkiln.cli:main` 即可(非必须,`python -m` 已可用)。

### 7.2 目标目录(每任务一个子包,内部自包含)

```
torchkiln/
├─ cli.py                  # yolo <task> <mode> 入口(已实现)
├─ engine/                 # 平台侧适配(与 ptcore 对接,任务无关)
│  ├─ trainer.py           #   (现 torchkiln/trainer.py:Trainer + 任务注册表查找)
│  └─ predictors/          #   yolo / ocr_det / ocr_rec 三种推理器(现 tools/infer/*)
├─ tasks/                  # ★ 按任务分家:每任务一个文件/子包
│  ├─ __init__.py          #   TASK_REGISTRY = {"detect": DetectTask, ...};get_task(name)
│  ├─ _base.py             #   TaskAdapter 公共工具(task_head / num_classes_of / 默认 collate)
│  ├─ detect.py            #   DetectDataset(引用 data/det.py) + DetLoss/DetMetric/DetPostProcess + DetectTask
│  ├─ segment.py           #   SegmentTask(...)
│  ├─ obb.py  pose.py  semantic.py  depth.py  classify.py
│  ├─ plate_det.py         #   PlateDetTask(含 PlateDetLoss/PostProcess/Metric)
│  ├─ plate_rec.py         #   PlateRecTask(含 PlateRecLoss/PostProcess/Metric)
│  └─ attribute.py         #   AttributeTask(含 AttributeDataset/MultiLabelLoss/AttrMetric/RandAugment)
├─ data/                   # 各任务数据集(位置不变;被 tasks/* 引用)
│  ├─ det.py seg.py pose.py sem.py depth.py plate.py __init__.py(cls) augment.py
├─ models/                 # 各任务模型构建器(位置不变;YAML 图仍走 nn/graph.py)
│  ├─ __init__.py det.py seg.py sem.py depth.py plate.py
├─ nn/                     # 模块库 + YAML 图解析 + 专用网络(位置不变)
│  ├─ modules.py graph.py plate.py attribute.py
├─ cfg/                    # 默认配置与模型 YAML(现 cfg/models/**,另加 cfg/datasets/**)
└─ det/                    # 检测族公共算子/损失/指标(detect/obb/segment/pose 共用)
```

### 7.3 迁移映射与步骤(纯搬迁 + 重导出,零功能变化)

| 现在 | 目标 |
|---|---|
| `torchkiln/task.py`(7 个 `Yolo*Task`) | 拆到 `tasks/{classify,detect,obb,segment,pose,semantic,depth}.py`(类名不变)|
| `torchkiln/plate_det.py` / `plate_rec.py` / `attr.py` / `attr_randaug.py` | `tasks/plate_det.py` / `tasks/plate_rec.py` / `tasks/attribute.py` |
| `seg.py` / `pose.py` / `sem.py` / `depth.py` 的 `build_*` | 就近并入对应 `tasks/*.py`(或保留在 `det/` 同级作为公共实现)|
| `trainer.py::build_task()` 的 if-链 | `tasks/__init__.py::get_task(task)` |
| `task_head()`(task.py)/ `_head_of()`(plate_det.py)重复实现 | `tasks/_base.py` 单份 |
| `torchkiln/task.py` | 兼容层:`from torchkiln.tasks import *`,旧 import 不破 |

步骤:① 建 `tasks/` 与 `_base.py` → ② 逐个搬适配器并改 `build_task` 查表 → ③ `task.py` 变兼容层 →
④ 跑三条基线复绿(`smoke_all` 80 OK / `check_graph_build` 53 OK / `check_plate_models`)→
⑤ 更新本文档与 `docs/STRUCTURE.md` 的路径。对外(`configs/*.yml`、`Architecture.task`、CLI)完全不变。
