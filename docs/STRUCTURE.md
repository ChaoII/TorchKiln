# 项目结构

> **2026-09 结构变更(重要)**
> * 多任务包 pytorchyolo/ → **pytorchx/**(含 YOLO 七任务、上游 YAML 图模型、车牌、属性);
> * OCR 包 pytorchocr/ → **pytorchx/ocr/**;顶层 pytorchocr/ 仅保留**兼容 shim**(pytorchocr.* 自动别名到 pytorchx.ocr.*);
> * 任务适配器按任务分家:**pytorchx/tasks/<task>.py**(classify/detect/obb/segment/pose/semantic/depth/plate_det/plate_rec/attribute)
>   + pytorchx/tasks/__init__.py::TASK_REGISTRY / get_task();pytorchx/task.py 变成兼容重导出层;
> * 命令行:python -m pytorchx <task> <mode> [args](	rain|val|export|predict|check)。
> 下面树中标注的 pytorchyolo/、pytorchocr/ 路径请按上述映射阅读(树本身待下一轮刷新)。

`E:\PytorchOCR` 是一个**统一的 PyTorch 训练/推理平台**,同时承载三条产品线:

| 产品线 | 包 | 说明 |
|---|---|---|
| OCR(文本检测/识别) | `pytorchx/ocr/` | PaddleOCR 复现:PP-OCRv2~v6 共 17 个模型 |
| YOLO 家族(7 任务) | `pytorchx/` | 自研 YOLO 实现(det/seg/obb/pose/cls/sem/depth)+ 上游 YAML 图模型(53 个配置) |
| 车牌 / 属性识别 | `pytorchx/`(plate\_\*, attr) | 上游车牌原版复刻(检测+识别)与 PaddleX 行人/车辆属性识别 |

三者共用平台层 `ptcore/`(配置、优化器、EMA、精度、训练循环、任务抽象、工厂)。

```
E:\PytorchOCR
├─ README.md                    总入口文档(含 OCR/YOLO/车牌/属性全部说明)
├─ docs/                        详细文档(本目录)
│  ├─ STRUCTURE.md              本文件:目录树 + 模块职责
│  ├─ MODEL_ZOO.md              全部模型 / 配置 / 数据格式 / 指标
│  ├─ TRAINING.md               训练·评估·导出·推理 + 训练特性 + 自检
│  ├─ CONFIG_REFERENCE.md       配置项全解(Global/Architecture/Optimizer/Loss/Metric/PostProcess/Train/Eval)
│  └─ FAQ.md                    常见问题与 Windows 环境坑
│
├─ ptcore/                      平台层(与任务无关)
│  ├─ config.py                 配置加载 / -o 覆盖 / merge
│  ├─ pretrained.py             权重解析(URL/本地/缓存)+ 按名加载
│  ├─ optimizer.py              优化器 + 调度器(Adam/AdamW/SGD/RAdam/NAdam × Cosine/Linear/OneCycle/Const)
│  ├─ precision.py              AMP(fp16/bf16)+ GradScaler
│  ├─ ema.py                    EMA 权重
│  ├─ batch_sampler.py          与 Paddle 一致的乱序采样(含多卡分片)
│  ├─ task.py                   TaskAdapter 抽象(pytorchx / pytorchocr 各自实现)
│  ├─ trainers/                 每个任务一个 trainer(共享 base 训练循环)
│  │  ├─ base.py                BaseTrainer:训练循环/评估/EMA/断点续训/多卡/日志/保存
│  │  ├─ detect.py segment.py pose.py classify.py obb.py semantic.py depth.py
│  │  ├─ plate_det.py plate_rec.py attribute.py pose_action.py video_cls.py
│  │  ├─ ocr.py                 OcrTrainer(det/rec 共用,内部由 OcrTask 分派)
│  │  └─ __init__.py            TRAINER_REGISTRY / get_trainer() / build_trainer(config)
│  ├─ trainer.py                兼容 shim(重导出 ptcore.trainers.base.BaseTrainer)
│  ├─ factory.py                build_trainer(config):按 model_family/task 分派
│  └─ __init__.py
│
├─ pytorchx/ocr/                  OCR 产品线(PaddleOCR 复现)
│  ├─ base_ocr_v20.py           模型基类
│  ├─ task.py                   OCR 任务的 TaskAdapter(det/rec/cls/e2e/sr/table...)
│  ├─ trainer.py                OCR Trainer(基于 ptcore.BaseTrainer)
│  ├─ modeling/
│  │  ├─ architectures/         base_model.py(按配置组装)/ uvdoc_model.py
│  │  ├─ backbones/             PPLCNetV3/V4、MobileNetV3、ResNet_vd、PPHGNet(_V2)、
│  │  │                         MobileNetV1Enhance、RecSVTR、NRTR-MTB、DenseNet、Donut-Swin...
│  │  ├─ necks/                 DBFPN、RSEFPN、LKPAN、RepLKFPN、RepLKPAN、FPN、RNN、SAST、EAST、FCE、PG、Table
│  │  ├─ heads/                 DBHead、PFHeadLocal、MultiHead、CTCHead、NRTR(Transformer)、SARHead、
│  │  │                         Attention、CAN、Cls、SAST/EAST/FCE/PSE/Table/ViTSTR...
│  │  └─ transforms/            TPS、STN、TBSRN、TSRN
│  ├─ data/
│  │  ├─ imaug/                 DecodeImage、DetLabelEncode、CopyPaste、IaaAugment、EastRandomCropData、
│  │  │                         ColorJitter、RandomPerspective、MakeBorderMap、MakeShrinkMap、RecResizeImg、
│  │  │                         RecAug、RecConAug、MultiLabelEncode...
│  │  ├─ simple_dataset.py      数据集
│  │  ├─ paddle_batch_sampler.py 与 Paddle 一致的采样器(含多卡)
│  │  └─ multi_scale_sampler.py 多尺度采样
│  ├─ losses/                   DBLoss 系列、CTCLoss、MultiLoss、NRTRLoss、SARLoss
│  ├─ metrics/                  DetMetric(hmean)、RecMetric(acc)、eval_det_iou
│  ├─ postprocess/              DBPostProcess(Paddle 原版)、CTCLabelDecode、EAST/SAST/FCE/PSE/PG/Table、poly_nms
│  ├─ optimizer/                兼容 shim(实现已移入 ptcore.optimizer)
│  └─ utils/                    config、pretrained、ema、precision(均为 ptcore 的兼容 shim)+ dict/ 字符集
│
├─ pytorchx/                 YOLO 家族 + 车牌 + 属性
│  ├─ trainer.py                Trainer + build_task(task → TaskAdapter 分派)
│  ├─ task.py                   YOLO 各任务的 TaskAdapter(cls/det/obb/seg/pose/sem/depth)
│  ├─ det/                      检测核心
│  │  ├─ ops.py                 xyxy/xywh、IoU、CIoU、make_anchors、dist2bbox、DFL 投影、split_head、NMS、letterbox
│  │  ├─ loss.py                DetLoss(TAL 分配 + BCE/CIoU/DFL)、ObbLoss
│  │  ├─ postprocess.py         DetPostProcess(Anchor-free 解码 + 分类别 NMS + end2end 跳过 NMS)
│  │  ├─ metric.py              COCO 风格 mAP(per-class, IoU 0.5:0.95)
│  │  └─ rbox.py                旋转框几何(ProbIoU / poly IoU)
│  ├─ data/
│  │  ├─ det.py                 DetDataset(检测/旋转框)
│  │  ├─ seg.py / pose.py / sem.py / depth.py   各任务数据集
│  │  ├─ plate.py               车牌识别数据集(48×168 + Caffe 归一化)
│  │  └─ augment.py             HSV / Affine(框·关键点·掩码同一 homography)/ mosaic4 / mosaic9 /
│  │                           mixup / copy_paste / RandomErasing / TrainAugmenter / DenseAugmenter
│  ├─ nn/
│  │  ├─ modules.py             模块库 + 注册表(Conv/C2f/C3/C3k2/PSA/Ghost*/Rep*/SPP*/Concat/Detect 家族/
│  │  │                         Segment/OBB/Pose/Proto/Classify/SemanticSegment/Depth/AreaAttention/A2C2f/
│  │  │                         ADown/AConv/ELAN1/SCDown/CBLinear/CBFuse/ResNetLayer/TorchVision)
│  │  ├─ graph.py               YAML 图解析器 + GraphModel(负数索引/Concat/缩放/depth_multiple/stride 探测)
│  │  ├─ plate.py               上游车牌原版模块:StemBlock/ShuffleV2Block/BottleneckV5/C3V5/
│  │  │                         PlateDetect(anchor+4角点)/PlateRecNet(myNet_ocr_color)+字符集·颜色
│  │  └─ attribute.py           PP-LCNet_x1_0 + 多标签头(AttributeNet)
│  ├─ models/
│  │  ├─ __init__.py            build_model / build_arch_model(YAML 图 或 手写模型)
│  │  ├─ det.py                 YOLO 手写检测模型
│  │  ├─ seg.py / sem.py / depth.py
│  │  └─ plate.py               车牌识别模型构建 + 车牌 YAML 路径
│  ├─ cfg/models/               上游 YAML 原样搬运:11/ 12/ 26/ v3/ v5/ v6/ v8/ v9/ v10/ plate/
│  ├─ plate_det.py              车牌检测任务:PlateDetLoss(v5 anchor+WingLoss)/PostProcess/Metric/Task
│  ├─ plate_rec.py              车牌识别任务:PlateRecLoss(CTC+颜色CE)/PostProcess/Metric/Task
│  ├─ attr.py                   属性识别任务:AttributeDataset/MultiLabelLoss/AttrMetric/PostProcess/Task
│  ├─ attr_randaug.py           RandAugment(对齐 TimmAutoAugment rand-m9-mstd0.5-inc1)
│  ├─ pose.py / seg.py / sem.py / depth.py   各任务的 Loss/Metric/PostProcess 构建入口
│  └─ utils/                    属性名表、字符集等资源文件
│
├─ configs/                     可直接训练/评估的配置
│  ├─ det/ (9)  rec/ (8)        OCR
│  ├─ yolo/ (59)                YOLO:53 个 YAML 图配置 + 6 个手写模型
│  ├─ plate/ (2)                plate_det.yml / plate_rec.yml
│  ├─ attr/ (2)                 pedestrian_attribute.yml / vehicle_attribute.yml
│  └─ debug/ (5)                调试用小配置
│
├─ tools/
│  ├─ train.py / eval.py / export.py        三个通用 CLI(按 model_family 自动分派)
│  ├─ smoke_all.py                           全量自检:每个配置 1 步训练 + 1 步评估(当前 80 OK)
│  ├─ check_graph_build.py                   只建图+前向:校验 53 个 YAML 图配置
│  ├─ check_plate_models.py                  车牌:上游权重/ONNX 张量与数值对齐校验
│  ├─ download_pretrained.py                 预下载 OCR 预训练权重到本地缓存
│  ├─ gen_configs.py / make_paddle_cfg.py    由官方配置生成 configs/*
│  ├─ infer/                                 predict_det.py / predict_rec.py / predict_yolo.py(七任务可视化)
│  └─ convert/                               权重转换与数值对比
│     ├─ dump_all.py / dump_paddle_sd.py     (paddlex 环境)导出 .pdparams → numpy pickle
│     ├─ convert_all.py / convert_det.py     (ptocr 环境)pickle → torch .pth(OCR)
│     ├─ dump_ultralytics.py                 (ultralytics 环境)上游 .pt → 纯张量 state_dict
│     ├─ convert_plate_weights.py            车牌:上游 .pt/.pth → torch .pth
│     ├─ dump_attribute_paddle.py            (paddlex 环境)属性 .pdparams → pickle
│     ├─ convert_attribute_weights.py        属性:pickle → torch .pth(含 _mean→running_mean、Linear 转置)
│     ├─ parity_*.py / paddle_forward.py     Paddle↔PyTorch 逐层数值对比工具
│     └─ test_*                             旧调试脚本(可保留)
│
├─ datasets/                    示例数据集(可直接跑通训练)
│  ├─ ocr_det_dataset_examples / ocr_rec_dataset_examples      OCR(官方示例)
│  ├─ det_demo / obb_demo / seg_demo / pose_demo / cls_demo / sem_demo / depth_demo
│  ├─ plate_det_demo / plate_rec_demo                          车牌合成样例
│  └─ pedestrian_attribute_demo / vehicle_attribute_demo       属性合成样例
│
├─ _downloads/                  第三方源码与权重(不入库逻辑,仅本地缓存)
│  ├─ official/                 官方权重 + 转换后的 *_ptocr.pth(OCR 与属性)
│  ├─ upstream/                 上游 YOLO 权重/配置(user_det_state.pth、user_det.yaml)
│  ├─ plate/                    车牌转换后的 state_dict
│  ├─ Chinese_license_plate_detection_recognition/  车牌上游仓库(含 weights/)
│  ├─ PaddleOCR/ PaddleOCR2Pytorch/ paddle_cfg/     参考实现与配置
│  └─ det_v4/                   参考权重
│
├─ output/                      训练/评估/导出产物(日志、best_accuracy.pth、inference、ONNX)
└─ wheels/                      离线依赖轮子
```

## 依赖方向(单向,无环)

```
configs/*.yml ──> ptcore.config ──> ptcore.factory ──> ptcore.trainers.build_trainer
                                                           │
                              ┌────────────────────────────┴───────────────────────┐
                              ▼                                                    ▼
                   ptcore.trainers.<task>.XxxTrainer                    ptcore.trainers.ocr.OcrTrainer
                   (共享 ptcore.trainers.base.BaseTrainer)              (共享 ptcore.trainers.base.BaseTrainer)
                              │                                                    │
                              ▼                                                    ▼
                   pytorchx.tasks.<task>.Yolo<Task>Task                pytorchx.ocr.task.OcrTask
                              │                                                    │
                              ▼                                                    ▼
       pytorchx.{nn,models,det,data,plate_*,attr}            pytorchx.ocr.{modeling,losses,metrics,...}
                              │                                                    │
                              └──────────► ptcore 共享能力(precision/optimizer/ema/pretrained)◄──────────┘
```

* `ptcore` **不反向依赖**任何产品线(OCR/YOLO 均可独立替换)。
* `pytorchx/ocr/utils/*` 与 `pytorchx/ocr/optimizer/` 是从旧路径保留下来的**兼容 shim**,真实现在 `ptcore/`,新代码请直接用 `ptcore.*`。

## 命名与约定

| 约定 | 说明 |
|---|---|
| 配置文件 | `configs/<family>/<name>.yml`;`Architecture.model_family ∈ {ocr, yolo}` 决定用哪个 Trainer |
| 任务名 | `Architecture.task`:`det/rec/cls/...`(OCR);`detect/segment/obb/pose/classify/semantic/depth/plate_det/plate_rec/attribute`(YOLO 系)|
| 图模型 | `Architecture.yaml_file` 指向 `pytorchx/cfg/models/**`;没有则走手写模型构建器 |
| 权重格式 | 统一为 `torch.save({'state_dict': ...})` 或纯张量 dict;加载按**名字+形状**匹配 |
| 输出目录 | `Global.save_model_dir`,内含 `best_accuracy.pth`、`latest.pth`、`train.log`、`config.yml` |
| 导出 | `tools/export.py` 产出 `inference.pth / inference.yml / model.pt / model.onnx / model_slim.onnx` |
