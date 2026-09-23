# 训练 / 评估 / 导出 / 推理

三个 CLI 都是**通用入口**:读 `Architecture.model_family` 自动选择 OCR 或 YOLO Trainer,
再用 `Architecture.task` 选择任务适配器。所有产品线(OCR / YOLO / 车牌 / 属性)用法完全一致。

---

## 0. 快速开始(3 步)

```powershell
# 1) 训练(不改 yml,用 -o 覆盖任意字段)
python tools/train.py -c configs/det/ch_PP-OCRv4_det_mobile.yml `
  -o Global.pretrained_model=~/.torchkiln/pretrained/ch_PP-OCRv4_det_mobile_ptocr.pth `
  -o Global.epoch_num=100 -o Train.dataset.data_dir=D:/mydata/det

# 2) 评估
python tools/eval.py -c configs/det/ch_PP-OCRv4_det_mobile.yml --weights output/.../best_accuracy.pth

# 3) 推理 / 导出
python tools/infer/predict_det.py -c configs/det/ch_PP-OCRv4_det_mobile.yml --weights output/.../best_accuracy.pth --image_dir imgs
python tools/export.py -c configs/det/ch_PP-OCRv4_det_mobile.yml --weights output/.../best_accuracy.pth --onnx --slim
```

`-o` 覆盖语法:`-o A.B.C=value`,值会按 yml 里原字段的类型自动转换(int/float/bool/list);
可重复多次;`-o Train.dataset.label_file_list=[a.txt,b.txt]` 支持列表。

### 0.1 ultralytics 风格的任务/模式 CLI(推荐)

> 命令名 **tkiln**(原 `ptx`,2026-09 随仓库改名 TorchKiln;根目录提供免安装启动器 tkiln.bat / tkiln;
> pip install -e . 后可用 pyproject.toml 注册的 tkiln 入口)。下面示例里的 python -m torchkiln 与 tkiln 完全等价。

```powershell
python -m torchkiln <task> <mode> [args...]        # mode = train | val | export | predict | check

python -m torchkiln detect    train  -c configs/yolo/yolov8_graph.yml -o Global.epoch_num=100
python -m torchkiln obb       val    -c configs/yolo/yolov8-obb_graph.yml --weights output/x/best_accuracy.pth
python -m torchkiln pose      export -c configs/yolo/yolov8-pose_graph.yml --weights ... --onnx
python -m torchkiln plate_det train  -c configs/plate/plate_det.yml
python -m torchkiln plate_rec train  -c configs/plate/plate_rec.yml
python -m torchkiln attribute train  -c configs/attr/vehicle_attribute.yml
python -m torchkiln ocr_det   train  -c configs/det/PP-OCRv5_mobile_det.yml
python -m torchkiln detect    check  -c configs/yolo/yolov8_graph.yml    # 只构建 + 打印摘要
python -m torchkiln --help                                              # task/mode 一览
```

`<task>` 只做命名空间与一致性校验(真任务由配置的 `Architecture.task` 决定),其余参数与
`tools/*.py` 完全一致(同一套实现,CLI 只是分发层)。`python tools/train.py ...` 的老用法**依然可用**。

## 1. 典型训练命令

```powershell
# OCR 识别(需字符集:v5/v6 会自动选 dict)
python tools/train.py -c configs/rec/PP-OCRv5_mobile_rec.yml -o Global.epoch_num=100

# YOLO 检测(图模型,59 个配置任选)
python tools/train.py -c configs/yolo/yolov8_graph.yml -o Global.epoch_num=100

# YOLO 端到端(NMS-free)与多尺度变体
python tools/train.py -c configs/yolo/yolo26-p2_graph.yml -o Global.epoch_num=200

# 车牌:检测 + 识别(用上游官方权重微调)
python tools/train.py -c configs/plate/plate_det.yml `
  -o Global.pretrained_model=_downloads/plate/plate_detect_state.pth
python tools/train.py -c configs/plate/plate_rec.yml `
  -o Global.pretrained_model=_downloads/plate/plate_rec_color_state.pth

# 属性识别(行人 / 车辆)
python tools/train.py -c configs/attr/pedestrian_attribute.yml `
  -o Global.pretrained_model=~/.torchkiln/pretrained/PP-LCNet_x1_0_pedestrian_attribute_ptocr.pth
python tools/train.py -c configs/attr/vehicle_attribute.yml `
  -o Global.pretrained_model=~/.torchkiln/pretrained/PP-LCNet_x1_0_vehicle_attribute_ptocr.pth
```

## 2. 训练特性(与 ultralytics 对齐)

```yaml
Global:
  amp: true              # 混合精度(amp_dtype: fp16/bf16)
  freeze: 8              # 冻结前 N 层,或 ["conv1", "blocks2"]
  accumulate: 2          # 梯度累积(等效 batch ×2)
  use_ema: true          # EMA 权重
Optimizer:
  name: AdamW            # Adam / AdamW / SGD / RAdam / NAdam(Paddle 的 Momentum ≡ SGD)
  momentum: 0.937
  lr:
    name: Cosine         # Cosine / Linear / OneCycle / Const
    learning_rate: 0.01
    lrf: 0.01            # 余弦终值系数(不给则按 Paddle 语义衰减到 0)
    warmup_epoch: 3
  regularizer: {name: L2, factor: 0.0005}
Loss:
  label_smoothing: 0.05
Train:
  dataset:
    augment:
      mosaic: 1.0        # 4 图拼图
      mosaic9: 0.0       # 9 图拼图(目标更小、更多截断)
      mixup: 0.1
      copy_paste: 0.0    # 实例拷贝粘贴(需掩码 → 分割)
      close_mosaic: 60   # epoch 数后自动关闭 mosaic/mosaic9
      multi_scale: 0.3   # 每 epoch 随机尺度 ±30%
      hsv: {p: 0.5, hgain: 0.015, sgain: 0.7, vgain: 0.4}
      affine: {degrees: 0.0, translate: 0.1, scale: 0.5, shear: 2.0, perspective: 0.0005}
      erasing: {p: 0.25}                 # 分类 / 属性
      randaugment: {p: 0.8, num_ops: 2, magnitude: 9, magnitude_std: 0.5, inc: 1}   # 属性
```

几何增广对**框 / 旋转框 / 关键点 / 实例掩码**用同一 homography,不会错位。
实测增益:v8-det `mAP50-95 0.690→0.717`;v8-obb `mAP50 0.940→0.990`。

## 3. 断点续训与预训练

```powershell
# 断点续训(保留优化器/epoch/EMA)
python tools/train.py -c <cfg>.yml -o Global.checkpoints=output/exp/latest -o Global.epoch_num=80

# 预训练权重:本地路径 / URL / 已缓存名字
python tools/train.py -c <cfg>.yml -o Global.pretrained_model=output/pre/best_accuracy.pth
python tools/download_pretrained.py --models ch_PP-OCRv4_det_mobile      # 预下载到缓存
```
加载规则:**按参数名 + 形状**匹配,不匹配的跳过并打印统计(`missing / unexpected`),
支持加载上游 YOLO `.pt`(先用 `tools/convert/dump_ultralytics.py` 转)与 Paddle `.pdparams`(见 §6)。

## 4. 导出

```powershell
python tools/export.py -c configs/yolo/yolo26_graph.yml --weights output/.../best_accuracy.pth `
  --save-dir output/exp --onnx --slim
```
产出:`inference.pth`(eval 权重)、`inference.yml`(推理配置)、`model.pt`(TorchScript)、
`model.onnx`、`model_slim.onnx`(onnxslim 精简)。
* 端到端模型(v10/v26,`Head.end2end: true`)导出的 ONNX **图内没有 NMS**,部署侧无需再写后处理;
* 输入/输出契约见 `docs/MODEL_ZOO.md` 的任务表(det=`[reg,cls]`,obb/seg/pose 见各任务)。

## 5. 推理

```powershell
python tools/infer/predict_det.py --image_dir imgs --weights <pth> -c <cfg>
python tools/infer/predict_rec.py --image_dir imgs --weights <pth> -c <cfg>
python tools/infer/predict_yolo.py --image_dir imgs --weights <pth> -c <cfg>   # 七任务,自动按 task 可视化
```

## 6. 权重转换(Paddle / ultralytics → 本平台)

```powershell
# OCR:① paddlex 环境导出 numpy pickle ② ptocr 环境映射成 torch
& C:\ProgramData\miniconda3\envs\paddlex\python.exe tools/convert/dump_all.py
python tools/convert/convert_all.py

# 上游 YOLO .pt(ultralytics 环境) → 纯张量 state_dict
& C:\ProgramData\miniconda3\envs\ultralytics\python.exe tools/convert/dump_ultralytics.py yolo11n.pt --out _downloads/upstream

# 车牌(两条命令,含 unpickle 上游模型对象)
python tools/convert/convert_plate_weights.py --detect <plate_detect.pt> --rec <plate_rec_color.pth> --out _downloads/plate

# 属性(PaddleClas .pdparams)
& C:\ProgramData\miniconda3\envs\paddlex\python.exe tools/convert/dump_attribute_paddle.py --weights <x.pdparams> --out <x.pkl>
python tools/convert/convert_attribute_weights.py --pkl <x.pkl> --num-classes 26 --out <x_ptocr.pth>
```

## 7. 自检(改代码后必跑)

```powershell
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python tools/smoke_all.py          # 每个配置:建模型 + 取 1 个 batch + 1 步训练 + 1 步评估
#  期望:80 OK, 0 FAIL

python tools/check_graph_build.py  # 53 个 YAML 图配置建图+前向
#  期望:53 configs: 53 OK, 0 FAIL

python tools/check_plate_models.py # 车牌:上游权重张量对齐 + ONNX 数值对齐
#  期望:检测 500/500、识别 86/86,识别 ONNX max|diff| ~1e-5
```

## 8. 多卡与 Windows 注意事项

```powershell
python tools/train.py -c <cfg>.yml -o Global.distributed=true -o Global.gpu=0,1,2,3
```
* Windows 上 `num_workers` **不能开太多**(每 worker 约 3.5~4GB 提交内存;≥10 会耗尽并报
  `WinError 1455 页面文件太小`/`OpenBLAS error`)。默认 `Train=4 / Eval=0`,本机小内存请设 0~2。
* 首次运行建议 `num_workers=0` 定位问题;确认无误再开多进程。
* 详细坑位见 `docs/FAQ.md`。


---

## 9. 权重与数据的外置策略(仓库只放代码 / 配置 / label 文本)

仓库**不入库**图片、掩码、深度图与任何权重文件(.gitignore 已挡住)。两类外部资产的获取方式:

### 9.1 预训练权重
* 本地缓存目录:**~/.torchkiln/pretrained/**(Windows:C:\Users\<用户>\.torchkiln\pretrained)。
  平台按**配置名**找权重:<config_name>_ptocr.pth(	ools/smoke_all.py 的 --pretrained-dir 默认即此目录)。
* 配置里可以直接写 ModelScope URL(推荐),平台会**自动下载 + 缓存**:
  `yaml
  Global:
    pretrained_model: https://www.modelscope.cn/models/ChaoII0987/PytorchOCR/resolve/master/pretrained/<file>.pth
  `
  也可临时覆盖:tkiln train -c <cfg> -o Global.pretrained_model=<本地路径或URL>。
* 上传命名约定(必须与配置名一致):
  | 文件 | 模型 |
  |---|---|
  | <config_name>_ptocr.pth | OCR 17 个(PP-OCRv3/4/5/6 det/rec) |
  | PP-LCNet_x1_0_pedestrian_attribute_ptocr.pth / ..._vehicle_attribute_ptocr.pth | 行人 / 车辆属性 |
  | plate_detect_state.pth / plate_rec_color_state.pth | 车牌检测 / 识别 |
  | <name>_state.pth | 上游 YOLO(可选) |

### 9.2 数据集
`powershell
tkiln data list                  # 每个数据集:label / 图片 / URL 是否就绪
tkiln data get <name>            # 从 ModelScope 下载并解包到 datasets/<name>/
tkiln data get --all
python tools/make_demo_data.py --all     # 没网时生成本地占位图(仅供自检)
`
约定:datasets/<name>/{train.txt,val.txt,images/},	rain.txt 只列图片相对路径;
YOLO 系标注放 labels/<split>/*.txt,seg 另有 masks/(sem)、depth/(depth)。
清单与上传说明:datasets/manifest.yml、datasets/README.md。

### 9.3 上游权重转换(拿不到官方转换件时)
`powershell
# 车牌(GitHub 原始 .pt/.pth → 平台格式)
python tools/convert/convert_plate_weights.py --detect <plate_detect.pt> --rec <plate_rec_color.pth> --out <dir>
# 属性(PaddleClas .pdparams → 平台格式;需 paddlex 环境 dump 一次)
& C:\ProgramData\miniconda3\envs\paddlex\python.exe tools/convert/dump_attribute_paddle.py --weights <x.pdparams> --out <x.pkl>
python tools/convert/convert_attribute_weights.py --pkl <x.pkl> --num-classes 26 --out <x_ptocr.pth>
# 上游 YOLO(ultralytics 环境)
& C:\ProgramData\miniconda3\envs\ultralytics\python.exe tools/convert/dump_ultralytics.py <name>.pt --out <dir>
`
