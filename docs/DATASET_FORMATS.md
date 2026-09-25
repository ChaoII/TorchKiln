# 各任务数据集:标注格式示例 + 组织形式

> 命令/参数全表见 [`USER_GUIDE.md`](USER_GUIDE.md);本文件专注**标签格式 + 从零接入**。

> 原则:**每个模型都用它原生的数据格式**(YOLO 系 = ultralytics 原生;OCR 系 = PaddleOCR 原生),
> 不做强行统一;跨格式用 `tools/convert/dataset_format.py` 转一次即可。
> 所有路径都写在配置里:`Train/Eval.dataset.data_dir` + `label_file_list`(**没有** `train_list`/`val_list` 字段);
> 清单里图片路径**相对 `data_dir`**(或写绝对路径,见下)。

## 0. 从零接入一个自定义数据集(完整步骤)

以 **YOLO 单类检测** 为例(其它任务只改标签行与少量 `-o` 字段):

```powershell
# 1) 按 §3 摆好目录(标签格式看对应任务小节;模板可拷 datasets/_format_examples/det/)
#    D:/mydata/det/
#      images/train/*.jpg  images/val/*.jpg
#      labels/train/*.txt  labels/val/*.txt
#      train.txt  val.txt                       # 每行一个相对路径,如 images/train/0001.jpg

# 2) 用 -o 指向数据(不改 yml;绝对路径 label 也合法)
.\tkiln.bat check -c configs/yolo/yolo11-det.yml `
  -o Train.dataset.data_dir=D:/mydata/det `
  -o Train.dataset.label_file_list=train.txt `
  -o Architecture.Head.num_classes=1 `
  -o Train.dataset.names=[plate]

# 3) 训练 / 评估 / 推理 / 导出(完整参数见 docs/TRAINING.md)
.\tkiln.bat train -c configs/yolo/yolo11-det.yml `
  -o Train.dataset.data_dir=D:/mydata/det `
  -o Train.dataset.label_file_list=train.txt `
  -o Eval.dataset.data_dir=D:/mydata/det `
  -o Eval.dataset.label_file_list=val.txt `
  -o Architecture.Head.num_classes=1 `
  -o Train.dataset.names=[plate] `
  -o Global.save_model_dir=./output/my_y11 `
  -o Global.device=cuda:0 `
  -o Global.epoch_num=100

.\tkiln.bat val -c configs/yolo/yolo11-det.yml `
  --weights output/my_y11/best_accuracy.pth `
  -o Eval.dataset.data_dir=D:/mydata/det -o Eval.dataset.label_file_list=val.txt

.\tkiln.bat predict -c configs/yolo/yolo11-det.yml `
  --weights output/my_y11/best_accuracy.pth --input D:/img.jpg --output out.jpg

.\tkiln.bat export -c configs/yolo/yolo11-det.yml `
  --weights output/my_y11/best_accuracy.pth --save-dir output/exp/y11 --onnx --slim
```

路径语义(`torchkiln/data/det.py` 等):

| `label_file_list` 项 | 解析 |
|---|---|
| 不是绝对路径、且当前不是已存在文件 | 拼到 `data_dir` 下 |
| 已是存在的文件(含绝对路径) | 直接打开 |

因此 `-o Train.dataset.label_file_list=train.txt` 与
`-o ...=D:/mydata/det/train.txt` 等价(后者依赖文件已存在)。

各任务**必须额外对齐**的字段(在 3 的基础上加):

| 任务 | 额外 `-o` / 配置字段 |
|---|---|
| OBB | `Train/Eval.dataset.box_format=xywhr` + **9 字段角点标签**(§4) |
| pose | `kpt_shape` 写三处:`Architecture.Head` + `Loss` + `Train/Eval.dataset` |
| seg | `transform.mask_stride`(默认 4);多边形或 `masks/` |
| cls | `names` + `Architecture.Head.num_classes` |
| plate_det | `dataset.kpt_shape=[4,2]` + `Head.kpt_label=4` + `Head.num_classes=2` |
| plate_rec | `transform.image_size=[48,168]` + `Head.color_num=5` |
| attribute | `dataset.label_ratio=true` + `Head.label_list` |
| OCR det/rec | `dataset.name=SimpleDataSet` + **`transforms:` 列表**(不是 YOLO 的 `transform:`) |
| OCR rec | 另加 `Global.character_dict_path`、`max_text_length` |

数据集就绪与下载:

```powershell
tkiln data list                 # datasets/manifest.yml 就绪性
tkiln data get <name>           # ModelScope zip → datasets/<name>/
python tools/make_demo_data.py --all          # 本地占位图(仅 smoke 自检)
python tools/make_format_examples.py          # 各任务标签+配置片段模板
```

## 通用组织约定

| 类型 | 组织形式 |
|---|---|
| **YOLO 系**(det/obb/seg/pose/cls/sem/depth) | `images/<split>/x.jpg` + `labels/<split>/x.txt`(或 `masks/`、`depth/`),`train.txt`/`val.txt` 只列图片路径 |
| **单行标签**(OCR rec / 车牌 rec / 属性 / 骨架 / 视频) | `train.txt`/`val.txt` 每行 `路径<TAB或空格>标签…`,图片(或序列)放 `images/` 或任意子目录 |
| **OCR det** | `train.txt`/`val.txt` 每行 `路径<TAB>[{"transcription":…,"points":[[x,y]×4]}]`(内联 JSON) |

`<split> ∈ {train, val}`;`train.txt`/`val.txt` 内容是**相对 `data_dir` 的路径**。

---

## 1. OCR 文本检测(`task: det`,family `ocr`)

**组织形式**
```
datasets/ocr_det/
├─ images/               # 图片(可平铺,也可 images/train, images/val)
│  └─ 0001.jpg
├─ train.txt
└─ val.txt
```
**标注示例**(`train.txt`,PaddleOCR 内联 JSON,一点一行)
```
images/0001.jpg	[{"transcription": "07", "points": [[1552, 382], [1647, 382], [1647, 470], [1552, 470]], "difficult": false}]
images/0002.jpg	[{"transcription": "ABC123", "points": [[935, 379], [1052, 379], [1052, 481], [935, 481]], "difficult": false}]
```
* `points`:四点**像素坐标**(左上→右上→右下→左下);`transcription`:文本(可空 `""`);
* 配置:`label_file_list: [datasets/ocr_det/train.txt]`,无需 `labels/`。

## 2. OCR 文本识别(`task: rec`)

**组织形式**:`images/` + `train.txt`/`val.txt`(单行标签)+ 字符集 `dict.txt`(一行一个字符,首行可为 blank)
**标注示例**
```
images/word_0001.jpg	纯文本内容
images/word_0002.jpg	ABC123
```
配置需 `Global.character_dict_path: <dict.txt 路径>`。

## 3. YOLO 目标检测(`task: detect`)—— 原生 ultralytics

**组织形式**
```
datasets/dx_det/
├─ images/train/*.jpg      images/val/*.jpg
├─ labels/train/*.txt      labels/val/*.txt       # 与图片同名
├─ data.yaml               # path / train / val / nc / names
├─ train.txt  val.txt      # 只列图片路径(平台用)
```
**标注示例**(`labels/train/frame_00001.txt`)
```
0 0.463161 0.391599 0.054624 0.088076
1 0.762000 0.501000 0.120000 0.090000
```
* 每行 `cls cx cy w h`,**全部归一化到 [0,1]**(`cx,cy` 为中心点,`w,h` 为宽高);
* `data.yaml`:
```yaml
path: E:/dx_ocr/ultralytics
train: images/train
val: images/val
nc: 1
names: {0: plate}
```

## 4. 旋转框 / OBB(`task: obb`)—— 加载器要求 **9 字段角点**

**组织形式**同 detect(`images/<split>` + `labels/<split>` + `train.txt`/`val.txt`)。

**标注示例**(`labels/train/xxx.txt`,每行 **cls + 4 个归一化角点 = 9 个数**):
```
0 0.310000 0.360000 0.650000 0.360000 0.710000 0.440000 0.370000 0.440000
```
* 顺序:`cls x1 y1 x2 y2 x3 y3 x4 y4`,**全部归一化 [0,1]**;角点可顺/逆时针(内部 `cv2.minAreaRect` 规整);
* 加载真值:`torchkiln/data/det.py::_load_raw_labels` —— `box_format=xywhr` 时 **`len(parts) < 9` 整行丢弃**,
  再 `poly2rbox` 转成内部 `(N,6) xywhr`;**不是** 6 字段 `cls cx cy w h angle`。
* 配置:`Train/Eval.dataset.box_format: xywhr`(增广/评估按旋转框走);
* 与 DOTA/ultralytics 四角点导出一致;若只有 `xywhr`,请先用 `poly2rbox`/`xywhr2xyxyxyxy` 转成四角点。

> ⚠️ 旧文档与 `tools/make_format_examples.py` 曾写成 `cls cx cy w h angle`——以**加载器 9 字段角点**为准;
> `_format_examples/det/README.txt` 已同步更正。

## 5. 实例分割(`task: segment`)

**组织形式**:`images/<split>/` + `labels/<split>/`(多边形)或 `masks/`(位图)
**标注示例**(多边形,归一化,点对)
```
0 0.10 0.20 0.40 0.20 0.40 0.55 0.10 0.55
1 0.60 0.30 0.75 0.32 0.72 0.50 0.58 0.48
```
* 每行 `cls x1 y1 x2 y2 …`(至少 3 点,自动闭合);平台按外接框 + 光栅化掩码训练;
* 若用位图掩码:每实例一个 PNG,像素 255=前景,放在 `masks/<split>/`。

## 6. 关键点 / 姿态(`task: pose`)

**组织形式**:同 detect;**标注示例**(`kpt_shape: [17, 3]`)
```
0 0.500000 0.500000 0.200000 0.600000 0.480 0.120 2 0.520 0.120 2 0.460 0.300 2 ...
```
* 每行 `cls cx cy w h` + `K × (px py v)`;`px,py` 归一化,**`v` 为可见性**(0=不存在/未标,1=遮挡,2=可见);
* 车牌四角点(无可见性)用 `kpt_shape: [4, 2]` → `cls cx cy w h p1x p1y p2x p2y p3x p3y p4x p4y`。

## 7. 图像分类(`task: classify`)

**组织形式**:`images/` + `train.txt`/`val.txt`

### 7.1 单标签(默认)

**标注示例**
```
images/cat_001.jpg 0
images/dog_002.jpg 1
```
* `路径 类别号`(类别号从 0 开始)。

### 7.2 多标签(一键开关)

**标注示例**
```
images/a.jpg 1 0 1 0 0
images/b.jpg 0 1 1 0 1
```
* 第 2 列起为 **0/1 multi-hot**(或 0~1 软标签),长度 = `Head.num_classes`;
* 打开方式**只需一条开关**(自动切 BCE + sigmoid 阈值后处理 + mAP 指标):

```powershell
# 命令行（推荐；完整示例见 USER_GUIDE.md §3.7）
-o Loss.multi_label=true

# 或复制 configs/yolo/yolo11-cls.yml → configs/local/my.yml 后写死:
# Loss.multi_label: true
# Train/Eval.dataset.multi_label: true   # 可省;写在 Loss 上即可
```

* 可选:`PostProcess.threshold`(默认 0.5)、`Metric.main_indicator`(默认 mAP,可改 `mA`);
* 类平衡:数据集写 `label_ratio: true`,损失开 `Loss.weight_ratio: true`。

## 8. 语义分割(`task: semantic`)

**组织形式**:`images/<split>/x.jpg` + `masks/<split>/x.png`(同名)+ `train.txt`/`val.txt`
**标注示例**:`masks/train/0001.png` 是 **uint8 单通道**图,**像素值 = 类别索引**(0..C-1);
忽略像素用 `Train.dataset.ignore_index`(默认 255)表示。
```yaml
Train.dataset.transform.image_size: 256
```

## 9. 深度估计(`task: depth`)

**组织形式**:`images/<split>/x.jpg` + `depth/<split>/x.png`(同名)+ `train.txt`/`val.txt`
**标注示例**:`depth/train/0001.png` 为 **uint16 单通道**,数值 = 深度;
单位由 `Train.dataset.depth_scale` 决定(默认 `1000` = 毫米,即 1000 → 1 米)。

## 10. 车牌检测(`task: plate_det`,2 类 + 4 角点)

**组织形式**:同 detect(原生 YOLO 布局),配置里 `kpt_shape: [4, 2]`
**标注示例**
```
0 0.482960 0.373269 0.444334 0.120977 0.260793 0.312780 0.705127 0.312780 0.705127 0.433757 0.260793 0.433757
```
* `cls`:`0`=单层牌、`1`=双层牌;后 8 个数是**四角点归一化坐标**(左上,右上,右下,左下)。

## 11. 车牌识别 + 颜色(`task: plate_rec`)

**组织形式**:`images/`(48×168 车牌切图)+ `train.txt`/`val.txt`
**标注示例**
```
images/plate_0001.jpg 5 53 52 60 49 45 43 1
```
* 前 7 个是**字符表下标**(`torchkiln/nn/plate.py::PLATE_CHARSET`,78 项,`0` = CTC blank),
  最后一个是**颜色号**(`['黑色','蓝色','绿色','白色','黄色']` 的下标 0..4);
* 双层车牌可设 `Train.dataset.double_plate: true`(自动上下拼接成单行)。

## 12. 多标签属性识别(`task: attribute`,行人 26 / 车辆 19)

**组织形式**:`images/` + `train.txt`/`val.txt`
**标注示例**(行人:26 个数)
```
images/p_0001.jpg 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
```
* 每个属性 1 位:`1`=有该属性、`0`=无;**允许软标签**(0~1 的小数);
* `Train.dataset.label_ratio: true` 时平台会统计**全局正例比例**并按类别加权(对齐 PaddleClas `MultiLabelLoss`)。

## 13. 骨架行为识别(`task: pose_action`)

**组织形式**:关键点序列 `.npy` + `train.txt`/`val.txt`
```
datasets/pose_action/
├─ seqs/fall_0001.npy      # (T, V, C) 或 (C, T, V);C 默认 2(x,y)
├─ train.txt  val.txt
```
**标注示例**
```
seqs/fall_0001.npy 0
seqs/fight_0002.npy 1
```
* `V` 默认 17(COCO/YOLO-pose),可用 `Backbone.num_joints` 改;
* `transform.clip_len` 控制统一时长(不足重复/超出随机裁剪),`Head.num_classes` 为行为类别数。

## 14. 视频行为识别(`task: video_cls`)

**组织形式**:`train.txt`/`val.txt` 里写**视频文件**或**帧目录**
```
datasets/video_cls/
├─ frames/fight_01/0001.jpg … 0100.jpg      # 或直接 videos/fight_01.mp4
├─ train.txt  val.txt
```
**标注示例**
```
frames/fight_01 0
videos/walk_02.mp4 1
```
* 采样:`transform.num_segments × frames_per_seg` 帧(均匀分段),`image_size`/`crop_size` 控制分辨率;
* `Backbone.scale` 调容量,`Head.num_classes` 为行为类别数。

## 15. 车道线-分割式(`task: lane_seg`)

**组织形式**:同 semantic —— `images/<split>/x.jpg` + `masks/<split>/x.png`(同名)+ `train.txt`/`val.txt`
**标注示例**:`masks/train/0001.png` uint8,像素值 = 类别索引(0=背景,1..C=车道类);忽略像素 = `ignore_index`(默认 255)。
```yaml
Architecture.task: lane_seg
Architecture.Head.num_classes: 2        # 0=bg + 1=lane
Loss.name: LaneSegLoss                  # CE + dice + focal
Metric.name: LaneSegMetric
Metric.main_indicator: lane_IoU
```
* 可与 `semantic` 共用数据/头,仅 metric 换 `lane_IoU`(前景∪所有车道类 IoU)。

## 16. 车道线-行式/UFLD(`task: lane_row`)

**组织形式**:`images/<split>/*.jpg` + `labels/<split>/*.txt`(同名)+ `train.txt`/`val.txt`
**标注示例**(`labels/train/0001.txt`,每行一条车道,固定 `num_lanes` 行):
```
1 0.1500 0.1525 0.1550 ... 0.3500
1 0.5500 0.5525 0.5550 ... 0.7500
0 0 0 ... 0
```
* 每行:`valid x0 x1 ... x_{R-1}`;`valid∈{0,1}`;`x∈[0,1]` 归一化横坐标;`R=num_rows`(默认 100);
* 配置:`Architecture.task: lane_row`、`Head.num_lanes/num_rows/num_bins`;
* Loss `LaneRowLoss`(行分类 CE/BCE,ignore 无效行);Metric `LaneRowMetric`(main_indicator `F1`,threshold_px=50)。

## 17. 点云分割(`task: pc_seg`)

**组织形式**
```
datasets/pc_demo/
├─ clouds/train/*.npy|.bin     # float32 N×3(x,y,z) 或 N×4(+intensity)
├─ labels/train/*.npy|.txt     # 每点 int 类别(与点一一对应);缺省视为全 0
├─ train.txt  val.txt          # 每行一个点云相对路径,如 clouds/train/0001.npy
```
**标注示例**(`labels/train/0001.txt` 或 `.npy`,一行/一元素一个点标签):
```
0
0
1
1
...
```
* `.bin` 为 KITTI 风格 float32 reshape;`.npy` 推荐 demo;
* 配置:`Architecture.task: pc_seg`、`Head.num_classes=C`;
* `Train.dataset`:`pc_range=[xmin,xmax,ymin,ymax,zmin,zmax]`、`pillar_size=[dx,dy,dz]`、`max_points_per_pillar`;
* Loss `SemLoss`(pillar CE);Metric `SemMetric`(main_indicator `mIoU`);
* **图像预训练权重不适用**,`Global.pretrained_model: null`。

## 18. 3D 检测(`task: det3d`,CenterPoint 简化)

**组织形式**
```
datasets/det3d_demo/
├─ clouds/train/*.npy|.bin     # 同 pc_seg
├─ labels/train/*.txt          # 每行一个 3D 框
├─ train.txt  val.txt
```
**标注示例**(`labels/train/0001.txt`):
```
car 12.1 -3.4 0.8 3.7 1.6 1.5 1.57
ped  5.0  2.2 0.9 0.6 0.6 1.7 0.10
```
* 每行:`cls x y z l w h yaw`(LiDAR 系;`cls` = 类名或 int;可选第 9 列 difficulty 忽略);
* 配置:`Architecture.task: det3d`、`Head.num_classes=C`、`Train.dataset.names: [...]`;
* `pc_range=[xmin,ymin,zmin,xmax,ymax,zmax]`、`pillar_size=[dx,dy,dz]`(与 pc_seg 相同语义);
* Loss `Det3DLoss`(heatmap focal + L1);Metric `Det3DMetric`(BEV rotated IoU AP,thr [0.5,0.7]);
* PostProcess `Det3DPostProcess`(score_thres + nms_thres + BEV NMS);
* **图像预训练权重不适用**,`Global.pretrained_model: null`。

---

## 19. 标签模板与格式转换

### 19.1 `_format_examples`(可整目录拷走)

```powershell
python tools/make_format_examples.py              # 全部任务
python tools/make_format_examples.py --task det --task pose   # 指定任务
```

生成 `datasets/_format_examples/<task>/{images, labels?, train.txt, val.txt, README.txt}`;
`README.txt` 含字段说明 + 对应配置片段。已含任务:
`det/segment/pose/classify/semantic/depth/plate_rec/attribute/pose_action/video_cls/ocr_det/ocr_rec/lane_seg/lane_row/lane_bev/pc_seg/det3d`。

### 19.2 格式转换(`tools/convert/dataset_format.py`,纯 CPU)

```powershell
# 原生 YOLO  ->  PaddleOCR 检测(内联 JSON)
python tools/convert/dataset_format.py --from yolo_det --to ocr_det --src <src> --dst <dst>

# PaddleOCR 检测  ->  原生 YOLO(生成 images/ labels/ data.yaml)
python tools/convert/dataset_format.py --from ocr_det --to yolo_det --src <src> --dst <dst> --nc 1 --names plate

# 规整成原生 YOLO 布局(硬链接,不占空间;--copy 则复制)
python tools/convert/dataset_format.py --from yolo_det --to yolo_det --src <src> --dst <dst>
```
已往返自检:`yolo → ocr_det → yolo`,1014 框,**不一致 0**。
**当前 READERS/WRITERS 仅实现 `yolo_det` 与 `ocr_det`**;`ocr_rec` / `cls` / `seg` 等方向需在脚本里加一对 reader/writer(同一框架)。
