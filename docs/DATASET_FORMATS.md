# 各任务数据集:标注格式示例 + 组织形式

> 原则:**每个模型都用它原生的数据格式**(YOLO 系 = ultralytics 原生;OCR 系 = PaddleOCR 原生),
> 不做强行统一;跨格式用 `tools/convert/dataset_format.py` 转一次即可。
> 所有路径都写在配置里:`Train/Eval.dataset.data_dir` + `label_file_list`;
> 图片路径**相对 `data_dir`**。

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

## 4. 旋转框 / OBB(`task: obb`)

**组织形式**同 detect;**标注示例**
```
0 0.512000 0.402000 0.210000 0.070000 0.7854
```
* `cls cx cy w h angle`,角度**弧度**,范围 `[-π/2, π/2)`(le90);
* 配置:`Train.dataset.box_format: xywhr`(平台据此按旋转框解析/增广/评估)。

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
**标注示例**
```
images/cat_001.jpg 0
images/dog_002.jpg 1
```
* `路径 类别号`(类别号从 0 开始)。

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

---

## 格式转换(`tools/convert/dataset_format.py`,纯 CPU)

```powershell
# 原生 YOLO  ->  PaddleOCR 检测(内联 JSON)
python tools/convert/dataset_format.py --from yolo_det --to ocr_det --src <src> --dst <dst>

# PaddleOCR 检测  ->  原生 YOLO(生成 images/ labels/ data.yaml)
python tools/convert/dataset_format.py --from ocr_det --to yolo_det --src <src> --dst <dst> --nc 1 --names plate

# 规整成原生 YOLO 布局(硬链接,不占空间;--copy 则复制)
python tools/convert/dataset_format.py --from yolo_det --to yolo_det --src <src> --dst <dst>
```
已往返自检:`yolo → ocr_det → yolo`,1014 框,**不一致 0**。
需要 `ocr_rec`(路径↔文本)、`cls`、`seg`(多边形↔掩码)方向时,在脚本里加一对 reader/writer 即可(同一框架)。
