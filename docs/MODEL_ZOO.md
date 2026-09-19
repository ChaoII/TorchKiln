# 模型总表

平台共 **4 条产品线 / 88 个可训练配置**,全部通过 `tools/smoke_all.py` 验证(80 OK,0 FAIL)。

---

## 1. OCR 文本检测(det,9 个配置)

`configs/det/*.yml` · 任务 `Architecture.task: det` · 指标 `DetMetric`(hmean / precision / recall)

| 配置 | 主干 / 结构 | 预训练权重(缓存于 `~/.pytorchocr/pretrained/`) |
|---|---|---|
| `ch_PP-OCRv3_det_student` | MobileNetV3 + RSEFPN + DBHead | `ch_PP-OCRv3_det_student_ptocr.pth` |
| `ch_PP-OCRv4_det_mobile` | MobileNetV3 + RSEFPN + PFHeadLocal | `ch_PP-OCRv4_det_mobile_ptocr.pth` |
| `ch_PP-OCRv4_det_server` | ResNet_vd + LKPAN + PFHeadLocal | `ch_PP-OCRv4_det_server_ptocr.pth` |
| `ch_PP-OCRv4_det_cml` | ResNet_vd + LKPAN(多教师) | `ch_PP-OCRv4_det_cml_ptocr.pth` |
| `PP-OCRv5_mobile_det` | PPLCNetV3 + RepLKFPN + PFHeadLocal | `PP-OCRv5_mobile_det_ptocr.pth` |
| `PP-OCRv5_server_det` | PPHGNetV2_B4 + RepLKPAN + PFHeadLocal | `PP-OCRv5_server_det_ptocr.pth` |
| `PP-OCRv6_{tiny,small,medium}_det` | PPHGNetV2 + (R)LKPAN + PFHeadLocal(+v6 结构) | `PP-OCRv6_*_det_ptocr.pth` |

其他手写/参考配置:`det_mv3_db`、`det_r50_vd_db`、`det_mv3_east`、`det_mv3_pse`、`det_r50_vd_east`、
`det_r50_vd_sast_*`、`det_r50_vd_fce_ctw`(`configs/debug/` 与 `_downloads/paddle_cfg/` 内有参考)。
损失:DB 系列 Shrink(Dice/BCE+OHEM)+ Threshold(L1)+ Binary(Dice);后处理 `DBPostProcess`(Paddle 原版)。

## 2. OCR 文本识别(rec,8 个配置)

`configs/rec/*.yml` · 任务 `task: rec` · 指标 `RecMetric`(acc / norm_edit_dis)

| 配置 | 结构 |
|---|---|
| `ch_PP-OCRv3_rec` | MobileNetV3 + RNN + CTC/SAR(蒸馏) |
| `ch_PP-OCRv4_rec` | PPLCNetV3 + SVTR-LCNet + CTC |
| `ch_PP-OCRv4_rec_hgnet` | PPHGNet + SVTR-LCNet + CTC |
| `PP-OCRv5_mobile_rec` / `PP-OCRv5_server_rec` | SVTR-LCNet(PPHGNetV2 主干) |
| `PP-OCRv6_{tiny,small,medium}_rec` | PPHGNetV2 + SVTR + v6 结构(字符集不同) |
| `rec_svtrnet` / `rec_mtb_nrtr` / `rec_r31_sar` / `rec_d28_can`(...) | 参考实现(SVTR/NRTR/SAR/CAN) |

字符集放在 `pytorchx/ocr/utils/dict/`(v5: `ppocrv5_dict.txt`、v6: `ppocrv6_dict.txt`、v6-tiny: `ppocrv6_tiny_dict.txt`)。

## 3. YOLO 家族(7 任务,59 个配置)

* 图模型(YAML 驱动,NMS-free/多尺度/各家族的模块都支持):`configs/yolo/*_graph.yml` **53 个**
* 手写模型:`YOLO11n_{det,seg,obb,pose,cls,sem,depth}.yml` + `yolov8n_det.yml` 等 **6 个**

| 任务 | `Architecture.task` | 头 / 损失 / 指标 | 数据格式 |
|---|---|---|---|
| 检测 | `detect` | `Detect(26/DFL)` + TAL + CIoU/DFL + BCE;COCO mAP50-95 | `cls cx cy w h`(归一化)|
| 旋转框 | `obb` | `OBB` + ProbIoU;mAP | `cls cx cy w h angle`(弧度,le90)|
| 实例分割 | `segment` | `Segment` + mask 原型/系数;box_mAP + mask_mAP | 检测标签 + 每实例多边形/掩码 |
| 关键点 | `pose` | `Pose`(kpt_shape,OKS);OKS-mAP | `cls cx cy w h px py v ...` |
| 分类 | `classify` | `Classify` + CE;top1/top5 | `路径 类别号` |
| 语义分割 | `semantic` | `SemanticSegment`;mIoU / acc | 掩码 PNG(类别索引)|
| 深度 | `depth` | `Depth`;δ1/δ2/δ3、AbsRel、RMSE | 深度图(16bit PNG/npy)|

上游 YAML 家族覆盖:`v3`(3)、`v5`(2)、`v6`(1)、`v8`(12)、`v9`(8)、`v10`(7)、`11`(5)、`12`(5)、`26`(9)、`plate`(1)。
`tools/check_graph_build.py` 逐个建图+前向:**53 configs, 53 OK, 0 FAIL**。

### 训练特性(与 ultralytics 对齐的部分)
AMP(fp16/bf16)、`freeze`、梯度累积、Adam/AdamW/SGD/RAdam/NAdam、Cosine/Linear/OneCycle/Const + warmup、
`label_smoothing`、增广(`mosaic4/mosaic9/mixup/copy_paste/close_mosaic/multi_scale/hsv/affine/erasing`)、
EMA、断点续训、多卡(DP/DDP)、DFL 与 NMS-free 端到端头、ONNX 导出(端到端图内无 NMS)。

## 4. 车牌(上游原版复刻,2 个配置)

来源:`we0091234/Chinese_license_plate_detection_recognition`(检测)+ 子仓库 `crnn_plate_recognition`(识别)。

| 配置 | 模型 | 结构 | 指标 |
|---|---|---|---|
| `configs/plate/plate_det.yml` | 检测(单层/双层 2 类 + 4 角点) | yolov5-lite(`StemBlock`/`ShuffleV2Block`)+ yolov5-face 头 `PlateDetect`(逐 anchor `[xy,wh,obj,kpt8,cls2]`,no=15) | 框 mAP + 角点 NME/kpt_px |
| `configs/plate/plate_rec.yml` | 识别 + 颜色(5 类) | `PlateRecNet`(`myNet_ocr_color`):CNN+CTC(78 类)+ 颜色头;输入 48×168,mean/std=0.588/0.193 | 整牌 acc / 字符 acc / 颜色 acc |

权重与对齐证据(见 `tools/check_plate_models.py`):
* 检测:官方 `plate_detect.pt` **500/500 张量**,与上游 `models/yolo.py::Detect` 逐元素 **max|diff| = 0.000e+00**;
* 识别:官方 `plate_rec_color.pth` **86/86 张量**,与出厂 ONNX 数值一致(6.1e-05 / 6.7e-06);
* 注意:用户 `test_data` 里的 `yolov5plate.onnx` 是**更早版权重**(首层卷积即不同),不能作为对齐基准。

## 5. 属性识别(PaddleX 复刻,2 个配置)

来源:`E:\PaddleX` 的 `multilabel_classification` 模块(PaddleClas 后端)。

| 配置 | 类别 | 输入 | 说明 |
|---|---|---|---|
| `configs/attr/pedestrian_attribute.yml` | **26**(帽子/眼镜/短袖/长袖/上衣条纹…)| 256×192(竖版)| PA100K 属性集 |
| `configs/attr/vehicle_attribute.yml` | **19**(9 颜色 + 10 车型)| 192×256(横版)| 车辆属性集 |

统一结构 `AttributeNet` = `PPLCNetX1_0`(`NET_CONFIG`:blocks5 无 SE、blocks6 仅 2 单元;**1,707,386 参数**)
+ 多标签头(`1×1 last_conv 512→1280 → hardswish → dropout → Linear`)。
损失 `MultiLabelLoss`(BCE + `ratio2weight = exp((1-t)r + t(1-r))`,`size_sum=True`),
指标 `AttrMetric`(mA@0.5 + mAP),增广对齐 `TimmAutoAugment(rand-m9-mstd0.5-inc1, p=0.8)` + RandomErasing。

权重对齐:官方 `PP-LCNet_x1_0_{pedestrian,vehicle}_attribute_pretrained.pdparams`
→ 键名映射后 **146/146 张量、0 冲突**,`load_state_dict` missing=0/unexpected=0;
同一输入下 top-5 属性与 Paddle 结果 **4/5 一致**。前向 logits 仍有 ~0.8 的数值偏差,
已定位到 `blocks3.0.pw_conv` 的裸 1×1 卷积(权重 bit 相同、输入几乎相同),待继续排查。

---

## 指标说明与对齐情况

| 指标 | 定义 | 对齐状态 |
|---|---|---|
| det hmean | 2·P·R/(P+R),IoU 0.5(Paddle 口径)| ✓ 与 Paddle 一致 |
| rec acc | 整句完全匹配率 | ✓ |
| COCO mAP50-95 | per-class AP 平均(IoU 0.5:0.05:0.95)| ✓ 自实现,已与训练曲线交叉验证 |
| mA(属性) | 逐属性阈值 0.5 准确率均值 | ✓ 语义与 PaddleClas `ATTRMetric` 一致 |
| 车牌框 mAP | 同 COCO mAP | ✓ |
| 角点 NME | 匹配框(IoU≥0.5)上四角平均像素误差 / 框对角线 | 平台新增指标(上游无)|
| v3 det/rec | 2.x 代模型 | ⚠ 官方 3.x 文档未列指标,本地无精确基线 |
| 上游 seg/obb/pose(图模型)| — | ⚠ 官方权重站点不可达,仅做结构级对齐 |
