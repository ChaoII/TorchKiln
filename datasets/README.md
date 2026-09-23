# 数据集清单(仓库内只保留 label 文本,图片打包在 ModelScope)

> 约定:每个数据集目录结构为
> ```
> datasets/<name>/
> ├─ train.txt / val.txt        # label 文本(入库)
> ├─ images/                    # 图片(不入库;从 ModelScope 下载或本地生成)
> └─ (掩码 / 深度图等按任务另放,同样不入库)
> ```
> `train.txt` / `val.txt` 里写**相对 `data_dir` 的图片路径**(如 `images/0001.jpg`),
> 图片缺失时训练会报错,用下面的命令补齐即可。

## 拉取/生成命令

```powershell
# ① 从 ModelScope 下载(在 manifest 里填好 url 后)
tkiln data list                                  # 看有哪些数据集、缺什么
tkiln data get plate_det_demo                    # 下载并解包到 datasets/plate_det_demo/

# ② 本地生成"占位图片"(仅用于自检 smoke_all,不含真实标注语义)
python tools/make_demo_data.py --all
python tools/make_demo_data.py --dataset plate_det_demo
```

## 清单

| 数据集 | 任务 | label 格式 | 图片数(参考) | ModelScope |
|---|---|---|---|---|
| `det_demo` | detect | `cls cx cy w h`(归一化)| 80/20 | *(待填)* |
| `obb_demo` | obb | `cls cx cy w h angle`(弧度,le90)| 80/20 | *(待填)* |
| `seg_demo` | segment | 检测行 + 多边形/掩码索引 | 80/20 | *(待填)* |
| `pose_demo` | pose | `cls cx cy w h px py v ...`(kpt_shape [4,3])| 80/20 | *(待填)* |
| `cls_demo` | classify | `路径 类别号` | 每类若干 | *(待填)* |
| `sem_demo` | semantic | 掩码 PNG(像素=类别索引)| 80/20 | *(待填)* |
| `depth_demo` | depth | 深度图(16bit PNG/npy)| 80/20 | *(待填)* |
| `plate_det_demo` | plate_det | `cls cx cy w h p1x p1y ... p4x p4y`(kpt_shape [4,2])| 24/8 | *(待填)* |
| `plate_rec_demo` | plate_rec | `路径 c1..c7 颜色号`(字符表见 `torchkiln/nn/plate.py`)| 32/8 | *(待填)* |
| `pedestrian_attribute_demo` | attribute | `路径 v1..v26`(多热)| 24/8 | *(待填)* |
| `vehicle_attribute_demo` | attribute | `路径 v1..v19`(多热)| 24/8 | *(待填)* |
| `ocr_det_dataset_examples` | OCR det | PaddleOCR 格式(四点 + 文本)| 250 | *(待填)* |
| `ocr_rec_dataset_examples` | OCR rec | `路径\t文本` | 6545 | *(待填)* |

## ModelScope 上传建议

* 每个数据集打成一个 zip:`<name>.zip`(内部保持 `images/` + `train.txt` + `val.txt`);
* 上传到同一个 ModelScope 模型仓库(例如 `ChaoII0987/PytorchOCR`,目录 `datasets/`),
  然后填到上表,并把下载 URL 写进 `datasets/manifest.yml`;
* 权重同理放 `pretrained/`(`*_ptocr.pth`),配置里用 URL 引用即可(平台支持 URL 自动下载+缓存)。
