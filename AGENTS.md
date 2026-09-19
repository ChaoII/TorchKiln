# AGENTS.md

## 回复语言
- 所有大模型（AI 助手/Agent）在本仓库中的回复一律使用**中文**。
- 包括：解释代码、回答提问、汇总结果、生成文档、提交说明等所有面向用户的文本。

## 显存 / batchsize 约束（重要）
- **batchsize 不能开太大，最多只允许占用显存的 1/4。**
- 本机 GPU 为 NVIDIA GeForce RTX 4060 Ti，**16 GB（约 16380 MiB）**。
- 在 **imgsz = 1024**（DOTA 旋转框 OBB 数据集常见输入）下，`batch_size = 8` 也会 **CUDA out of memory**；
  实测 **`batch_size = 4` 稳定可跑**（约占 4~6 GB，即 1/4~1/3 显存）。
- 若显存不足（CUDA out of memory），优先调小 `Train.loader.batch_size_per_card` 与 `Eval.loader.batch_size_per_card`，
  例如 `batch_size_per_card=4` 或 `2`，而**不是**升级模型 / 减小 imgsz。
- 训练/评估时建议设置：
  ```powershell
  $env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
  ```
- 评估用 `Eval.loader.num_workers=0`、训练默认 `Train.loader.num_workers=4`（Windows 下 worker 过多会耗尽提交内存）。

## 环境
- 框架（PytorchOCR / pytorchx）使用 conda 环境 **`ptocr`**（Python 3.12 + PyTorch 2.12 + CUDA）。
- 原版 ultralytics 使用 conda 环境 **`ultralytics`**（Python 3.12 + ultralytics 8.4.x）。
- 两者均可用 GPU（CUDA 可用）。

## 数据集
- **DOTA128（OBB 旋转检测）**位于 `datasets/dota128/`（来自 `G:\迅雷下载\dota128.zip`）。
  - 拆分为：`images/train`(102 张) + `images/val`(26 张)，标签为 DOTA 四角点(归一化 8 个数)格式。
  - 清单文件：`datasets/dota128/train.txt`、`datasets/dota128/val.txt`（图片相对路径，供框架 DetDataset 使用）。
  - ultralytics 侧数据配置：`datasets/dota128/dota128.yaml`。
  - 类别：15 类（plan/ship/storage-tank/... 见 dota128.yaml）。
- 图片与权重一律**不入库**（由 `.gitignore` 忽略），只保留标签文本与清单文件。

## OBB（旋转框）实现与 ultralytics 的差异（分析结论）
- **旋转框 NMS 是最大性能瓶颈**：框架自研 `pytorchx/det/rbox.py::nms_rotated` 是 **O(N²) 纯 NumPy
  逐对多边形裁剪**（Sutherland-Hodgman `poly_iou_np`）。实测 2000 个框 `nms_rotated` 需 **87 秒**；
  而 DOTA 图很密集、评估 `conf_thres=0.001` 会产生成千上万框 × 每类、每张 val 图，导致**评估阶段几乎跑不完**，
  表现为训练在 first eval 卡住、train.log / checkpoint 迟迟不落盘。
  对比：ultralytics 用编译版 rotated NMS（快得多）。
- **OBB 损失不含单独的 angle loss**：`pytorchx/det/loss.py::ObbLoss.forward` 只有
  `box_gain*(1-probiou) + cls_gain*BCE`（`angle_gain` 定义了但未参与 loss）。
  ultralytics `v8OBBLoss` 有 4 项：box + cls + dfl + **angle loss**(×`hyp.angle=1.0`)。角度仅靠 probiou 隐式优化。
- **角度解码**：框架 `rbox.py::dist2rbox` 用 `theta = atan(angle_raw)`；ultralytics 用 `angle` 分支解码。
- **指标 IoU**：框架 `det/metric.py` 用多边形裁剪 IoU(`poly_iou_np`)；ultralytics OBB 指标多用 `probiou`。
- **回归布局**：框架配置 `reg_layout: upstream`（`[reg(4), cls(nc), angle(ne)]`）与 ultralytics 兼容；
  `obb_ours` 是自研手写布局（`[reg(4+ne), cls]`）。现配置都走 `upstream`。

## 待办 / 已知问题
- 框架 OBB 评估几乎不可行（NMS 太慢）。要真正对齐 ultralytics 精度，**先把 `nms_rotated` 换成
  快速实现**（如 `torchvision.ops.nms` 的旋转版 / 批量化 poly IoU），再对比 dota128 mAP。
- dota128 训练从零、imgsz=1024、batch=4 已验证能跑 1 步；现配置为 30 epoch（可再调）。

## git 约定
- 仓库已在 `E:\PytorchOCR` 初始化。
- `.gitignore` 会忽略：`__pycache__`、`output/`、`*.log`、权重(`*.pt/*.pth`)、数据集图片、
  数据集压缩包(`*.zip/*.tgz/*.tar`)、缓存(`*.cache`、`.labels_cache_*.pkl`)、`_downloads/`、`_ref/`。
- 提交信息使用中文、简洁说明改动即可。
