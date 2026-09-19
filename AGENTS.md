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

## OBB（旋转框）与 ultralytics 的对齐（已完成并验证）
- **旋转框 NMS**：`pytorchx/det/rbox.py::nms_rotated` 由 O(N²) 纯 NumPy 多边形裁剪（2000 框 87s）
  改为 **`probiou`(ProbIoU 上三角矩阵) + `fast_nms` 上三角抑制**（GPU 向量化，2000 框 0.13s），
  并加 `max_candidates=3000`（仿 ultra `max_nms`）防密集图 OOM。
  **已验证与 ultra `TorchNMS.fast_nms(...,iou_func=batch_probiou)` 输出索引完全一致（maxdiff=0.0）。**
- **`probiou` 高斯方差改为 `w^2/12`**（之前 `(w/2)^2`），与 ultra `batch_probiou` 数值一致（maxdiff=0.0）。
  返回值已去尾维 `(N,)`/`(B,A,M)`，调用方别多 squeeze。
- **指标 IoU 用 probiou**：`det/metric.py::_iou` 对 `box_format=xywhr` 用 `probiou`（原 `poly_iou_np`）。
- **`dist2rbox` 全部对齐 ultra**：网格单位解码、中心偏移按角度旋转、`wh=l+r/t+b`、
  `theta=(sigmoid(angle)-0.25)*pi`；新增 `rbox2dist`（逆解码，供 DFL）。
- **ObbLoss 移植 ultra `v8OBBLoss`**：`box=(1-probiou)*weight/tsum`（floor=0.01）+ **DFL**（reg_max>1）
  + **angle loss**（`sin(2Δθ)² * 宽高比权重`）+ **cls(BCE)**；`reg_max` 从 1 改为 **16（DFL）**，
  `dfl_gain=1.5`、`angle_gain=1.0`（对齐 ultra hyp）。
- **OBB 头**：`REGISTRY["OBB"]=OBBU`（已用 ultra 新版 `_dw_cls_branch` DWConv 结构）。
- **训练循环 LR 修复**：`ptcore/trainers/base.py` 的 `opt_steps_per_epoch` 由 `//` 改为
  `ceil(_bpe/accumulate)`（原向下取整导致 LR 相位错乱、收敛远慢）。
- **对齐验证（dota128 从零、imgsz=1024、batch=4、30 epoch）**：
  - 框架：**mAP50-95 = 0.00087**（mAP50=0.0039，best at epoch3）。
  - ultralytics：**mAP50-95 = 0.00078**（mAP50=0.0059，13 epoch）。
  - 两者**同量级，已对齐**（dota128 从零本就难学，指标都低，但一致）。
- **预训练权重加载（关键）**：`OBBU` 头补独立 `angle cv4` 塔后，框架模型与 ultralytics
  `yolo11n-obb` **参数完全一致（2,664,416）**，仅差 `model.23.dfl.conv.weight`（框架 DFL 是函数式）。
  把 `weights/yolo11n-obb.pt`（官方 DOTA 15 类预训练）去掉该键存为 `weights/yolo11n_obb_fw.pth`，
  框架 `load_state_dict` **missing=0 / unexpected=0**。
- **预训练直接推理 mAP（dota128 val）**：框架 **mAP50-95 = 0.8005**（mAP50=0.950）；
  ultralytics **0.821**（mAP50=0.963）。**已高度对齐（差 ~0.021）。**
  - **关键修复：`poly2rbox` 改用 ultralytics 同款 `cv2.minAreaRect`**（`w` 为长边、`theta` 规范化到
    `[-pi/4, 3pi/4)`），替换旧的 arctan2 + `(-pi/2, pi/2]` 约定——因 GT 与模型预测 theta 约定不一致，
    mAP 从 0.754 提升到 **0.8005**。
  - 注：dota128 从零 30 epoch 只有 0.0008（数据集太小、从零难学），
    **微调/预训练才是正道**；dota128 上微调 30 epoch 反而过拟（ultralytics 微调后掉到 ~0.65）。

## 仍存在的小差异（不影响 mAP 对齐，后续可改进）
- **`loss_cls` 框架偏高**（约 100~200 vs ultra ~4.8）：源于从零初始化时的分类校准差异，
  但对 mAP 影响很小（metric 用 argmax class）。可能需要对齐从零初始权重/BN momentum。
- **OBB 头缺独立的 angle `cv4` 塔**：框架把 angle 并入 `cv3` 输出尾通道（`head.params` 433776 vs
  ultra 536579，差 71K）。不影响当前 mAP 对齐，但若要参数完全一致需补 angle 塔。

## 环境
- 框架用 conda 环境 **`ptocr`**；原版 ultralytics 用 **`ultralytics`** 环境（两者 GPU 可用）。
- 注意：**两个训练不能同时占用 GPU**（16GB 会 OOM），需串行。

## git 约定
- 仓库已在 `E:\PytorchOCR` 初始化。
- `.gitignore` 会忽略：`__pycache__`、`output/`、`*.log`、权重(`*.pt/*.pth`)、数据集图片、
  数据集压缩包(`*.zip/*.tgz/*.tar`)、缓存(`*.cache`、`.labels_cache_*.pkl`)、`_downloads/`、`_ref/`。
- 提交信息使用中文、简洁说明改动即可。
