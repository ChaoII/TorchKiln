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

## OBB（旋转框）实现与 ultralytics 的差异（已部分对齐 ultralytics）
- **旋转框 NMS 已替换为 ultralytics 原生方案**（`pytorchx/det/rbox.py::nms_rotated`）：
  由原来 **O(N²) 纯 NumPy 逐对多边形裁剪**（2000 框要 87 秒）改为
  **`batch_probiou`(ProbIoU 上三角矩阵) + `TorchNMS.fast_nms` 上三角抑制**，GPU 向量化，
  2000 框降到约 **0.13 秒**；并加了 `max_candidates`（默认 3000，仿 ultralytics `max_nms`）
  在密集图上按置信度截断候选，避免 `(N,N)` 矩阵 OOM。
  **已验证：框架 `nms_rotated` 与 ultralytics `TorchNMS.fast_nms(boxes,scores,0.7,iou_func=batch_probiou)`
  输出索引完全一致（200 框随机用例 maxdiff=0.0）。**
- **`probiou` 高斯方差已改为 `w^2/12`**（之前是 `(w/2)^2`），与 ultralytics `batch_probiou`
  数值完全一致（maxdiff=0.0）。注意返回值已去尾维（`(N,)`/`(B,A,M)`），调用方别多 squeeze。
- **指标 IoU 改为 probiou**：`pytorchx/det/metric.py::_iou` 对 `box_format=xywhr` 用
  `probiou`（原用多边形裁剪 `poly_iou_np`），与 ultralytics OBB 指标一致。
- **仍存在的差异**：
  - OBB 损失**不含单独的 angle loss**：`ObbLoss.forward` 只有 `box_gain*(1-probiou)+cls_gain*BCE`
    （`angle_gain` 定义了但未参与）；ultralytics `v8OBBLoss` 有 box+cls+dfl+**angle loss**。
  - 角度解码：框架 `dist2rbox` 用 `theta=atan(angle_raw)`；ultralytics 用 `angle` 分支解码。

## 待办 / 已知问题（重要）
- **框架 OBB 从零训练 mAP 全程为 0、loss 至 epoch 21 起变 NaN**：初始 loss 巨大
  （loss_cls ≥~500-1600），dota128 训练 30 epoch 未收敛。这是**框架自身训练管线/损失问题**，
  在改造 NMS 前就存在（原始 also loss~1100）。要真正对齐 ultralytics 精度，需先定位并修复
  OBB 损失/数据管线（含 `accumulate=nbs/batch` 较大导致 LR 调度/步数偏少、cls 分配异常）。
- dota128 训练从零、imgsz=1024、batch=4 已验证能跑通 1 步 + 评估（eval fps ~40）；30 epoch 现配置可复现。

## git 约定
- 仓库已在 `E:\PytorchOCR` 初始化。
- `.gitignore` 会忽略：`__pycache__`、`output/`、`*.log`、权重(`*.pt/*.pth`)、数据集图片、
  数据集压缩包(`*.zip/*.tgz/*.tar`)、缓存(`*.cache`、`.labels_cache_*.pkl`)、`_downloads/`、`_ref/`。
- 提交信息使用中文、简洁说明改动即可。
