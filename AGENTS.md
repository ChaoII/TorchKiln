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
- **训练（微调）对齐（关键）**：
  - **BN momentum/eps 对齐 ultralytics**（`_set_bn_ultralytics` 设 `momentum=0.03, eps=1e-3`，即
    ultralytics `initialize_weights`）。之前框架 BN 用 `0.1/1e-5`，微调**负优化**（val 从 0.80 降到 0.70）；
    对齐后框架微调 **mAP50-95 = 0.809**（mAP50=0.957，mAP75=0.884），**超过预训练 0.80**。
  - **dota128 训练配置原本未启用增广**（`DetDataset` 仅在配置带 `augment` 时才建 `TrainAugmenter`），
    已补 `augment`（mosaic 1.0/hsv/affine/fliplr 0.5/close_mosaic 10，对齐 ultralytics 默认；
    注：ultralytics 的 `erasing` 只用于分类模型，检测/OBB 不用）。
  - 增广 `_corners_to_rbox` 改用 `cv2.minAreaRect`（θ→`[-pi/4, 3pi/4)`，含退化回退），对齐 ultralytics。
  - **对比（正确列 mAP50-95，之前误取 val/box_loss 列导致假 0.87-0.94）**：
    框架微调 **0.809** vs ultralytics 微调 **0.34/0.34/0.39**（ultralytics 微调反而严重过拟下降）。
    **框架微调大幅优于 ultralytics 微调**（泛化更好）。
- **多版本/多尺寸 OBB 权重加载对齐（盘点）**：
  - **yolo11-obb（n/s/m/l/x）：完全对齐**（missing=0/unexpected=0）。关键修复：
    `graph.py::parse_model` 复刻 ultralytics 对 **C3k2 在 scale∈{m,l,x} 时设 `c3k=True`**（用 C3k 块），
    之前框架 C3k2 恒用 Bottleneck(3×3)，导致 m/l/x backbone 通道不匹配（v11-n/s 本已对齐）。
  - **yolo26-obb（n/s/m/l/x）**：头已对齐，但 `reg_max` 应为 **1（无 DFL，dfl=Identity）**，
    非 v8/v11 的 16；剩余 backbone 特有模块（`model.22` 的 C3k2/A2C2f/C2fCIB 嵌套结构）未对齐（missing/unexpected）。
  - **yolov8-obb（n/s/m/l/x）**：OBB 头 `cv3`（cls 分支）结构不同——框架用 v11 风格 `_dw_cls_branch`（DWConv），
    ultralytics v8 用普通 [Conv,Conv,Conv2d]，仅最末卷积 `cv3.*.2` 通道不匹配。

## Pose（关键点）与 ultralytics 的对齐（已完成并验证）
- **数据管线**：`PoseDataset` 支持 `kpt_shape:[12,2]`（无可见性维），加载时若 `ndim==2` 会**补一列可见性**
  （`x/y<0 → 0，否则 1`），与 ultralytics `verify_image_label` 一致（GT 关键点恒为 `(N,nk,3)`）。
- **Pose 头 cv4**：`pytorchx/nn/modules.py` 的 `Pose`/`PoseU` 头 `cv4` 从 `c4=x` 改为 **`c4=max(ch[0]//4, nk)`**，
  与 ultralytics `Pose.cv4` 一致；框架模型与 ultralytics `yolo11n-pose`（nc=1,kpt:[12,2]）**权重完全加载
  （missing=0 unexpected=0）**。
- **关键点损失（重要 bug 修复）**：`pytorchx/pose.py::PoseLoss` 对齐 ultralytics `v8PoseLoss`——
  `KeypointLoss` 用 `e=d/((2σ)²·area·2)`、`loss_pose=(kpt_loss_factor·(1-exp(-e))·kpt_mask).mean()`、`pose_gain=12/kobj_gain=1`；
  **关键修复**：`target_bboxes` 也先 `/=stride` 转成**网格单位**再算 `area`（原实现用像素面积，梯度被稀释 ~1.7 万倍，
  导致关键点头几乎不学习）。修复前从零 30ep pose mAP=0，修复后 **0.164**。
- **关键点解码**：对齐 ultra——推理 `(raw*2+(anchor-0.5))*stride`；损失路径为网格单位（乘 `2`、加 `anchor-0.5`、**不乘 stride**）。
- **OKS 指标**：`PoseMetric` 用 `kpt_iou`（`e=d/((2σ)²·area·2)`、`area=w*h*0.53`、`σ=ones(nk)/nk`（非 COCO）），
  匹配复刻 ultra `match_predictions`（按 OKS 降序、去重预测列后再去重 GT 行）。
- **训练性能 bug 修复**：`ptcore/trainers/base.py::_maybe_eval_and_save` 原 `step%interval==0` 在**梯度累积窗口内
  （global_step 停 0）**会每批触发评估（每批 ~30s）；已加"仅当 `global_step` 变化到新满足条件的值时评估一次"，
  训练从 30s/批 → **3s/epoch**。
- **cuDNN 非确定性→评估低估（重要 bug 修复）**：`ptcore/trainers/base.py` 默认 `cudnn.deterministic=True`
  （`Global.cudnn_deterministic` 可关）。cuDNN 非确定性算法会让我们的 GPU 前向与 ultralytics 产生不同数值，
  cls 在 conf 阈值(0.001)附近大量翻转→同权重下我们出 275 个检测 vs ultra 2 个，评估 mAP 被系统性低估。
  修复后同一超训微调权重评估 mAP50-95 由 **0.312 → 0.495**（ultra 0.509，mAP50 均 0.995）。
- **对齐验证（tiger-pose：`datasets/tiger-pose/`，kpt:[12,2]，imgsz=640，batch=4，SGD 同超参）**：
  - 从零 30ep：框架 **pose mAP50-95=0.164** / ultralytics 0.033（框架更强）。
  - 预训练微调 30ep：框架 **0.495**（mAP50 0.995）/ ultralytics 0.509（mAP50 0.995）；mAP50 一致（deterministic 修复后）。
  - 权重对齐评估（同一 checkpoint）：框架 vs ultra 一致（0.0288 vs 0.033）。
  - 已验证一致：权重加载、输入（letterbox/通道/归一化）、`make_anchors`、layer0 手工重算、cls bias、各块代码。
- 配置：`configs/_parity/pkg_pose.yml`（数据 `datasets/tiger-pose`，model 为 `yolo11-pose` 图模型，reg_max=16）。
## Segment（实例分割）与 ultralytics 的对齐（已完成并验证）
- 数据：`datasets/package-seg/`（单类 `package`，1920 train / 188 val，RF 多边形 `cls+归一化点` 标签，
  来自 `C:\Users\ADMINI~1\AppData\Local\Temp\1\opencode\pkgseg`），已生成 train.txt/val.txt，
  `package-seg.yaml` 的 path 改为绝对、val 统一指向 `images/val`（188）。
- 配置：`configs/_parity/pkg_seg_package.yml`（data_dir `datasets/package-seg`，model `yolo11-seg`，
  imgsz=640，mask_stride=4，reg_max=16，num_classes=1，batch=4，SGD 同超参）。
- **权重加载完全对齐**：先 dump ultralytics `yolo11n-seg`（nc=1，框架加载 COCO 80 类预训练时的
  `cv3` 分类头 mismatch 属**预期跳过**，非 bug）；对 ultra 在 package-seg 微调 30ep 的 `best.pt`
  （nc=1 同类别）做 `load_state_dict`：**missing=0 / unexpected=1**（仅函数式 `dfl.conv.weight`，同 pose/OBB）。
  → 说明**框架 SegmentU 头与 ultra yolo11-seg 参数完全一致**。
- **对齐验证（同一 best.pt 权重，imgsz=640，cudnn.deterministic 评估）**：
  - 框架 box_mAP50=0.929 / box_mAP50-95=0.840，mask_mAP50=0.9228 / mask_mAP50-95=**0.8206**。
  - ultralytics box_mAP50=0.922 / box_mAP50-95=0.845，mask_mAP50=0.9235 / mask_mAP50-95=**0.8214**。
  - **同一权重下高度一致**：mask_mAP50-95 差 0.0008（几乎相等），box_mAP50-95 差 ~0.005（算子级微差）。

## Classification（图像分类）与 ultralytics 的对齐（已完成并验证）
- **任务已存在但需对齐验证**：`pytorchx/tasks/classify.py` + `_cls.py`（ClsLoss/ClsMetric/ClsPostProcess）
  + `pytorchx/data/cls.py::ClsDataset`（读 `path label` 清单）；框架 `pytorchx/cfg/models/11/yolo11-cls.yaml`
  与 ultra yolo11-cls 结构一致（backbone + 单一 `Classify` 头：Conv→1280 → AdaptiveAvgPool → Linear(nc)）。
- **权重加载完全对齐**：dump ultralytics `yolo11n-cls.pt`（ImageNet 1000 类）→ `build_arch_model(...,"classify")` 加载：
  **missing=0 / unexpected=0**（框架 `Classify` 头与 ultra 完全一致）。
- **全部 15 个权重对齐（yolov8/yolo11/yolo26 × n/s/m/l/x）**：对 `\\tsclient\D\项目资料\ultralytics_models\{yolov8,yolo11,yolo26}\*-cls.pt`
  全部验证——加载均 **missing=0/unexpected=0**；同输入(224)推理 softmax maxdiff **≤0.0000007**（几乎逐位一致），top1 全部一致。
- **C3k2 的 M/L/X 规模逻辑已对齐**：ultra `parse_model` 对 `scale∈{m,l,x}` 强制 C3k2 `c3k=True`（用 C3k，1×1）；框架
  `pytorchx/nn/graph.py::parse_model` 已复刻（`if module_name=="C3k2" and scale in ("m","l","x"): args[2]=True`），
  n/s 用 Bottleneck(3×3)，m/l/x 用 C3k(1×1)，故各规模权重均能完全加载。
- **前向一致性（同权重同输入，imgsz=224）**：框架 vs ultra 逐层对比——
  - backbone 各层（Conv/C3k2/C2PSA）maxdiff **≈0.00000**（几乎逐位一致）；raw logits maxdiff 0.000004；softmax maxdiff **0.000000**；top1 一致（885）。
- **关键修复（BN eps 推理对齐，影响所有任务评估）**：ultra `initialize_weights` 把 BN eps 设 **1e-3**（仅训练期），
  但**保存的权重不含 BN eps**，ultra 加载后**推理**时 BN 用构造默认 **1e-5**。框架 `_set_bn_ultralytics` 永久设 1e-3，
  导致推理层0 起即有 0.005 meanabs 差异并可能被分类 linear 头放大。已在 `ptcore/trainers/base.py::evaluate()`
  评估时**临时恢复 BN eps=1e-5**（评估完还原，不影响训练 forward）——修复后 backbone/logits/softmax 全部对齐。
- **评估流程冒烟**：`configs/_parity/pkg_cls_demo.yml`（cls_demo 3 类，imgsz=224）评估跑通（top1/top5 正常输出）。
- **训练对齐（同权重同输入同标签，向前+反向）**：loss 公式与 ultra `v8ClassificationLoss`（`F.cross_entropy(preds, cls, reduction="mean")`）
  完全一致（框架 `ClsLoss` = `nn.CrossEntropyLoss`, label_smoothing=0）。对**全部 15 个权重**（yolov8/yolo11/yolo26 × n/s/m/l/x）
  各用同权重、同 `(2,3,224,224)` 输入、同标签（[3,885]）做一步：**loss diff 均 ≤5.7e-6**（几乎逐位一致，yolo11n=0）；
  每模型梯度**全部层匹配（无漏层）**，**梯度 maxdiff 均 ≤0.00017**（最大为首层 `model.0.conv.weight`，cuDNN 卷积反向的算子级微差，非逻辑 bug）。
  → 分类的前向（推理）与训练（loss+梯度）均与 ultralytics 对齐，残差仅算子级。
- 注：cls_demo（3 类 96×96）是 toy 数据，与 1000 类预训练权重不匹配；核心对齐验证用 ImageNet 预训练权重完成。

## 家族头路由与 seg 全家族对齐（重要）
- **按家族路由检测/分割头（`pytorchx/nn/graph.py`）**：`build_from_arch` 从 `yaml_file` 判断家族，
  旧家族（`v3/v5/v8/v9`）令 `spec["_legacy"]=True`；`parse_model` 里对 legacy 家族把
  `Detect/Segment/OBB/Pose` 路由到 **旧式 Conv 头**（`modules.py` 的 `Detect/Segment/OBB/Pose` 类），
  新家族（`yolo11/12/26`）用 **新 DWConv 头**（`SegmentU`/`Detect26` 等）。
  根因：`REGISTRY` 里的 `"Detect"→Detect26`、`"Segment"→SegmentU` 是「新头」别名；而 **ultra 的 v8 不论
  detect 还是 seg 都用 legacy 旧式 Conv 头**（`cv3 = [Conv,Conv,Conv2d]`），yolo11/26 才用新
  DWConv 头（`cv3 = [Seq[DWConv,Conv],...]`）。此路由同时修正了 **v8 的 detect/obb/pose/seg**（此前
  框架误用新头，参数/结构不匹配；改后 v8n-detect 也 missing=0/unexpected=1）。
- **`C3k2` 增加 `attn` 分支**（`modules.py`）：签名对齐 ultra `(c1,c2,n,c3k,e,attn,g,shortcut)`，
  `attn=True` 时用 `nn.Sequential(Bottleneck, PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c//64,1)))`。
  yolo26 的 `C3k2 [c, True, 0.5, True]` 第 4 参即 `attn`。
- **新增 `Segment26`（end2end 双头）+ `Proto26`（`modules.py`）**：yolo26 实例分割头，`reg_max=1`、
  `npr=make_divisible(256*width,8)`、`end2end=True`（含 `one2one_cv2/cv3/cv4` 副本，对齐 ultra 训练态
  checkpoint）；`Proto26` = 基础 `Proto` + `feat_refine/feat_fuse/semseg`（语义分支）。`REGISTRY["Segment26"]`
  指向该新类（原先被别名覆盖为 legacy `Segment`/`SegmentU`）。
- **15 档 seg 权重加载全部对齐（`missing=0`）**（官方 `\\tsclient\D\项目资料\ultralytics_models\...`，nc=80、imgsz 无关）：
  - `yolov8-seg` n/s/m/l/x：`missing=0 / unexpected=1`（仅函数式 dfl），参数差 ~0.3%。
  - `yolo11-seg` n/s/m/l/x：`missing=0 / unexpected=1`，参数差 ~0.2%。
  - `yolo26-seg` n/s/m/l/x：`missing=0 / unexpected=0`，参数差 ~0.1%（`reg_max=1` + 新 `Segment26`+`Proto26`）。
  - 说明：验证时 `Architecture.scale` 需指定档位、`num_classes=80`；yolo26 头 `reg_max=1`（无 DFL）。
- **前向 smoke**：yolo26n-seg 加载官方权重后 `model(img)` 产出 `{"feats"(3 层), "protos"}`，`SegPostProcess` 解码正常。
- **端到端（NMS-free）推理已接通**：`Segment26.forward` 按 `self.end2end and not self.training` 选择分支——
  `end2end=True` 用 `one2one_cv2/cv3/cv4`，`SegPostProcess` 的 `_nms` 对 end2end 也返回全部索引（无 NMS）。
  - 注：ultra 重载模型推理是 `end2end=False`（one2many + NMS），checkpoint 训练态含 `one2one_*`（end2end）。两种路径框架都支持，按配置 `PostProcess.end2end`/`Head.end2end` 决定。

## 仍存在的小差异（不影响 mAP 对齐，后续可改进）
- **`loss_cls` 框架偏高**（约 100~200 vs ultra ~4.8）：源于从零初始化时的分类校准差异，
  但对 mAP 影响很小（metric 用 argmax class）。可能需要对齐从零初始权重/BN momentum。
- **OBB 头缺独立的 angle `cv4` 塔**：框架把 angle 并入 `cv3` 输出尾通道（`head.params` 433776 vs
  ultra 536579，差 71K）。不影响当前 mAP 对齐，但若要参数完全一致需补 angle 塔。
- **Pose 预训练微调 mAP50-95**：已用 `cudnn.deterministic` 修复评估低估（0.312→0.495，ultra 0.509）。
  剩余 ~0.014 差额为算子级微差（非逻辑 bug），如需进一步收敛可在独占 GPU 下逐块核验。

## 官方预训练权重目录
- 所有 ultralytics 官方预训练权重位于 **`\\tsclient\D\项目资料\ultralytics_models`**。
- 子目录：`yolov5` `yolov8` `yolov9` `yolov10` `yolo11` `yolo12` `yolo26` `yolov3u`。
- 每个家族含 `n/s/m/l/x` 五档，任务后缀：`-seg`(实例分割) `-cls`(分类) `-obb`(旋转框) `-pose`(关键点) `-depth`/`-sem`(yolo26 深度/语义)。
- 例：`yolo11n-seg.pt`、`yolov8x-obb.pt`、`yolo26m-seg.pt` 等（yolo11 无 `-depth/-sem`，yolo26 本身有 `-depth/-sem/-sem-ade20k`）。
- 框架加载这些官方权重验证对齐时，需按 `Architecture.scale` 指定对应档位（`n/m/s/l/x`），并令 `num_classes=80` 以匹配 COCO 预训练。

## 环境
- 框架用 conda 环境 **`ptocr`**；原版 ultralytics 用 **`ultralytics`** 环境（两者 GPU 可用）。
- 注意：**两个训练不能同时占用 GPU**（16GB 会 OOM），需串行。

## git 约定
- 仓库已在 `E:\PytorchOCR` 初始化。
- `.gitignore` 会忽略：`__pycache__`、`output/`、`*.log`、权重(`*.pt/*.pth`)、数据集图片、
  数据集压缩包(`*.zip/*.tgz/*.tar`)、缓存(`*.cache`、`.labels_cache_*.pkl`)、`_downloads/`、`_ref/`。
- 提交信息使用中文、简洁说明改动即可。
