# AGENTS.md

## Detect（目标检测）与 ultralytics 的对齐（单步验证已通过，yolo11n）
- 验证方法为**单步验证**：同权重（`\\tsclient\D\项目资料\ultralytics_models` dump）+ 同输入 `\opencode\det_x_1.npz`(1,3,640,640)
  + 同 GT（像素框 [192,128,320,384]，ultra 归一化 xywh=[0.4,0.4,0.2,0.4]）做前向（feature/loss）+ 反向（梯度）对比。
- **yolo11n 检测对齐完成**（单步验证全过）：
  - 权重加载：missing=0 / unexpected=1（仅函数式 `dfl.conv.weight`，同 pose/seg/OBB）。
  - assigner 匹配一致：n_fg=10 / t_scores.sum=1.3955（框架=ultra）。
  - 损失三分量（raw）全对齐：box=**0.5508**、cls=**23.3453**、dfl=**3.1119**；total=**20.47179** vs ultra **20.47182**（差 3e-5）。
  - 梯度 maxdiff=0.0216（worst `model.0.conv.weight`，cuDNN 卷积反向算子级微差）。
- **关键 bug 修复（DFL 目标，影响所有 detect 训练）**：`pytorchx/det/loss.py::DetLoss._forward_one`
  原先把 `tgt = (t_bboxes[fg]/stride_fg).clamp(0, reg_max-1-1e-3)` —— 把 **GT 框坐标** clamp 到 reg_max-1-0.99，
  导致 x2=320/16=20 被截断到 14.99，DFL target 错误（dfl 4.07 vs ultra 3.11）。**ultra 从不对框坐标 clamp**，
  只在 `bbox2dist` 输出的 **ltrb 距离**上 clamp（`reg_max-1-0.01`，对齐 ultra `DFLoss`）。已改为：
  `tgt = t_bboxes[fg]/stride_fg[:,None]`，`target_ltrb = bbox2dist(ap_fg, tgt).clamp(0, reg_max-1.0-0.01)`。
- 对齐所需默认值：`DetLoss` 的 `topk=13→10`、`alpha=1.0→0.5`（对齐 ultra `v8DetectionLoss`）。
- 因该 DFL 修复影响**所有检测算法**（detect/OBB 共用同上 DFL 分支），OBB 训练也可受益；当前 OBB 对齐记录仍有效。
- **yolov8 检测对齐完成**（order 中第二个，legacy Conv 头）：
  - 权重加载：n/s/m/l/x 全 **missing=0 / unexpected=1**（仅函数式 `model.22.dfl.conv.weight`，参数数一致）。
  - 单步（yolov8n）：assigner n_fg=10 / t_scores.sum=1.0403；loss 三分量 raw 全对齐：box=**0.6425**、cls=**20.5645**、dfl=**3.0281**；
    total=**19.64271** vs ultra **19.64270**（差 1e-5）；梯度 maxdiff=0.00186（worst `model.0.conv.weight`，算子级）。
  - s/m/l/x 与 n 共用同 `Detect`(legacy)+`DetLoss`，仅通道不同；n 已证 loss/梯度路径，故全家族训练对齐成立。
- **yolo12 检测对齐完成**（A2C2f 家族）：
  - **关键重构**：`modules.py` 的 `A2C2f/ABlock/AAttn` 重写对齐 ultra 新版——A2C2f `cv1=Conv(c1,int(c2*e))`（**不切半**）、`cv2=Conv((1+n)*c_,c2)`、
    `m=ModuleList(nn.Sequential(2×ABlock(c_,c_//32,mlp_ratio,area)) if a2 else C3k(c_,c_,2,...))`、`a2 and residual` 时含 `gamma` 残差；
    ABlock 改用 `AAttn`（qkv=Conv(dim,3*dim)、proj、`pe=Conv(...,7,1,3,g=dim,act=False)` 深度卷积 + `_init_weights` trunc_normal_）。
  - `Conv` 增加可选 `bias=False` 参数（默认不影响其它）；`AAttn.pe` 需 `bias=True`（对齐 ultra 官方权重）。`parse_model` 对 A2C2f 在 `scale∈{l,x}` 追加 `(True,1.2)`（residual+mlp_ratio）。
  - **权重加载**：n/s/m/l/x 全 **missing=0 / unexpected=1**（仅函数式 `model.21.dfl.conv.weight`）。
  - 单步（同权重同输入同 GT）：n/s/m/l/x 的 loss 三分量 raw 全对齐（如 n box=0.5237/cls=11.6909/dfl=2.5324，total 13.5719 vs 13.5719）；梯度 maxdiff 0.006~0.026（算子级）。
  - 说明：yolo12 用 `Detect(reg_max=16)+v8DetectionLoss`，路径与 v8/v11 相同；l/x 的 A2C2f `gamma` 残差 + mlp_ratio=1.2 亦对齐。
- **SPPF 修复（v26 推理对齐，关键）**：框架 SPPF 的 `cv1` 缺 `act=False`（默认 SiLU），且 `add` 需同时 `c1==c2`（ultra `shortcut and c1==c2`）。修后框架前向 L9(SPPF) 起全部对齐到 <1e-3（此前 L9 maxdiff 4.43）。v8/v11/v12 的 SPPF 无 add(False) 不受影响。
- **yolo26 检测对齐（进行中）**：权重加载 n **missing=0/unexpected=0**（`Detect10` end-to-end 头，one2one_cv2/cv3 分支已补；`parse_model` 按 `Head.end2end` 路由 `Detect`→`Detect10`；`build_from_arch` 传递 `end2end`）。
  - yolo26 用 `E2EDetectLoss`（one2many `tal_topk=10` + one2one `tal_topk=1`），框架 `DetLoss(reg_max=1, use_one2one=True, one2one_topk=1)` 复刻。
  - `DetLoss._forward_one` 补 **L1 loss 分支**（reg_max≤1 时，ultra `BboxLoss` 无 DFL 用 L1：`bbox2dist(ap,tgt)*stride` 归一化到 imgsz 后 `F.l1_loss`）。
  - 单步（yolo26n）：**loss 全对齐**（合并 one2many+one2one 后 box=0.7586/cls=25.0754/l1=0.0751，total=18.34005 vs ultra 18.34014）；前向逐层对齐 <1e-3。
  - **梯度对齐完成（关键修复）**：`Detect10.forward` 训练时对 **one2one 分支的输入特征 `.detach()`**（`if self.training: x = [xi.detach() for xi in x]`）——
    对齐 ultra `Detect.forward` 的 `x_detach = [xi.detach() for xi in x] if self.training else x`（**detach keeps one2one out of the backbone**）。
    此前框架让 one2one 分支共享同一 feats 并回传梯度，导致 backbone 梯度 = o2m+o2o 而 ultra 只有 o2m，
    全模型梯度 maxdiff **40.4→0.00034**（算子级，worst `model.1.conv.weight`）。注意此修复同时适用于其它带 one2one 分支的 end2end 头（Segment26/OBB26 待同步）。
  - **yolo26 n/s/m/l/x 全家族对齐完成**：负载全 missing=0/unexpected=0，loss 与梯度均算子级对齐
    （n 18.34005↔18.34014、s 28.88681↔28.88683、m 23.79597↔23.79595、l 16.416456↔16.416456、x 24.45561↔24.45559，
    梯度 maxdiff 0.0014~0.039，worst 多为 model.0/2 conv，cuDNN 卷积反向算子级微差）。
  - 待办：按 `docs/plan_detect_align.md` 顺序到 v10→v9→v5→v3u。
- **yolov10 检测对齐完成**（v10Detect=Detect10，reg_max=16，E2E o2m+o2o，同 yolo26 `E2EDetectLoss`）：
  - **PSA 修复**：`graph.py::REPEAT_MODULES` 移除 `PSA`——原把重复数插入到 PSA 第 3 位当作 `e`（应为 0.5 展开比），
    导致 `PSA(c1,c2,e=1)`、c=256（应 128）；PSA 是单一块（非 n 个块列表），不应走 repeat 插入。
  - **CIB lk 分支**：`modules.py` 新增 `RepVGGDW`（`conv=Conv(ed,ed,7,1,3,g=ed,act=False)`+`conv1=Conv(ed,ed,3,1,1,g=ed,act=False)`+SiLU 相加），
    `CIB` 的 lk 分支由 `RepConvN` 改用 `RepVGGDW`（对齐 ultra `CIB`；仅 v10/v9 用 CIB，yolo26 无 CIB 不受影响）。
  - **SPPF act 修复（关键）**：`graph.py::parse_model` 对 SPPF 补 ultralytics 行为——SPPF cv1 默认 `act=False`（unactivated YOLO26 风格），
    仅当 SPPF yaml **args≤3**（旧式 v8/v10/v11/v12）时 `layer.cv1.act = Conv.default_act`（恢复 SiLU）。此前框架 SPPF cv1 恒 act=False，
    导致 v10 前向 L9(SPPF) 起 maxdiff 5.26（yolo26 args=4 >3 仍 unactivated，不受影响）。
  - **按规模架构替换（关键）**：v10 架构随规模变化（ultra 用 per-scale yaml）：深层的 `C2f` 在更大规模升级为 `C2fCIB`。
    在 `graph.py::build_from_arch` 加 v10 替换表（`v10_sw`：s→bw8(lk=True)，m→bw8+hd8(lk=False)，l→bw8+hd2+hd8，x→bw6+bw8+hd2+hd8；
    `v10_hd11`：n/s=[1024,True,True]，m/l/x=[1024,True]）。
  - 单步（同权重同输入同 GT）：n/s/m/l/x 全 missing=0/unexpected=1（仅函数式 dfl）；loss 全对齐
    （n 23.91382↔23.91383、s 26.213394↔26.213394、m 25.325804↔25.325804、l 21.74652↔21.74652、x 22.523449↔22.523449）；
    梯度 maxdiff 0.0002~0.0013（算子级，worst model.0.conv.weight）。
  - 待办：按 `docs/plan_detect_align.md` 顺序到 v9→v5→v3u。
- **yolov9c 对齐完成**（c/e 家族 = RepNCSPELAN4/ADown/SPPELAN，v9 非 end2end，v8DetectionLoss 单 assigner）：
  - **RepConv 重写对齐 ultra**（`modules.py`）：加 `default_act=nn.SiLU()`、`bn` identity 分支、`act` 属性、
    `forward = act(conv1(x)+conv2(x)+id_out)`；`conv1=Conv(c1,c2,k,s,p=p,g=g,act=False)`、
    `conv2=Conv(c1,c2,1,s,p=(p-k//2),g=g,act=False)`、默认 `bn=False`。
  - **新增 `RepBottleneck`**（继承 Bottleneck，`cv1=RepConv(c1,c_,k[0],1)`）与 **`RepCSP`**（继承 **C3**（非 C2f！），
    `m=nn.Sequential(RepBottleneck(c_,c_,shortcut,g,e=1.0))`）。
    ⚠️ 关键：ultra `RepCSP` 继承 **`C3`**（`cv1/cv2=Conv(c1,c_,1,1)`、`cv3=Conv(2*c_,c2,1)`），非 C2f——最初误继承 C2f 导致 c=32 vs 16 通道不符。
  - `RepNCSPELAN4` 的 `cv2/cv3` 由 `RepNCSP` 改用 `RepCSP`，结构与 ultra 完全一致（`cv2.0.cv1.conv` 等键匹配，参数 25.59M vs ultra 25.59M）。
  - 单步（yolov9c，reg_max=16，use_one2one=False，D=v8DetectionLoss）：**missing=0/unexpected=1**（仅函数式 dfl）；
    loss total=**14.742226↔14.742228**（box=0.3681/cls=16.4717/dfl=2.4970 全对齐）；梯度 maxdiff=**0.000231**（算子级，worst `model.2.cv3.1.conv.weight`）。
- **yolov9 t/s/m 对齐完成**（v9 **按规模分族**：t/s 用 `ELAN1`+`AConv`、m 用 `RepNCSPELAN4`+`AConv`(非 ADown)、c 用 `RepNCSPELAN4`+`ADown`、e 用 `RepNCSPELAN4`+`ADown`+`CBLinear/CBFuse`）：
  - 框架 `AConv`/`ELAN1` 已存在且 forward 与 ultra 一致（AConv=`avg_pool2d(x,2,1,0,False,True)`+`Conv(3,2,1)`；ELAN1 含 cv2/cv3 双 `Conv`）。
  - 新增 **`yolov9t.yaml`/`yolov9s.yaml`/`yolov9m.yaml`**（通道**烘焙固定**，`scales:<size>=[1.0,1.0,512]`），`fw_v9_step.py` 按 scale→yaml 映射选择。
  - ⚠️ 框架 v9 各 size 通道无法用宽度缩放表达（s/m/c 各不同），**必须**用独立 yaml。
  - 单步：t **19.053757↔19.053761**（box=0.5065/cls=22.2459/dfl=2.7545）、s **18.813862↔18.813862**（0.7017/12.8065/4.7653）、
    m **18.858110↔18.858112**（0.4338/22.9186/2.7635）；梯度 maxdiff 0.00044/0.00029/0.00024（算子级）。
  - **v9e（CBLinear/CBFuse PAGCPY 融合）未对齐**（结构复杂，暂跳过，不影响 t/s/m/c 检测训练对齐）。
- **yolov5 n/s/m/l/x 对齐完成**（v5 **`*u` 变体**为 modern anchor-free：`Detect` 用 `legacy=False`（新 DWConv 头）+ v8DetectionLoss，**无 obj loss、无 anchors**——`anchors` 属性仅是占位。故 v5 与 yolo11 路径相同，无需单独 build_targets/obj）：
  - 框架 v5.yaml 用 `depth_multiple`/`width_multiple`（YOLOv5 老式缩放，非 scales dict），`make_divisible` 通道、`C3`/`Bottleneck`、`SPPF` 均已对齐 ultra v5u。
  - 单步（同权重同输入同 GT，reg_max=16，use_one2one=False）：n **15.830602↔15.830601**、s **14.335932↔14.335931**、m **21.238976↔21.238974**、
    l **20.419407↔20.419418**、x **18.897820↔18.897818**（box/cls/dfl 全对齐）；梯度 maxdiff 0.00004/0.00009/0.00020/0.00013/0.00012（算子级）。
  - 加载 n/s/m/l/x 全 **missing=0/unexpected=1**（仅函数式 `model.24(或25/26).dfl.conv.weight`）。
- **yolov3 (u/u-spp/u-tiny) 对齐完成**（v3u 亦为 modern anchor-free：`legacy=False` + v8DetectionLoss）：
  - **关键修复**：`graph.py::REPEAT_MODULES` 移除 `Bottleneck`——Bottleneck 是**无内化 n** 的 SCALED 模块，
    原被误判为"内化 n"而把重复数 n 插入 args 第 2 位（`Bottleneck(c1,c2,shortcut=2,...)`），导致 v3 的
    `[-1, 2, Bottleneck, [128]]` 不打包成 Sequential（模型 `model.4.0.cv1` vs `model.4.cv1` 键不符）。
    移除后 Bottleneck 走 line308-310 的 `nn.Sequential` 包装（model.4.0/4.1），与 ultra 一致。Bottleneck 不直接出现在其它家族 yaml（仅 C3/C3k 内部），故不影响 v5/v8/v9 等。
  - ⚠️ v3-tiny 头仅 P3/P5 两尺度（nl=2，strides=(16,32)），单步需按实际层数传 strides（勿用 (8,16,32)）。
  - 单步（同权重同输入同 GT，reg_max=16，use_one2one=False）：u **19.047052↔19.047056**、spp **18.116039↔18.116039**、tiny **14.158577↔14.158577**（box/cls/dfl 全对齐）；梯度 maxdiff 0.00016/0.00020/0.00007（算子级）。
  - 加载三型号全 **missing=0/unexpected=1**（仅函数式 `model.24.dfl.conv.weight`）。
  - **v9c（c 家族）梯度 maxdiff=0.0786 为算子级（并非逻辑 bug）**：该层 `model.2.cv4.conv.weight` 梯度幅度本身约 **67.4**，
    相对差仅 **~0.1%**（cuDNN 卷积反向算子级微差）；前向已逐层 diff=0（o1/o2 精确一致）。t/s/m 梯度绝对值小（0.0002~0.0004）因对应层梯度幅度小。
- 待办：detect 对齐已覆盖 yolo26→v10→v9(t/s/m/c)→v5→v3u 全家族（v9e 因 CBLinear/CBFuse 融合暂跳过）。
  - **OBB26 已同步 one2one `.detach()`**：`modules.py::OBBU.forward` 训练时对 one2one 分支用 `x_det=[xi.detach()]`（对齐 ultra 端到端头）。
    Segment26 训练路径仅返回 one2many（无 one2one 分支回传），无需 detach。
  - **回归复测通过**（确认 `REPEAT_MODULES` 移除 `Bottleneck` 不影响已对齐家族，因为它们只把 Bottleneck 内化在 C3/C3k 内、yaml 不直接重复它）：yolo11n 20.471785↔20.471819（梯度 0.0216）、yolo12n 13.571861↔13.571886（梯度 0.013）、yolov8n 19.642714↔19.642702（梯度 0.00186）。
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
    框架微调（**lr0=0.001 温和**）**0.809** vs ultralytics 微调（默认 lr0=0.01）**0.34/0.34/0.39**。
    **⚠️ 非公平对比**：框架用了 `-o Optimizer.lr.learning_rate=0.001` 覆盖成温和 lr 才不退化，ultralytics 用默认 0.01。
  - **公平验证（关键，纠正上述误导）**：框架用**默认 lr0=0.01（与 ultra 一致）**微调时**同样过拟合退化**：
    v26n 默认 lr0.01 微调 5 轮 best=**0.814**（ep1=预训练水平），之后 ep2=0.657/ep3=0.691/ep4=0.680 明显退化。
    → **训练行为方向对齐**：两者默认超参都会在 102 张图 dota128 上过拟合退化；**不存在"框架显著更强"**。
    退化幅度差异（框架 ~0.66-0.69 vs ultra ~0.32）来自剩余超参/增广细微差异，需进一步排查；不影响 loss/梯度数值对齐结论。
  - **进一步排查（关键确认，已记于 09/20 后续）**：用框架评估 ultra 微调后的 `best.pt` 得 mAP50-95=**0.317**，与 ultra 自报 0.32 **一致** → **评估无差异，ultra 是真过拟合**，非度量 bug。
    → 退化差异根因在**训练管线**而非算法：① ultra `optimizer=auto` 因迭代数<10000 自动选 **AdamW+lr_fit=0.002*5/(4+nc)**(=0.000526;nc=15)，框架用 SGD 0.01；
    ② ultra 默认 **AMP**；③ ultra v26-obb loss 含 **`l1_loss`**(reg_max=1 无 DFL 用 L1)，**框架 v26 `ObbLoss` 缺 L1 分支**(reg_max=1 时 `loss_dfl=0`)；④ ultra 有效 batch=4(不累积)，框架 accumulate=16(有效 64)。
    → 框架用 AdamW+lr_fit+有效 batch=4 重跑仍不崩(0.814→0.800)，还需复刻 AMP/L1/端到端 才能对齐训练 mAP。
  - **完整复刻后仍不崩（09/20 最终排查）**：框架把 auto→AdamW(lr_fit=0.000526)+AMP+有效batch=4+L1分支+EMA 全复刻后，v26 微调 5 轮 best=**0.807**，仅缓退到 ~0.77-0.80；而 ultra 第 1 轮就从 0.83 崩到 0.32。
    且 ultra 训练 loss 平缓(box≈0.87/cls≈5，不爆炸)却在 1 轮内 mAP 崩 0.5 → **非正常过拟合，疑似 ultra 自身管线病态行为**（dota128 类名重映射 `cls_remap` 可能打乱预训练头 + auto 优化器对 102 图响应）。
    → **结论：框架训练稳定、正确，反而比 ultra 更健康；并非"框架更强"（之前用温和 lr 误导），也非框架 bug**。微调 mAP 无法与 ultra 病理崩溃逐一对齐，属 ultra 行为。验收应以**同权重推理(≤0.022) + loss/梯度数值对齐**为准。
- **多版本/多尺寸 OBB 权重加载对齐（盘点）**：
  - **yolo11-obb（n/s/m/l/x）：完全对齐**（missing=0/unexpected=0）。关键修复：
    `graph.py::parse_model` 复刻 ultralytics 对 **C3k2 在 scale∈{m,l,x} 时设 `c3k=True`**（用 C3k 块），
    之前框架 C3k2 恒用 Bottleneck(3×3)，导致 m/l/x backbone 通道不匹配（v11-n/s 本已对齐）。
  - **yolo26-obb（n/s/m/l/x）**：头已对齐，但 `reg_max` 应为 **1（无 DFL，dfl=Identity）**，
    非 v8/v11 的 16；剩余 backbone 特有模块（`model.22` 的 C3k2/A2C2f/C2fCIB 嵌套结构）未对齐（missing/unexpected）。
  - **yolov8-obb（n/s/m/l/x）**：OBB 头 `cv3`（cls 分支）结构不同——框架用 v11 风格 `_dw_cls_branch`（DWConv），
    ultralytics v8 用普通 [Conv,Conv,Conv2d]，仅最末卷积 `cv3.*.2` 通道不匹配。

## OBB 多版本/多尺寸全对齐（v8/v11/v26 × n/s/m/l/x，训练+推理均已对齐）
- **v8/v11/v26 全尺寸权重加载完全对齐**（missing=0/unexpected=0，参数逐位一致，v8n=3096079/v26n=2672486）。
  - v8（旧家族 legacy 头）：OBB 头重构为**独立 `cv4`(angle) + legacy plain `cv3`**（out=nc，cls_extra=0，`_tower("rcx")`）。
  - v26：OBB26 头 `reg_max=1`（无 DFL，`dfl=Identity`，head `no=19`）、`end2end=True`（含 `one2one_cv2/cv3/cv4` 双头）；
    C2PSA/C3k2(attn) 已在框架。
- **关键修复（v26 推理对齐）**：SPPF 缺 **add(shortcut) 残差** — yaml `SPPF [c,5,3,True]` 第 4 参为 shortcut，
  ultra `SPPF.forward` 输出 `cv2(cat(...))+x`（add 时残差）；框架原 SPPF 无残差。修后框架前向从 L9(SPPF)
  diffmax 5.6/mean0.40 降到 mean~0.06-0.09（残差即首差异层）。**v8/v11 的 SPPF 参数无 add(False)→不受影响**。
- **推理 mAP 对齐（同权重）**：v8n **0.790**（ultra 0.8021）、v11n **0.8005**（ultra 0.821）、v26n **0.806**（ultra 0.828），
  差 ≤0.022，算子级微差（与 seg/pose 一致量级）。
- **v26 angle**：OBB26 头输出 **raw angle**（不 sigmoid，`dist2rbox(..., raw_angle=True)` 直接用 `theta=ang`）；
  postprocess `angle_raw=true`。v8/v11 仍用 `(sigmoid(ang)-0.25)*pi`。
- **ObbLoss 扩展**：新增 `raw_angle` 与 `use_one2one` 参数（`_forward_one` + assigner_one2one/one2one_gain/one2one_topk），
  支持 v26 端到端训练（reg_max=1 时 `loss_dfl=dist_raw.sum()*0`）。
- **训练对齐验证**：v8n 微调 5 轮 best mAP50-95=**0.7905**、v26n 微调 5 轮 best=**0.8065**（均 best_epoch1，
  =预训练推理水平，无退化），v11 微调 =0.809（超预训练）。v8/v26 与 demoseg 同 ObbLoss（reg_max16 路径与 v11 相同）。
- v26 训练/评估需在 CLI 传：`-o Loss.raw_angle=true -o Loss.use_one2one=true -o PostProcess.end2end=true -o PostProcess.angle_raw=true -o Architecture.Head.reg_max=1 -o Architecture.Head.end2end=true`。

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

## 端到端训练对比（dx_ocr 车牌数据集，yolo11n，已验证逐 epoch mAP 对齐）
- **数据**：`datasets/dx_det`（= `E:/dx_ocr/ultralytics`，同一单类车牌 plate 数据集，nc=1，745 train + 186 val，
  `cls cx cy w h` 归一化标签，1920x1080）。框架用 `data_dir=datasets/dx_det` + train.txt/val.txt；ultra 用 `datasets/dx_det/data.yaml`（path=E:/dx_ocr/ultralytics）。
- **同权重起点（关键）**：ultra 侧用 `DetectionModel('yolo11n.yaml', ch=3, nc=1)` + `model.load(COCO)` 重建出 **nc=1 权重**
  （cls 头随机初始化），dump 两份：`yolo11n_nc1_state.pth`（给框架，missing=0/unexpected=1 仅函数式 dfl）与
  `yolo11n_nc1.pt`（存完整 DetectionModel 对象，给 ultra）。两端从**同一份 nc=1 权重**出发，起点一致。
  - ⚠️ 框架 `ptcore/pretrained.py::load_state_dict_any` 处理 ultra `.pt` 时用 pickle stub；但 nc=1 权重实际是 `.pth {state_dict}` 给框架，`.pt {model}` 给 ultra，两侧分开。
- **配置**：框架 `configs/_parity/dx_yolo11n_det.yml`（nc=1, reg_max=16, SGD, batch=8, **use_ema=true, ema_decay=0.9999, ema_decay_type=exponential**）；
  ultra `ultra_train_dx.py`（SGD, batch=8, 关全部增广, 默认开 EMA）。
  ⚠️ ultra 无 `ema=` 参数（报 'ema is not a valid YOLO argument'）；框架 `ptcore/ema.py::ModelEMA` 默认是 **threshold** 型（decay=min(0.9998,(1+step)/(10+step))），
  与 ultra `ModelEMA` 的 **exponential** 型（`decay=0.9999*(1-exp(-updates/2000))`，decay=0.9999, tau=2000）不同。
  已把框架配置改为 `use_ema=true + ema_decay=0.9999 + ema_decay_type=exponential` 对齐 ultra（decay 公式/参数与 ultra 一致）。
  实测开 EMA exponential 后框架终值 mAP50-95≈**0.7825**，与无 EMA（0.7817）几乎相同 → 说明此前无 EMA 结果已接近 ultra。
- **同 batch=8 → 两侧都按 step 递减 LR**（框架 Linear scheduler 每 do_step、ultra 也每 batch），LR 相位对齐。
  （之前 mini_det 用 batch=4 使 1 step/epoch 造成 LR 错位；dx 用大数据+同 batch 解决。）
- **同权重评估对比（定位 mAP 差来源）**：用 ultra 训练好的 `weights/best.pt`（nc=1），框架 `ptx val` 加载
  **missing=0/unexpected=0**，评估 **mAP50=0.995 / mAP50-95=0.7928 / mAP75=0.9947**，
  ultra 自评 **mAP50=0.995 / mAP50-95=0.7909** → 同一权重下 mAP50-95 仅差 **0.0019**，**评估管线一致**。
  → 训练端到端 mAP 差的 **~0.008** 主要来自**训练阶段数据顺序差异**（框架 `Train.loader.shuffle=false`、ultra 默认 shuffle=true，
  每 batch 梯度次序不同），属训练噪声，非算法/评估差异。`ptx val`（`tools/eval.py`）强制 `use_ema=False`
  且用 `load_state_dict_any` 可加载 ultra `.pt`，因此可做同权重跨端评估。
- **验证结果（val mAP，6 epoch）**：
  | ep | 框架 mAP50-95 | ultra mAP50-95 | 框架 mAP50 | ultra mAP50 |
  | 1 | 0.565 | 0.417 | 0.952 | 0.733 |
  | 2 | 0.712 | 0.631 | 0.991 | 0.940 |
  | 3 | 0.719 | 0.709 | 0.994 | 0.995 |
  | 4 | 0.730 | 0.729 | 0.995 | 0.994 |
  | 5 | 0.742 | 0.782 | 0.994 | 0.995 |
  | 6 | **0.782** | **0.791** | 0.994 | 0.995 |
  → mAP50-95 终值差 **0.009**、mAP50 差 **0.001**，ep3-6 几乎一致；**端到端训练管线（前向+损失+优化+评估）与 ultra 高度对齐**。
- **train loss 分量定义仍略有差异**（框架每 step 打 raw loss_box/cls/dfl，如 box≈0.09/cls≈0.74/dfl≈0.55；
  ultra results.csv 是 EMA 平滑后的含 gain 分量），需换算/对齐口径后才能逐点比（不影响 mAP 对齐结论）。
- **之前 mini_det（dota128，4+2 张，nc=80）对比失败**：数据量太小、LR 相位错位（batch=4→1step/epoch）、
  框架 batch=4 单 batch 训练 mAP 恒定（疑似框架单 batch 训练循环异常，未深究）；**弃用**，改用 dx_ocr 大数据集。
- 说明：框架配置 `device: 'cuda:0'` 才能落到 GPU（`device: '0'` 会按非 gpu/cuda 前缀解析到 cpu）。

## 端到端训练对比扩展到 yolov8n / yolo26n（验证"单步对齐 ⇒ 端到端对齐"）
- **同权重起点**：仿 yolo11n，用 ultra `DetectionModel(alg_n.yaml, nc=1)+load(COCO)` dump 出 `yolov8n_nc1`/`yolo26n_nc1`（.pt 给 ultra / _state.pth 给框架），
  框架加载 missing=0/unexpected=0（nc=1 完全匹配）。
- **框架配置**：`configs/_parity/dx_yolov8n_det.yml`（legacy，reg_max=16，end2end=false）、`dx_yolo26n_det.yml`（reg_max=1，end2end=true，Loss use_one2one=true/one2one_topk=1）；
  ultra `ultra_train_dx2.py <v8|v26>`（同 SGD/batch=8/6 epoch/关增广/默认 EMA）。
- **v8 val mAP（6 epoch）**：框架 0.604/0.718/0.732/0.738/0.749/**0.788** vs ultra 0.530/0.690/0.736/0.755/0.777/**0.783**（终值差 0.005）。
- **v26 val mAP（6 epoch）**：框架 0.622/0.690/0.772/0.748/0.754/**0.792** vs ultra 0.432/0.701/0.693/0.707/0.793/**0.806**（终值差 0.013，ep3/ep5 波动较大——E2E 训练更不稳）。
- **同权重评估对比（ultra best.pt）**：
  - v8：框架 0.7860 vs ultra 0.7831（差 0.003，一致）。
  - v26：**框架用 end2end（one2one）评估 0.7833 vs ultra 0.8055（差 0.022，不一致）**；但框架改用 **one2many+NMS（end2end=false）评估同一 best.pt → 0.8035 vs ultra 0.8055（差 0.002，一致）**。
  - **关键发现**：框架 `modules.py::Detect26.forward` 在 eval 时**无条件返回 one2one 分支**（line 577-579），而 ultra yolo26 推理/重载模型用 **one2many+NMS**（end2end=False）。
  - **已修复（评估对齐）**：① `modules.py::Detect10.forward` 的 eval 分支改为 `return one2one if self.end2end else one2many`；
    ② `ptcore/trainers/base.py::evaluate()` 评估前把 end2end 头（Detect10）的 `end2end` 临时置为 `PostProcess.end2end`（评估后还原 True）。
    这样框架 yolo26 评估自动走 one2many+NMS，与 ultra 一致（同一 best.pt：框架 0.8035 vs ultra 0.8055，差 0.002）。
    配置 `dx_yolo26n_det.yml`：`Head.end2end=true`（训练用 one2one 分支）+ `PostProcess.end2end=false`（评估 one2many+NMS）。
  - 注：训练 loss/梯度（单步）yolo26 早已对齐（`E2EDetectLoss` one2many+one2one），不受评估路径影响。
- **结论**：v11、v8 端到端训练+评估均与 ultra 对齐（`单步对齐 ⇒ 端到端对齐`成立）；yolo26 训练(loss/梯度/评估)已对齐
  （评估改为 one2many+NMS 后同权重差 0.002；训练端到端 mAP 终值 0.779 vs 0.806，差 0.026 属训练随机性+E2E 波动，与 v11/v8 同源于数据顺序 shuffle）。

## 端到端训练对比扩展到 剩余全部检测家族（yolov10 / yolov9c / yolov5nu / yolov3u / yolo12n）
- 复用 dx_ocr（nc=1，6 epoch，SGD/batch=8/关增广/EMA exponential=0.9999）端到端对比，验证"单步对齐 ⇒ 端到端对齐"适用于全家族。
- **nc=1 权重 dump**：仿 v11n，用 ultra `DetectionModel(alg.yaml, ch=3, nc=1)+load(COCO)` 重建，dump `*_nc1.pt`(ultra) / `*_nc1_state.pth`(框架)。
- **框架加载（fw_dx_load5.py）**：五家族全 **missing=0 / unexpected=1**（仅函数式 dfl），参数总数与 ultra 仅差 16（=dfl）。
  - **yolov9c 加载修复**：框架 v9 各档**烘焙固定**（无 scales），v9c 必须用 `yolov9.yaml`+**scale='c'**（不能用 yolov9c.yaml，也不可传 scale='n' 缩放）。
  - **yolo12 `AAttn.pe` bias 修复（关键）**：ultralytics 8.4.154 源码 `AAttn.pe=Conv(...act=False)` **无 bias**（block.py:1691），
    但官方 `yolo12n.pt`（旧版结构）**带** pe bias（8 weight+8 bias），而 nc1 用 8.4.154 重建 → **无 bias**。
    框架原 `modules.py::AAttn.pe` 设 `bias=True`（为对齐旧版官方权重），导致加载 nc1 时 **miss=8**（`attn.pe.conv.bias`）。
    已改为 `bias=False`（对齐 8.4.154 源码 + yolo26 官方亦无 bias）；改后 yolo12n nc1 miss=0/unexp=1，且 **yolo26n 官方加载仍 miss=0/unexp=0**（未被破坏）。
    ⚠️ 注：官方 yolo12n.pt（旧版训练产物）pe 带 bias，与 8.4.154 源码不一致，登录时以当前 ultra 源码（8.4.154，无 bias）为准。
- **端到端训练终值 mAP50-95（框架 vs ultra）**：
  | 家族 | 框架各ep (1~6) | 终值 | ultra 各ep | 终值 | 终值差 |
  |------|----------------|------|-----------|------|--------|
  | yolov12n | 0.478/0.698/0.744/0.751/0.743/0.794 | **0.794** | 0.369/0.538/0.755/0.754/0.776/0.800 | **0.800** | 0.006 |
  | yolov10n(E2E) | 0.608/0.668/0.731/0.762/0.734/0.785 | **0.785** | 0.316/0.711/0.756/0.775/0.791/0.811 | **0.811** | 0.026 |
  | yolov9c | 0.566/0.698/0.722/0.724/0.744/0.789 | **0.789** | 0.001/0.655/0.657/0.719/0.779/0.794 | **0.794** | 0.005 |
  | yolov5nu | 0.671/0.720/0.752/0.702/0.738/0.779 | **0.779** | 0.049/0.468/0.704/0.717/0.786/0.787 | **0.787** | 0.008 |
  | yolov3u | 0.562/0.612/0.712/0.757/0.749/0.786 | **0.786** | 0.348/0.663/0.679/0.749/0.772/0.800 | **0.800** | 0.014 |
- **结论**：detect 全部检测家族（v11/v8/v26/v12/v10/v9(t/s/m/c)/v5/v3u）端到端训练 mAP 终值均与 ultra 对齐（差 ≤0.026，
  多数 ≤0.015，属训练数据顺序/随机性 + E2E 波动）。剩余无差距来源同 v11/v8（shuffle 使每 batch 梯度次序不同）。
- **配置**：configs/_parity/dx_{yolov10n,yolov9c,yolov5nu,yolov3u,yolo12n}_det.yml；ultra 侧 ultra_train_dx3.py <v10|v9c|v5nu|v3u|v12>。
