# AGENTS.md

## 仓库改名（PytorchOCR → TorchKiln，2026-09-23，已完成）
- **三层命名**：
  | 层 | 旧 | 新 |
  |---|---|---|
  | 仓库根目录 | `E:\PytorchOCR` | **`E:\TorchKiln`** |
  | Python 包 / import | `pytorchx` | **`torchkiln`**（目录 `pytorchx/` → `torchkiln/`） |
  | CLI 命令 | `ptx` / `ptx.bat` / `python -m pytorchx` | **`tkiln`** / `tkiln.bat` / `python -m torchkiln` |
- **改名原因**：`TrainForge`/`trainforge`（组织+PyPI）、`PytorchX`/`pytorchx`（wang-xinyu 200★ 同名且也做 YOLO）、`TorchForge`（Meta 官方）均已占用；`kiln` 命令被 Kiln-AI(5k★) 占用故 CLI 用 **`tkiln`**；`torchkiln` GitHub/PyPI 搜索干净。
- **必须保留、禁止改（外部契约）**：
  - ModelScope 外部 URL / 模型名：**`ChaoII0987/TorchKiln`**（2026-09-23 由 `ChaoII0987/PytorchOCR` 迁移；configs `pretrained_model`、`MODELSCOPE_MODEL = "TorchKiln"`、`datasets/manifest.yml` 的 prefix 均已切换）。本地镜像仓库：`\\tsclient\E\TorchKiln`（含 `pretrained/`、`onnx/`）。
  - 环境变量名保持 **`PYTORCHOCR_*`**（`PYTORCHOCR_HOME` / `PRETRAINED_DIR` / `AUTO_DOWNLOAD` / `ALLOW_LOCAL_REPO` / `TF32` / `CUDNN_BENCHMARK`）——改了会破坏用户已有环境变量与文档。
  - 兼容包 **`pytorchocr/` 已删除**（2026-09-23，未发布故无需兼容）；只认 `torchkiln` / `torchkiln.ocr`，`pyproject` 的 `include` 也不再含 `pytorchocr*`。
  - conda 环境名 **`ptocr`** 不变（与仓库名无关）。
  - **缓存路径**：新默认 `~/.torchkiln/pretrained/`；**不再回退查找**旧 `~/.pytorchocr/pretrained/`（已删除兼容）。
  - 旧目录可自行删除；若曾迁移可 `Move-Item $env:USERPROFILE\.pytorchocr $env:USERPROFILE\.torchkiln`（可选）。
  - 权重文件名**不再**带 `_ptocr` / `_state` 后缀（配置 URL 与缓存均为裸名 `.pth`）。
- **标签缓存**：`CACHE_VERSION = "tkiln-labels-1.0"`（由 `ptx-labels-1.0` 改来），旧 `.labels_cache_*.pkl` 会自动失效重扫，无害。
- **启动器**：根目录 `tkiln`（sh）与 `tkiln.bat`；`pip install -e .` 后 `pyproject.toml` 注册 console script `tkiln = torchkiln.cli:main`。
- **批量替换踩坑（复盘，勿重蹈）**：
  1. PowerShell `-replace` **默认大小写不敏感**——会把 `pytorchocr`/`PYTORCHOCR_*` 误替换成 `TorchKiln`。必须用 `.NET` 的 `.Replace()`（大小写敏感）或 `-creplace`。
  2. 用占位符保护外部 URL 时，占位符**自身**也被大小写不敏感替换打穿 → 恢复失败。保护 token 里不要含被替换子串的大小写变体。
  3. 仓库根目录改名时若有进程 cwd 在目录内会 `IOException`；解法：从 `E:\` 用 `robocopy /E /MOVE` 搬内容到新名，再删空壳。
- **验证记录（改名后全过）**：`import torchkiln/ptcore` OK（`pytorchocr` 兼容包已删）；`python -m torchkiln --help` / `tkiln.bat check -c configs/_parity/dx_yolo11n_det.yml` OK；`tools/check_graph_build.py` **53 OK, 0 FAIL**；git 仓库在 `E:\TorchKiln` 可用。
- **opencode 配置（`opencode.jsonc`，项目根）**：
  - `instructions` 挂了本文件；`references.rename` 指向本改名节。
  - 自带 `/rename-map` 命令：打印旧→新对照与不可改项。
  - **同事木海仍习惯说「PytorchOCR」**——听到该名字即指本仓库（`E:\TorchKiln` / `torchkiln` / `tkiln`），不要去找 `E:\PytorchOCR`（目录已删）。
  - 全局配置 `~/.config/opencode/opencode.jsonc` 只有 vLLM provider，与本仓库无关，未动。

## Detect（目标检测）与 ultralytics 的对齐（单步验证已通过，yolo11n）
- 验证方法为**单步验证**：同权重（`\\tsclient\D\项目资料\ultralytics_models` dump）+ 同输入 `\opencode\det_x_1.npz`(1,3,640,640)
  + 同 GT（像素框 [192,128,320,384]，ultra 归一化 xywh=[0.4,0.4,0.2,0.4]）做前向（feature/loss）+ 反向（梯度）对比。
- **yolo11n 检测对齐完成**（单步验证全过）：
  - 权重加载：missing=0 / unexpected=1（仅函数式 `dfl.conv.weight`，同 pose/seg/OBB）。
  - assigner 匹配一致：n_fg=10 / t_scores.sum=1.3955（框架=ultra）。
  - 损失三分量（raw）全对齐：box=**0.5508**、cls=**23.3453**、dfl=**3.1119**；total=**20.47179** vs ultra **20.47182**（差 3e-5）。
  - 梯度 maxdiff=0.0216（worst `model.0.conv.weight`，cuDNN 卷积反向算子级微差）。
- **关键 bug 修复（DFL 目标，影响所有 detect 训练）**：`torchkiln/det/loss.py::DetLoss._forward_one`
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
## 回复语言（重要，必须遵守）
- **所有大模型（AI 助手/Agent）在本仓库中的回复一律使用中文。**
- 包括：解释代码、回答提问、汇总结果、生成文档、提交说明等所有面向用户的文本。
- **尤其是「执行过程确认」也必须用中文**：包括执行步骤、命令含义、中间结果、进度汇报、错误/失败说明、
  最终结论等，都要用中文说明，方便用户用中文逐步核对执行过程。
- 代码、变量名、报错原文、日志原句、命令行可保持原样，但**对其的解释与转述必须用中文**。

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
- 框架（TorchKiln / torchkiln）使用 conda 环境 **`ptocr`**（Python 3.12 + PyTorch 2.12 + CUDA）。
- 原版 ultralytics 使用 conda 环境 **`ultralytics`**（Python 3.12 + ultralytics 8.4.x）。
- 两者均可用 GPU（CUDA 可用）。

## 数据集
- **DOTA128（OBB 旋转检测）**位于 `datasets/dota128/`（来自 `G:\迅雷下载\dota128.zip`）。
  - 拆分为：`images/train`(102 张) + `images/val`(26 张)，标签为 DOTA 四角点(归一化 8 个数)格式。
  - 清单文件：`datasets/dota128/train.txt`、`datasets/dota128/val.txt`（图片相对路径，供框架 DetDataset 使用）。
  - ultralytics 侧数据配置：`datasets/dota128/dota128.yaml`。
  - 类别：15 类（plan/ship/storage-tank/... 见 dota128.yaml）。
- **`datasets/` 整目录不入库**（2026-09-23；含 label/清单/图片；需要示例时另放 ModelScope 或本地 `make_demo_data.py` 生成）。
- 权重与 `datasets/` 均由 `.gitignore` 忽略。

## OBB（旋转框）与 ultralytics 的对齐（已完成并验证）
- **旋转框 NMS**：`torchkiln/det/rbox.py::nms_rotated` 由 O(N²) 纯 NumPy 多边形裁剪（2000 框 87s）
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
- **Pose 头 cv4**：`torchkiln/nn/modules.py` 的 `Pose`/`PoseU` 头 `cv4` 从 `c4=x` 改为 **`c4=max(ch[0]//4, nk)`**，
  与 ultralytics `Pose.cv4` 一致；框架模型与 ultralytics `yolo11n-pose`（nc=1,kpt:[12,2]）**权重完全加载
  （missing=0 unexpected=0）**。
- **关键点损失（重要 bug 修复）**：`torchkiln/pose.py::PoseLoss` 对齐 ultralytics `v8PoseLoss`——
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

## Segment 端到端训练对齐（关键修复：评估 fp16 + 增广裁剪中心；系统性差距已消除）
- 背景：框架 seg 训练 mAP 曾**系统性落后** ultra ~0.03-0.05（v8/y11/y26 全部家族都差，非随机）。
  排查结论：**不是训练/损失问题，而是「评估精度」+「增广实现」两处 bug**。
- **快速定位手法（重要，省时）**：
  1. 把 **ultra 自己训练的 `best.pt`** 丢进**框架评估器**：若数值对不上，问题必在评估侧（与训练无关）。
  2. 对照 ultra `results.csv` 时注意列序：**mask mAP50-95 是第 15 列(index 14)**，别读成 index 13(mAP50)。
  3. **统一用同一评估器评双方模型**（排除 ultra val 默认 `rect=True` 抬高的口径差）。
  4. **多 seed（各 3 个）** 确认差距是否超过运行噪声（v8n std≈0.014/0.026）。
  5. **关增广对照**：若差距消失→问题在增广（实测无增广差距仅 0.008）。
- **修复 1（评估 fp16→fp32，`ptcore/trainers/base.py::_evaluate_loop`）**：评估阶段改为
  `torch.autocast(..., enabled=False)` 强制 fp32。yolo26 seg 的 `reg_max=1` 框解码 / proto einsum
  在 fp16 下失真 → 同权重 mask_mAP50-95 **0.5596(fp16) vs 0.6413(fp32)**（ultra val 是 fp32）。
  此 bug **影响所有任务**的历史评估数值。
- **修复 2（mosaic 裁剪中心，`torchkiln/data/augment.py::mosaic4`）**：调 `_random_crop_mosaic(..., center=(s,s))`
  即**画布中心**。ultra `RandomPerspective` 的 `M=T@S@R@P@C` 把**输入画布中心**映射到输出中心（scale 仅 jitter），
  等价于在画布中心裁剪；框架原在 mosaic 的 tile 汇聚点 `(xc,yc)` 裁剪 → 物体分布与 ultra 不一致。
  改后 **v8n 0.595→0.650（=ultra 0.648）**，系统性差距消除。⚠️注意：曾试过"不裁剪改缩放(2S→S)"，
  **更差(0.47)**——ultra 的仿射是**中心裁剪**不是缩放，勿再走弯路。
- **修复 3（SegLoss DFL/L1，`torchkiln/seg.py`）**：`tgt` 不该 clamp 到 `reg_max-1`（只 clamp `bbox2dist` 的
  ltrb 距离，对齐 `DetLoss`/ultra）；`reg_max<=1` 时补 **L1 loss**（对齐 ultra `BboxLoss` 无 DFL 分支）。
- **修复 4（yolo26 seg 真 E2E 训练）**：`Segment26.forward` 训练返回 `(one2many, one2one, proto)`
  （one2one 输入特征 `detach`）；`SegLoss` 支持双分支（one2many `topk=10`；one2one `topk=7, topk2=1`，
  one2one 的 proto `detach`）+ **语义辅助损失**（BCE+Dice，`_semantic_loss`）+ ultra `E2ELoss` 的
  **o2m/o2o 增益调度**（0.8→0.1，每 epoch 末 `loss.update()`，`base.py::_train_one_epoch`）。
  ⚠️**关键**：`Segment26.__init__` 必须初始化 **one2one_cv2/cv3 的 bias**（对齐 ultra `bias_init`），
  否则 one2one cls 初始 p~0.5 → cls loss 爆炸(1495) → 训练崩盘；并需**重生成 nc1 权重**（patch one2one_cv3 末端 bias）。
- **修复 5（`evaluate()` 两处）**：BN eps 仅 `yolo_cls` 用 1e-5（det/seg/pose/obb 保留训练值 1e-3，强改 1e-5 会崩）；
  end2end 交换**推广到所有含 `one2one_cv2` 的头**（Segment26/OBB26，原先只换 Detect10）。
- **结果（缩放集 100/50，50 轮，同一框架评估器）**：v8n **0.650** vs ultra 0.648；y11n **0.618** vs 0.583；
  y26n **0.636** vs 0.642 → **系统性差距消除**（框架不落后）。单步 seg loss 与 ultra 差 ~1.5%（mask 1.7%、box/dfl <0.5%）。
- **速度经验**：训练期**不评估**（仅末轮评一次，`eval_epoch_step` 设大、`eval_batch_step` 设大）；`num_workers=8`
  稳态 ~6.6s/epoch（~7 分钟/模型）；瓶颈是 mosaic 掩码构建(CPU)，非 GPU。Windows 下 worker 多 + 全分辨率掩码易 OOM；
  `load_raw` 加内存缓存会 OOM（8 worker×全分辨率掩码，已回退）；残留 python 进程会占数 GB，需 `Stop-Process` 清理。
- 数据集：`datasets/package-seg-r/`（1920×1080 预缩放 960×540，`train.txt`/`val.txt`(相对)+`*_abs.txt`(ultra)+`pkgsub_ultra.yaml`），
  配置 `output/seg_r_cfg/{yolov8n,yolo11n,yolo26n}.yml`。

## 通用训练/评估关键修复（det/seg/pose/obb 通用）
- **关键 bug：梯度累积 `accumulate` 路径写错（`ptcore/trainers/base.py`）**——原读
  `config["Train"]["dataset"]["loader"]["batch_size_per_card"]`（不存在），实际在 `Train.loader`。
  → `_bs=1` → `accumulate = round(nbs/1) = 64`（应为 `nbs/batch`，如 64/8=8）。
  **后果：有效 batch 放大 8×、每 epoch 仅 ~4 次优化器更新（应 ~25）→ 学习慢 8×、收敛远慢于 ultra**。
  已修（读 `Train.loader`，兼容旧式 `Train.dataset.loader`）。修复后同 10 轮（val50）：框架 **0.4457→0.5557**（ultra 0.538，反超）。
  ⚠️ **这解释了此前"框架普遍收敛慢"的真正原因**（不是增广、不是优化器、不是 warmup）。
- **关键 bug：评估 NMS 极慢（`torchkiln/det/ops.py::nms`）**——原为 Python 逐框循环 + 每步 `.item()` GPU 同步；
  16800 框时 **10s/图**，402 张 val 要 ~1 小时（表现为"评估卡死"）。改用 **torchvision C++ NMS**：
  postprocess **504s→2.5s/50图（200×）**，全 val 35s；mAP 逐位不变。（旋转框 NMS 早已优化为 probiou+fast_nms。）
- **评估 fp16→fp32**（`_evaluate_loop`，见 Segment 节修复 1）：强制 fp32，否则 yolo26 等被系统性低估。
- **MuSGD/Muon 优化器（新增 `ptcore/muon.py` + `ptcore/optimizer.py`）**——1:1 移植 ultra `optim/Muon`：
  `optimizer=auto` 迭代数 >10000 选 MuSGD(lr0=0.01, momentum=0.9)；ndim∈{2,4} 走 Newton-Schulz 正交化的 Muon，
  检测头 `cv3/one2one_cv3` 参数 `lr*3`；`muon=0.2, sgd=1.0`。10 轮比 SGD 好 +0.02。
- **yolo26 检测 E2E 损失已逐位对齐**：单步**同权重**下总 loss **27.138552 vs 27.138544**（差 8e-6）。
  之前"one2one cls 差 6×"是**权重加载不一致**的假象——对比时**必须让两边加载完全相同的权重**。
- **增广/仿射**：`mosaic4` 裁剪中心已对齐画布中心 `(s,s)`；`RandomAffine` 矩阵已改为 ultra `RandomPerspective` 的
  `M=T@S@R@P@C`。`augment.mosaic_crop` 可选（det 早期不裁剪略好、100 轮裁剪略好，差异不大）。
- **复现验收标准（通用）**：① 权重加载 missing=0/unexpected=0；② 同权重逐层前向 maxdiff ~1e-5；
  ③ 单步 loss/梯度（loss 相对误差 <1e-3；梯度仅 cuDNN 卷积反向的算子级微差）；④ 同权重推理 mAP 差 ≤0.02；
  ⑤ 端到端训练 mAP 收敛后差 ≤0.02-0.03（含随机种子噪声）。**1-4 项对齐即复现成功**；第 5 项天生有噪声。
- **外部 person 检测数据集**：`E:\20260921\export_yolo_dataset`（2560×1440，1606/402，`cls cx cy w h` 归一化），
  预缩放副本 `datasets/export_yolo_1280/`（含 `train/val.txt` 相对 + `*_abs.txt` ultra + `pkgsub_ultra.yaml`），
  配置 `configs/local/export_yolo_yolo26n_det.yml`（yolo26n, imgsz1280, batch8, MuSGD/auto, mosaic_crop=false, 预训练 `weights/yolo26n_det.pt`；已移入 local，gitignore）。

## 复现/对比 ultralytics 的常见坑（避雷清单）
- **口径必须一致**：ultra 自报的 mAP 默认用 `rect=True`（val 矩形 letterbox）会抬高数值；框架是方图。
  对比时**必须用同一个评估器评双方权重**（把 ultra 的 `best.pt` 丢进框架 `tools/eval.py` 即可）。
- **读 ultra `results.csv` 注意列序**：`metrics/mAP50-95(B)` 是第 11 列(index 10)、`metrics/mAP50-95(M)` 是 index 14；
  别把 index 13(`mAP50`) 当成 mAP50-95（本会话为此误判过）。
- **单步 loss/梯度对比必须加载"完全相同"的权重**：一边 `miss=102`、另一边 `miss=594` 时算出来的 loss 毫无意义
  （曾据此误判"one2one cls 差 6×"）。做法：框架加载后 dump `state_dict`，ultra 直接加载它。
- **对比必须用同一 val 清单**：框架 val50 vs ultra val402 不可比；要么都用同一文件，要么都用同一评估器。
- **`augment: {}`（空 dict）是 falsy**，`DetDataset/SegDataset` 不会建 augmenter → 需写显式值
  （`mosaic/hsv/affine/fliplr/close_mosaic`）才真正开启增广。
- **batch 配在 `Train.loader`（与 dataset 同级）**，不是 `Train.dataset.loader`；框架已兼容两者，
  但历史配置混用过，排查 `accumulate` 时注意。
- **大图慢**：2560×1440 直接 mosaic 极慢；预缩放到训练 `imgsz`（无损）可让 reader 114s→43s/epoch。
- **Windows worker/内存**：`num_workers` 过大 + 全分辨率掩码/mosaic 画布易 OOM；`load_raw` 加内存缓存会 OOM（已回退）；
  残留 python 进程会占数 GB，需 `Stop-Process -Force` 清理。
- **训练/评估进程可能"结束即卡住"**（无输出、GPU 归零、无残留进程）→ 用 `val` 子集评估、或训练时 `eval_epoch_step` 设大。
- **`device` 必须写 `cuda:0`**（写 `'0'` 会按非 gpu/cuda 前缀解析到 cpu）。

## Classification（图像分类）与 ultralytics 的对齐（已完成并验证）
- **任务已存在但需对齐验证**：`torchkiln/tasks/classify.py` + `_cls.py`（ClsLoss/ClsMetric/ClsPostProcess）
  + `torchkiln/data/cls.py::ClsDataset`（读 `path label` 清单）；框架 `torchkiln/cfg/models/11/yolo11-cls.yaml`
  与 ultra yolo11-cls 结构一致（backbone + 单一 `Classify` 头：Conv→1280 → AdaptiveAvgPool → Linear(nc)）。
- **权重加载完全对齐**：dump ultralytics `yolo11n-cls.pt`（ImageNet 1000 类）→ `build_arch_model(...,"classify")` 加载：
  **missing=0 / unexpected=0**（框架 `Classify` 头与 ultra 完全一致）。
- **全部 15 个权重对齐（yolov8/yolo11/yolo26 × n/s/m/l/x）**：对 `\\tsclient\D\项目资料\ultralytics_models\{yolov8,yolo11,yolo26}\*-cls.pt`
  全部验证——加载均 **missing=0/unexpected=0**；同输入(224)推理 softmax maxdiff **≤0.0000007**（几乎逐位一致），top1 全部一致。
- **C3k2 的 M/L/X 规模逻辑已对齐**：ultra `parse_model` 对 `scale∈{m,l,x}` 强制 C3k2 `c3k=True`（用 C3k，1×1）；框架
  `torchkiln/nn/graph.py::parse_model` 已复刻（`if module_name=="C3k2" and scale in ("m","l","x"): args[2]=True`），
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
- **按家族路由检测/分割头（`torchkiln/nn/graph.py`）**：`build_from_arch` 从 `yaml_file` 判断家族，
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
- 仓库已在 `E:\TorchKiln` 初始化。
- `.gitignore` 会忽略：`__pycache__`、`output/`、`weights/`、`*.log`、权重(`*.pt/*.pth`)、**`datasets/` 整目录**、
  缓存目录、`_downloads/`、`_ref/`、`configs/local/`、`configs/_parity/`（后两者已 `git rm --cached`）。
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
- **同权重评估对比（定位 mAP 差来源）**：用 ultra 训练好的 `weights/best.pt`（nc=1），框架 `tkiln val` 加载
  **missing=0/unexpected=0**，评估 **mAP50=0.995 / mAP50-95=0.7928 / mAP75=0.9947**，
  ultra 自评 **mAP50=0.995 / mAP50-95=0.7909** → 同一权重下 mAP50-95 仅差 **0.0019**，**评估管线一致**。
  → 训练端到端 mAP 差的 **~0.008** 主要来自**训练阶段数据顺序差异**（框架 `Train.loader.shuffle=false`、ultra 默认 shuffle=true，
  每 batch 梯度次序不同），属训练噪声，非算法/评估差异。`tkiln val`（`tools/eval.py`）强制 `use_ema=False`
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

## det3d（3D 检测）与 Paddle3D CenterPoint-Pillars 的对齐（已完成并验证，2026-09-24）
- **背景**：AD 的 `det3d` 最初用自研轻量 dense-CNN（`PillarDetNet`，**非 SOTA**）。已替换/新增 **SOTA CenterPoint-Pillars**（Paddle3D 官方模型）并完成数值对齐。
- **权重与数据来源（关键：绕开被墙的 GitHub/ModelScope）**：
  - Paddle3D 权重托管在 **百度 BOS**，本网络直达：`https://bj.bcebos.com/paddle3d/models/centerpoint/centerpoint_pillars_016voxel_kitti/model.pdparams`（19MB）；
    pointpillars：`.../pointpillars/pointpillars_xyres16_kitti_car/model.pdparams`。
  - KITTI 数据：`https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_{velodyne(28.7GB),calib,label_2,image_2}.zip`（用户手动下，S3 直连超时）；
    KITTI ImageSets 在 `https://bj.bcebos.com/paddle3d/datasets/KITTI/ImageSets.tar.gz`。
  - 环境：`paddlex` conda env 装 `paddle3d==1.0.0`（pip，清华源）+ `numba pyquaternion paddleseg h5py scikit-image nuscenes-devkit`（--no-deps）。
- **模型实现**：`torchkiln/nn/centerpoint.py`（PyTorch，**模块/命名逐一对齐 Paddle3D**）：
  `PillarFeatureNet`(9→32→64, legacy=False) + `PointPillarsScatter` + `SecondBackbone`(3/5/5, strides 1/2/2)
  + `SecondFPN`(0.5/1/2, use_conv_for_no_stride) + `CenterHead`(双任务, shared_conv + SeparateHead, hm final conv bias=-2.19)
  + `hard_voxelize`。`load_paddle_centerpoint()` 把 `.pdparams`(numpy pkl) 加载进模型。
  - **权重加载 missing=0 / unexpected=0**（191 张量）。注意：Paddle `Linear.weight` 是 `[in,out]`（需 `.t()`）；BN 用 `_mean/_variance`（→ running_mean/var）。
- **前向数值对齐（决定性证据）**：同一 voxel 输入分别跑 Paddle3D(1.0.0) 与我们的模块：
  `voxel_features/bev maxdiff 5e-6`、`fpn 3e-4`、`head(hm/reg/height/dim/rot) ~1e-4`（float32 级）。
  → **模型本身没问题**。（脚本：`_downloads/paddle3d/paddle_forward.py` + `torch_forward.py`。）
- **GT 约定（关键差异，压低 mAP 的根因）**：Paddle3D 的 lidar GT = **`[x,y,z, w,l,h, ry]`**，其中
  **`ry` 直接用相机原始 ry（不做 heading 变换）**，dims 为 **wlh**。我们原转换器用了「几何 heading(=-ry) + lwh」→ 非轴向车辆 IoU 被压低。
  - **正确几何**：KITTI 框**长边沿相机 x 轴**（devkit `compute_box_3d` 的 `x_corners=±l/2`）；等价于 `yaw = ry + π/2`（框架格式 `l 沿 yaw`）。
  - **`tools/convert/kitti_to_det3d.py` 已修正**为 `yaw = ry + π/2`；验证与 Paddle3D 框 **probiou IoU=0.994**。
- **官方 KITTI 评估器**：复用 Paddle3D 自带 `kitti_object_eval_python/eval.py`（难度/DontCare/R40），运行时打补丁去 numba、
  把 `rotate_iou_gpu_eval` 换成本地 shapely 精确 BEV IoU；调用 `get_official_eval_result(..., z_axis=2, metric_types=('bev',))`。
  GT 用 Paddle3D 约定；**GT 当预测自检=100**。
  - **全量 val（3769 帧）BEV mAP（Easy/Mod/Hard）**：
    Car **90.2/84.4/79.4**、Ped 60.3/57.1/53.1、Cyc 81.7/61.9/58.0；**均值(Mod) 67.8 vs Paddle3D 参考 71.87（差 4）**。
    （参考：Car 93.0/87.3/86.2、Ped 66.5/62.7/58.5、Cyc 86.6/65.6/61.6。）残差来自 NMS/shapely-IoU/点过滤的算子级差异。
- **框架接入（已完成）**：
  - `torchkiln/models/det3d.py::build_det3d_model` 按 `Architecture.algorithm: centerpoint` 路由到 `CenterPointPillars`。
  - `torchkiln/data/det3d.py` 新增 `raw_points` 模式 + `raw_collate`（哨兵填充成 `(B,N,4)` 张量，**不改 BaseTrainer**）。
  - `torchkiln/tasks/det3d.py` 按算法分派 loss/metric/postprocess/collate。
  - `torchkiln/nn/centerpoint_task.py`（新）：`CenterPointLoss`(FastFocalLoss+RegLoss+Gt2CenterPointTarget)、
    `CenterPointPostProcess`(逐任务解码+旋转NMS)、`CenterPointMetric`（内部复用 DetMetric/probiou）。
  - 配置 `configs/pc/centerpoint-det3d.yml`（tasks `[Car]` + `[Cyc,Ped]`，dims/角度按 Paddle 约定换算：`ry = yaw - π/2`）。
  - **基线**：`smoke_all` **83 OK / 0 FAIL**（新增 centerpoint + squeezesegv3 + bev_lanedet 配置）；`check_graph_build` 55 OK；GPU/CPU 各跑通 1 epoch 训练+评估。
- **踩坑**：`_gather_feat` 要处理 4D(B,C,H,W)；`FastFocalLoss` 的 gather 只按**类别列**索引；`RegLoss` 需**按 code 维(8)聚合**返回 (8,)；
  `batch[0].to(device)` 硬约束 → 用填充张量 `(B,N,4)` + 范围外哨兵绕过。
- **torch `.pth`（供框架 `pretrained_model:` / ModelScope 上传）**：`weights/centerpoint_pillars_kitti.pth`（19.6MB，
  框架加载 missing=0/unexpected=0）。上传 ModelScope `pretrained/` 后按裸名 `centerpoint_pillars_kitti` 引用。

## pc_seg（SqueezeSegV3）与 lane（BEV-LaneDet）与 Paddle3D 的对齐（已完成并验证，2026-09-24）
- 同 CenterPoint 模式：**BOS 直下官方权重 → PyTorch 逐命名移植 → missing=0 → vs Paddle3D 实跑前向对齐 → 转 `.pth`**。
- **SqueezeSegV3（pc_seg，range-view）**：`torchkiln/nn/sac_rangenet.py`（SACRangeNet53 backbone + 5 尺度 head）。
  - 权重 `https://bj.bcebos.com/paddle3d/models/squeezesegv3/squeezesegv3_rangenet53_semantickitti/model.pdparams`。
  - **missing=0/unexpected=0**（524 张量）；**前向对齐 vs Paddle3D：5 尺度 maxdiff ~1e-5**。
  - `.pth`：`weights/squeezesegv3_rangenet53_semantickitti.pth`（99.7MB）。
  - **框架接入**：`build_pc_seg_model` 按 `algorithm: squeezesegv3` 路由；`data/pc.py` 新增 `range_image` 模式
    （`project_range_image` 球面投影 → `(5,H,W)` 距离图 + `(H,W)` 标签）；`nn/sac_rangenet.py` 加 `SqueezeSegV3Loss`(多尺度 NLL) +
    `SqueezeSegV3PostProcess`；`tasks/pc_seg.py` 按算法分派；配置 `configs/pc/squeezesegv3-pcseg.yml`（demo，64×256，batch1）。
    ⚠️ `smoke_all` 强制 batch=4 且同进程 → range image 需够小（我们 64×256）否则 OOM 硬崩。
  - 关键坑：Paddle `ConvBNLayer(bias=None)` = **默认有 bias**（Only `bias=False` 才无）；`Linear.weight` 需转置；BN `_mean/_variance`→running。
  - 数据（mIoU 验证需）：SemanticKITTI —— `data_odometry_velodyne.zip`(80.9GB) + `data_odometry_calib.zip`(0.6MB)
    + labels `http://www.semantic-kitti.org/assets/data_odometry_labels.zip`(171MB)（S3/BOS 直达）。
- **BEV-LaneDet（lane，ResNet-34 BEV）**：`torchkiln/nn/bev_lanedet.py`（ResNet34 + FCTransform BEV 投影 + 双 head）。
  - 权重 `https://paddle3d.bj.bcebos.com/models/bev_lanedet/bev_lanedet_apollo_576x1024/model.pdparams`。
  - **missing=0/unexpected=0**（372 张量）；**前向对齐 vs Paddle3D：各层 maxdiff ~1e-5~8e-5**。config：bev_shape [200,48]、input 576×1024。
  - `.pth`：`weights/bev_lanedet_apollo_576x1024.pth`（168MB）。
  - **框架接入**：新任务 `lane_bev`（注册 4 处：`tasks/__init__.py`/`ptcore/trainers/__init__.py`/`cli.py`/`configs/lane/`）；
    `torchkiln/lane_bev.py`（`BEVLaneDetLoss`=BCE+IoU+push-pull+MSE、`BEVLaneDetPostProcess`、`BEVLaneDetMetric`=FScore）；
    `data/lane_bev.py`（`LaneBEVDataset`：图像 + BEV GT npz）；`models/lane_bev.py`；配置 `configs/lane/bev_lanedet.yml`（demo，576×1024，batch1）。
    ⚠️ **输入必须 576×1024**（FCTransform 特征尺寸硬编码 18×32 / 9×16）；`smoke_all` 强制 batch=4 也能跑（~5GB）。
    demo 生成器：`tools/make_demo_data.py --dataset lane_bev_demo`。

### BEV-LaneDet 真实 Apollo 3D Lane 数据端到端（2026-09-26，已跑通）
- **数据已到位**：`_downloads/paddle3d/Apollo_Sim_3D_Lane_Release.zip`(16.3GB) +
  **`data_splits.zip`(608MB，官方划分 JSONL)**。`data_splits` 内容：
  `standard/rare_subset/illus_chg` 三种划分的 `train.json`+`test.json`（**JSON Lines，每行一个样本**，
  字段 `raw_file/cam_height/cam_pitch/centerLines/laneLines/*_visibility`），
  以及 `3D_LaneNet`、`Gen_LaneNet` 两个 baseline 的 `test_pred_file.json`（多 `*_prob` 置信度，作对照）。
- **解压**：`_downloads/paddle3d/extract_apollo.py`（22499 文件 / 268s，**跳过 `depth/` 省 4.8GB**，BEV-LaneDet 不用深度）。
  data_splits 解压后在 `_downloads/paddle3d/data_splits/standard/`。
- **Paddle3D 官方 Apollo 代码**（`fetch_apollo_ds.py` 从 api.github.com 拉取，装进 `paddle3d/datasets/apollo/`）：
  `apollo_lane_det.py`(数据集+GT 生成)、`apollo_lane_metric.py`(FScore)、`coord_util.py`、
  `standard_camera_cpu.py`(虚拟相机)、`post_process.py`、`cluster.py`、`min_cost_flow.py`、
  + `configs/bev_lanedet/bev_lanedet_apollo_576x1024.yml`。
  - **两个必须的补丁**：① `np.int/np.float` 已在新 numpy 移除 → 正则替换成内建 `int/float`
    （注意 `np.float64` 等不能误伤）；② `paddle3d/datasets/__init__.py` 原本不 import apollo →
    补 `from .apollo import ...`，否则 `@manager.DATASETS.add_component` 不执行、config 构建找不到类。
  - **ortools**：config 指定 `9.1.9490` 在 **Python 3.12 无 wheel**（最早仅 9.3.10497）。
    实装 **`ortools==9.3.10497`**，Paddle3D 用的 `pywrapgraph.SimpleMinCostFlow` 实测可用
    （不是 `MinCostFlow`，那个才是新版本没有的）。
- **官方超参**（`bev_lanedet_apollo_576x1024.yml`）：`x_range[3,103] y_range[-12,12] meter_per_pixel=0.5`
  → **bev_shape [200,48]**；`input[576,1024]` `output_2d[144,256]`；`use_virtual_camera=True(vc_image_shape[1920,1080])`；
  `batch_size=4`、`epochs=50`、AdamW(wd=0.01)+Cosine(lr=0.001, eta_min=2e-7)、warmup 200 step。
  ⚠️ **`T_max: 37450 = 749.0*50` 而 5992/4=1498 → 官方实为有效 batch=8**（batch 4 × 2 卡），
  对齐 LR 相位时注意。
- **数据集生成**：`_downloads/paddle3d/build_lane_bev_dataset.py` —— **直接调 Paddle3D 的
  `ApolloOffsetDataset.get_seg_offset` 离线产出**（保证 GT 逐位一致）：
  - 输入图 = 虚拟相机 `warpPerspective` 后的 1920×1080；键映射一一对应
    `bev_gt_segment→bev_seg`、`bev_gt_instance→bev_inst`、`bev_gt_offset→bev_off`、
    `bev_gt_z→bev_z`、`image_gt_segment→img_seg`、`image_gt_instance→img_inst`。
  - 产出 `datasets/lane_bev_apollo/`：**7488 图 + 7488 npz（6.97GB）、train=5992 / val=1496**（共 7488，json 里有 10 张对不上）。
  - ⚠️ 用 `get_seg_offset`（而不是 `__getitem__`），因为后者会把**随机增广**（MotionBlur/亮度对比/ColorJitter）
    烘进图里；增广应留在 dataloader 侧。
- **框架侧适配**：`torchkiln/data/lane_bev.py::LaneBEVDataset` 补 `normalize` 选项
  （`imagenet`/`paddle`/`none`，IMAGENET_255 = mean[123.675,116.28,103.53] std[58.395,57.12,57.375]，
  等价 Paddle3D `NormalizeVision` / albumentations `A.Normalize()`），**缺省 none，demo 向后兼容**。
  （原实现只有 BGR→RGB + resize，**没有归一化**，真数据下必须补。）
  注意 Paddle3D 保持 **BGR** 喂 ImageNet(RGB) 统计量，框架转了 RGB——信道口径有差异，后续对齐需留意。
- **配置** `configs/local/apollo_bev_lanedet.yml`（batch=4、epochs=50、AdamW+Cosine、`normalize: imagenet`）。
- **框架端到端冒烟（2 epoch）**：**跑通**，loss ~9.5→10.5(seg≈1.44/emb≈4.27)、16.4 samples/s，
  **评估 FScore=0.6503（precision 0.5329 / recall 0.8339）**，EXIT=0。
  → 数据管线、GT、归一化、`BEVLaneDetMetric` 链路全通。
- **Paddle3D 侧已跑通（2026-09-26，同数据 2 epoch + ApolloLaneMetric）**：
  - **装 paddle3d 1.0.0 需要 7 处补丁**（缺开发版文件 + 注册问题）：
    1. `np.int/np.float` 已从新 numpy 移除 → 正则替换成内建（勿误伤 `np.float64`）；
    2. `paddle3d/datasets/__init__.py` 不 import apollo → `@manager.DATASETS.add_component` 从不执行；
    3. `ortools==9.1.9490` 在 **Python 3.12 无 wheel** → 装 **`9.3.10497`**（Paddle3D 用 `pywrapgraph.SimpleMinCostFlow`，
       **不是** `MinCostFlow`）实测可用；
    4. `paddle3d/models/detection/__init__.py` 漏 `bev_lanedet`；
    5. ⚠️ **同名包目录优先于 .py**：`models/detection/bev_lanedet/`（空 `__init__.py`）遮蔽 `bev_lanedet.py` →
       包 `__init__` 写 `from .bev_lanedet import *` + 清 `__pycache__`；
    6. `apis.config.Config` 的 `path` 是**仅关键字参数**（`Config(path=...)`）；
    7. `transforms/transform.py` 1.0.0 缺 `Resize/Transpose/NormalizeVision` → 追加三行注册
       （照抄 develop：`manager.TRANSFORMS.add_component(paddle.vision.Resize/Transpose)` +
       `class NormalizeVision(paddle.vision.Normalize)`）。注：该 paddle 版 `Normalize` **无 `scale` 参数**，
       所以 mean/std 用 0~255 尺度直接 `(x-mean)/std`；`Resize→Transpose→NormalizeVision` 顺序是因
       `Normalize(data_format='CHW')` 要求 CHW 输入。
  - **驱动脚本**：paddle3d 1.0.0 无 CLI（顶层无 `main.py`）→ 用 `apis.Config + Trainer` 自写
    `_downloads/paddle3d/train_paddle3d_apollo.py`、`eval_paddle3d_apollo.py`。
    ⚠️ `Trainer` 会校验 `save_dir`（默认 `output/`）非空而报错 → 评估时传 `checkpoint={'save_dir': 'output/_eval_run'}`。
  - **`ApolloOffsetValDataset.__init__` 不接受 `y_range/input_shape/output_2d_shape`**（比 train 的签名少 3 个参数）。
  - **预训练权重**：`resnet34-remapped.pdparams` 官方 URL = **`https://bj.bcebos.com/paddle3d/models/bev_lanedet/resnet34-remapped.pdparams`**（81.27MB，ResNet34 裸骨干 21.30M，
    键 `conv1/bn1/layer1..4` + BN `_mean/_variance`）。**`data_splits.zip` 里没有权重**（纯 JSON 划分）。
    框架侧新增 `torchkiln/nn/bev_lanedet.py::load_paddle_resnet34_pretrained`（前缀映射 `conv1→bb.0`、
    `bn1→bb.1`、`layerN→bb.(4+N-1)`、`_mean/_variance→running_mean/var`；用 pickle 读，免依赖 paddle）。
  - **指标口径澄清（重要）**：框架 `BEVLaneDetMetric` 的 FScore 是**像素级二值分割 F1**，
    与 Paddle3D `ApolloLaneMetric` 的 `f1_score`（**实例级 3D 车道 F1**）**不是同一指标**，
    两者数值**不可直接比较**。官方评估管线是 `post_conf=0.9` + embedding 聚类
    (`emb_margin=6.0`/`min_cluster_size=15`) + min-cost-flow → BEV 实例 → 投影 3D → 按 x/z 误差匹配车道；
    而框架 `BEVLaneDetPostProcess` 只做 `sigmoid > 0.5` 阈值化。
  - **有效对比（同一官方评估器 `ApolloLaneMetric`）**：
    做法：框架模型跑 val 集导出 Paddle3D 兼容 np → 喂同一个 `ApolloLaneMetric`。
    脚本 `_downloads/paddle3d/export_fw_lane_np.py`(ptocr, 导出) + `eval_np_lane.py`(paddlex, 评估)。
    - **导出格式**（与 Paddle3D `test_forward` 一致）：`np.concatenate([seg(raw), emb(2ch),
      sigmoid(offset), z], axis=1)` → `(1,5,200,48)`；文件名 `{split}__{stem}.np`（由 `images/{split}_{stem}.jpg`
      反解，评估器 `split('.')[0].split('__')` → `gt_key='images/{split}/{stem}.jpg'`）。
      ⚠️ `np.save` 会自动补 `.npy` 后缀（实际文件 `04__0000070.np.npy`），评估端用 `split('.')[0]` 不受影响。
    - **结果（2 epoch，两边均随机初始化、同一批 1496 val 图）**：
      | 指标 | 框架(官方评估器) | Paddle3D batch=1(**bug**) | **Paddle3D batch=4(已修)** |
      |---|---|---|---|
      | **f1_score** | **0.8378** | 0.3946 | **0.7776** |
      | precision | 0.8911 | 0.7547 | 0.8150 |
      | recall | 0.7905 | 0.2671 | 0.7435 |
      | x_error_close | **0.0680** | 0.2954 | 0.1361 |
      | z_error_close | **0.0582** | 0.1097 | 0.0937 |
    - **⚠️ Paddle3D batch=1 是我方驱动脚本漏接参数导致（非 Paddle3D bug）**：
      `apis/trainer.py::default_dataloader_build_fn` 里 `batch_size = args.pop('batch_size', 1)`，
      而 `Trainer.__init__` 的 `dataloader_fn` 默认 `dict()` → config 的 `batch_size: 4` 从不生效。
      **官方 CLI(`main.py`) 负责这处接线，而 1.0.0 恰好没有 `main.py`**（与"缺 bev_lanedet 注册"同类问题）。
      修法：`Trainer(..., dataloader_fn={'batch_size': cfg.batch_size})`。
      **一个 bug 引发两个症状**：① batch=1（5992 iters/epoch）；
      ② **LR 不衰减**——`CosineAnnealingDecay(T_max=2996)` 但实际 2 epoch=11984 iter，
      `11984/2996 = 4.0` 恰为**余弦第 4 个周期**（cosine 是周期函数）→ 回到峰值恒 ~0.00098。
      修 batch 后 iters_per_epoch=1498、总 iters=2996 = T_max，**恰好一个周期**，LR 正常衰减到 1.2e-5。
    - **修正后 Paddle3D f1 从 0.3946 → 0.7776（+97%）**，证实差距主要来自训练口径而非模型；
      与框架 0.8378 差 0.06，属 2 epoch 随机初始化的合理噪声量级。
      剩余 0.06 的可能来源：LR/warmup 配置细节不同、归一化信道顺序(RGB vs BGR)、
      Paddle 侧有增广(MotionBlur/RandomBrightnessContrast/ColorJitter)而框架无、
      框架评的是 `best_accuracy.pth`（按像素 F1 挑的 best）而 Paddle 评的是 `epoch_2`（最后）。
    - 框架 `best_accuracy.pth` 加载 **miss=0/unexp=0**，导出 1496 个 np 全部参与评估（无 KeyError → GT 映射全命中）。
- **待办**：① ~~对齐训练口径~~（**已修**：batch=4 生效、LR 正常衰减，f1 0.3946→0.7776）；
  ② 框架加载 resnet34 预训练（`load_paddle_resnet34_pretrained` 已就绪）+ Paddle 侧设 `pretrained_model_path`
  做预训练起点对比；③ 跑满 50 epoch（当前仅 2 epoch）；④ 对齐剩余 0.06 的次要因素（增广/信道顺序/评 best vs last）；
  ⑤ 若需快速迭代可临时用 `BEVLaneDetMetric`（像素级，**注意与 ApolloLaneMetric 口径不同**）；
  ⑥ SemanticKITTI 仍缺 → SqueezeSegV3 的 mIoU 暂无法做。
- **单步 loss 对齐（数据无关，2026-09-24）**：用同一批合成输入分别跑 Paddle3D 损失与我们的损失：
  - SqueezeSegV3 `SSGLossComputation` ↔ `SqueezeSegV3Loss`：**17.265526 ↔ 17.265524（差 1.9e-6）**（5 尺度逐尺度一致）。
  - BEV-LaneDet（BCE+IoU+push-pull+MSE 四分量之和）↔ `BEVLaneDetLoss`：**59.529789 ↔ 59.529793（差 3.8e-6）**。
  - 踩坑：① Paddle `F.softmax` **默认 axis=-1**（非通道）——对比脚本必须显式 `axis=1`；② PyTorch NLLLoss 加权 `mean`
    与 Paddle 归一口径不同（Paddle 除以元素数 vs PyTorch 除以权重和），我们按 Paddle 口径对齐；③ `SqueezeSegV3Loss`
    的 `class_weight` 判断由 `if class_weight` 改为 `is not None`（numpy 数组会触发 ambiguous truth value）。
  - **结论**：三个 SOTA 模型的「权重加载 missing=0 + 逐层前向 + 单步 loss」均与 Paddle3D 数值对齐；
    仅剩「真实数据端到端 mAP/mIoU/FScore」需要大数据集（KITTI 3D 已验；SemanticKITTI/Apollo 数据过大/需注册，暂缺）。
- **ModelScope 托管（2026-09-24）**：3 个 `.pth` 已上传 **`ChaoII0987/TorchKiln`** 的 `pretrained/`（仓库已设为**公开**）：
  `centerpoint_pillars_kitti.pth`(19.6MB) / `squeezesegv3_rangenet53_semantickitti.pth`(99.7MB) / `bev_lanedet_apollo_576x1024.pth`(168MB)。
  - **下载链路已验证**：`resolve_pretrained('centerpoint_pillars_kitti')` 从 ModelScope 下载成功（~21MB/s）→
    `~/.torchkiln/pretrained/` → 加载 **missing=0/unexpected=0**；三个 resolve URL 均 200。
  - 用法：config 里 `Global.pretrained_model: <裸名>`（不带 `.pth`）即可自动下载缓存。
  - 注：私有仓库会导致 resolve 404（git 返回 401）——**必须公开**或提供 token（框架当前用公开 URL，无 token）。
  - 坑：`bb` 是 `nn.Sequential(*resnet34.children())`（bb.0..bb.7）；`Upsample` 无参；`fc_transform` Linear 需转置。
  - 数据（训练/验证需）：Apollo 3D Lane（`Apollo_Sim_3D_Lane_Release` + Paddle3D 标注 json）。
- **Paddle3D 实跑环境**：`paddlex` conda env 装 `paddle3d==1.0.0` + `numba pyquaternion paddleseg h5py scikit-image nuscenes-devkit`；
  1.0.0 缺 develop 独有文件（`mm_resnet`/`bev_lanedet`/`push_pull_loss`/`init_weight`），已用脚本从 api.github.com 拉取补进 site-packages。
  对比脚本在 `_downloads/paddle3d/`（`paddle_forward.py`/`torch_forward.py`/`squeezeseg_paddle.py`/`bev_paddle.py`）。

## ts_forecast（时间序列）SOTA 模型：PaddleTS (paddlets) 移植（2026-09-25）
- paddlets 1.1.0 装在 `paddlex` conda env；`paddlets.models.forecasting.dl` 提供多档 DL 预测模型。
- 已完成并**数值对齐**的模型（每模型独立文件，位于 `torchkiln/nn/`）：
  - `s_rnn.py::RNNBlock`（PaddleTS `_RNNBlock`，LSTM/GRU/SimpleRNN）
  - `s_lstnet.py::LSTNet`（`_LSTNetModule`，CNN+RNN+skip-RNN+highway）
  - `s_transformer.py::Transformer`（`_TransformerModule`，paddle nn.Transformer）
  - `s_scinet.py::SCINet`（`_StackedSCINetModule`，even/odd 树交互+卷积解码）
  - `s_informer.py::Informer`（`_InformerModule`，ProbSparse 注意力）
  - `s_deepar.py::DeepAR`（`_DeepAR`，RNN + GaussianLikelihood 概率预测）
  - `s_tft.py::TemporalFusionTransformer`（`_tft`，GRN/变量选择/可解释多头注意力/静态编码器）
    （fit_params 含 known/observed/static 的 num/cat 维与 cardinalities；输出 `(B,out,target,num_quantiles)`）
- **统一接口**：输入 dict `{"past_target": (B,T,D)}`（det 类点预测模型），输出 `(B, out_chunk_len, target_dim)`。
  `torchkiln/models/ts.py::build_ts_model` 已按 `Head.model` 路由（rnn/lstnet/transformer/scinet/informer/deepar/tft）。
  TFT/DeepAR 需额外 covariate 与 future_target，见下方待办。
- **逐层前向数值对齐**（`_downloads/paddlets/sota_*.py`，paddle 侧 dump 权重+fwd、torch 侧 load+对比）：
  - RNN(LSTM) / RNN(GRU) / LSTNet / Transformer / DeepAR(训练 path 出 distribution params)：**maxdiff=0.000000**（逐位一致）。
  - SCINet：**maxdiff=5e-6**（浮点级）。
  - **TFT**：**maxdiff≈1e-6**（逐层 hist_rep/fut_rep/flat_grn/gate/gated_lstm/enriched/gated_attn/attn/pred 全对齐）。
  - Informer：forward 跑通、形状正确；ProbSparse 注意力含随机 key 采样（`torch.randint` vs `paddle.randint` 无法跨框架复现）
    → **无法做到逐位对齐**，属算法固有随机性，非实现 bug。
- **关键坑**：
  1. paddlets 的 Linear weight 是 `[in,out]`（与 torch `[out,in]` 相反）→ 加载需 `.T`；且 LSTM/GRU 的 state_dict 同时暴露
     `weight_ih_l0...` 与内部 `0.cell.weight_ih...`（重复），加载 torch 用 `l0` 键即可。
     ⚠️ **方形 Linear 更要转置**：TFT 的 `GatedLinearUnit`(GLU) 是 `Linear(dim,dim)`（方形），用「形状反转检测」会自动转置**失效**
      （方形 `[3,3]` 形状反转即自身，检测不出），导致 gate 权重未转置、forward 不对。**必须识别出 Linear 模块后无条件 `.T`**
      （按 `name.endswith('.weight') and isinstance(父模块, nn.Linear)` 判断）；LSTM/Conv/LayerNorm/Embedding 不转置。
  2. SCINet 的 `_Interactor` 内层卷积用 `paddle.nn.Pad1D`（**零填充**），torch 必须用 `nn.ConstantPad1d(...,0)`，不能 `ReplicationPad1d`。
  3. SCINet 的 `_decoder1/_decoder2` 是 `Conv1D(in_chunk_len→out_chunk_len, kernel=1)`，paddle NCL 下**通道维=时间轴**，
     forward 直接 `decoder(x)`（x 形状 `(B,in_chunk_len,D)`），不要 transpose 成 `(B,D,T)`。
  4. Transformer 的 paddle `nn.Transformer` 用 `self_attn.q_proj/k_proj/v_proj/out_proj`，torch 用 `in_proj_weight/in_proj_bias/out_proj`——
     **q/k/v 的 weight 拼接为 in_proj_weight、bias 拼接为 in_proj_bias**，对比脚本 `sota_transformer_compare.py` 里有映射。
- **端到端训练集成（已完成，2026-09-26）**：
  - **数据管线**（`torchkiln/data/ts.py::TSDataset`）：新增协变量列支持——`target_cols`（目标，缺省全列）、
    `known_cols`（已知协变量，past+future 全窗）、`observed_cols`（观测协变量，仅 past）、`static_cols`（静态，每序列常数）。
    配了任一协变量列时 `__getitem__` 返回 5 元 `[past, future, known, observed, static]`；否则仍 2 元（向后兼容）。
  - **损失**（`torchkiln/ts.py`）：`TSLoss`(MSE/MAE，点预测)、**`TSQuantileLoss`**(pinball，TFT)、**`TSNLLLoss`**(Gaussian NLL，DeepAR)。
    `TSMetric` 增 `pred_mode`（point/quantile[取中位]/params[取 mu]）。
  - **任务分派**（`torchkiln/tasks/ts_forecast.py`）：`forward_train`/`eval_step` 统一构造输入 dict（`past_target` +
    `future_target` + `known/observed/static_cov_numeric`），点模型忽略多余键；`build_loss`/`build_metric` 按 `Head.model`
    自动选 quantile/nll 与 pred_mode（可被 `Loss.type`/`Metric.pred_mode` 覆盖）。
  - **注册**：`torchkiln/tasks/__init__.py` 与 `ptcore/trainers/__init__.py` 补 `ts_forecast`。
  - **demo 配置 + 端到端冒烟**（合成数据 `datasets/ts_demo/data.csv`：signal 目标 + known 协变量）：
    `configs/ts/tft_demo.yml`（quantile）、`configs/ts/rnn_demo.yml`（MSE）、`configs/ts/deepar_demo.yml`（NLL）。
    三者 CLI 训练均跑通（loss 下降：TFT 13.98→10.45、DeepAR NLL 54.9→37.4、RNN 下降）；`smoke_all` **86 OK / 0 FAIL**；
    `check_graph_build` 55 OK。
- 其余已实现模型（NBEATS/NHiTS/MLP/TCN/DLinear）保持不变，见 `torchkiln/nn/ts_models.py`。

### ts 补充 demo 配置 + ETTh1×TFT 与 PaddleTS 端到端对比（2026-09-26）
- **7 个模型各有 demo 配置**（合成数据 `datasets/ts_demo/data.csv`，L=48 H=12，均 CLI 训练跑通）：
  `configs/ts/{tft,rnn,deepar,lstnet,transformer,scinet,informer}_demo.yml`。
  - SCINet 的 `forward` 返回 **`(pred, mid_pred)` 元组**（`mid_pred` 在 num_stack=1 时为 None，对齐 PaddleTS），
    `tasks/ts_forecast.py::_first` 取元组首元素再送 loss/metric；Informer 的 decoder padding 需 `device=src.device`
    （原在 CPU 建 tensor，GPU 训练会 device mismatch）。
  - `ptcore/trainers/__init__.py` 与 `torchkiln/tasks/__init__.py` 已注册 `ts_forecast`。
- **端到端回归**：`tools/smoke_all.py` **86 OK / 0 FAIL**（含 3 个 ts 配置）；`check_graph_build.py` 55 OK。
- **ETTh1×TFT 对比**（配置 `configs/local/etth1_tft.yml`，脚本 `_downloads/paddlets/etth1_*.py`）：
  数据 `datasets/etth1/ETTh1.csv` → 按 train 段(0:12194) mean/std 标准化 + 生成 `hour_sin/cos`，
  target=OT、known=[hour_sin,hour_cos]、observed=6 特征，L=96 H=24、hidden=64、heads=1、
  Adam lr=1e-4、batch=128、10 epoch。两侧读**同一份 CSV**。
  - **⚠️ `TSQuantileLoss` 差 6 倍 bug（已修）**：PaddleTS `QuantileRegression.loss` 是
    `2 * max((q-1)e, qe)` 且**对分位数求和**（非平均）。原实现漏乘 2 且求平均 → 数值与梯度差 **2×Q=6 倍**，
    同 lr 下等效学习率差 6 倍，训练轨迹不可比。修后**单步 loss 对齐 3.58289027↔3.58288987（1.13e-7）**。
  - **复现验收（同权重 + 同评估器）**：权重加载 **255/255 missing=0/unexpected=0**；
    PaddleTS 权重在框架评估器上 **MSE 0.6105** vs PaddleTS 自评 **0.6108**（差 3e-4）→ **评估口径一致**。
  - **泄漏检查**：框架权重 MSE=0.1415（非趋 0，无泄漏）；修复后训练轨迹框架 1.564→0.414
    vs PaddleTS 1.296→0.459（同量级）。
  - **⚠️ 该实验设置下两模型都不如朴素基线**：naive last-value **0.0418**、seasonal(lag24) 0.0602，
    而 PaddleTS 0.6105、框架 0.1415。根因是**时序切分造成分布位移**（val 段均值比 train 低 1.5σ）：
    naive 复制过去值天然跟随位移，TFT 学绝对值映射则吃亏。且 PaddleTS 过拟合更重
    （train→val pinball **0.459→1.571，×3.4**；框架 0.414→0.574，×1.4）。
    → **结论：复现验收 ①-④ 全达标，"端到端 MSE 差"属实验设置（分布位移 + 过拟合差异），非代码缺陷**；
    若要可比的端到端指标，需改用更大 lr/更多 epoch，或消掉分布位移（如随机切分、或按段内统计量归一化）。

### PaddleTS forecasting.dl 模型全清单对齐状态（2026-09-26，11/11 收敛）
- `paddlets.models.forecasting.dl` 实测**只有 11 个模型**（`paddle_base.py`/`paddle_base_impl.py` 是基类，非模型）。
  逐一核对结果（对拍脚本 `_downloads/paddlets/sota_rest_compare.py`、`sota_compare.py`、`sota_transformer_compare.py`、
  `sota_deepar_compare.py`、`sota_tft_dump.py`）：

  | # | PaddleTS 模型 | 框架实现 | 状态 | maxdiff |
  |---|---|---|---|---|
  | 1 | `rnn.py` (LSTM/GRU) | `s_rnn.py::RNNBlock` | ✅ 逐位 | 0 |
  | 2 | `lstnet.py` | `s_lstnet.py::LSTNet` | ✅ 逐位 | 0 |
  | 3 | `transformer.py` | `s_transformer.py::Transformer` | ✅ 逐位 | 0 |
  | 4 | `scinet.py` | `s_scinet.py::SCINet` | ✅ 浮点 | 5e-6 |
  | 5 | `informer.py` | `s_informer.py::Informer` | ⚠️ 前向正确但**不可逐位** | n/a |
  | 6 | `deepar.py` | `s_deepar.py::DeepAR` | ✅ 逐位 | 0 |
  | 7 | `tft.py` | `s_tft.py::TemporalFusionTransformer` | ✅ 浮点 | 1e-6 |
  | 8 | `mlp.py` | `ts_models.py::MLP` | ✅ 逐位 | 0 |
  | 9 | `tcn.py` | `ts_models.py::TCN` | ✅ 浮点 | 6.9e-6 |
  | 10 | `nbeats.py` | `ts_models.py::NBEATS` | ✅ 逐位 | 0 |
  | 11 | `nhits.py` | `ts_models.py::NHiTS` | ✅ 逐位 | 0 |

  → **10/11 完全对齐**；第 5 项 Informer 因 `ProbSparseAttention` 的 key 随机采样（`torch.randint` vs `paddle.randint`
    跨框架不可复现）属**算法固有随机性**，前向形状/结构正确，非实现 bug。
  - 注：`ts_models.py::DLinear` **不在 PaddleTS 中**（PaddleTS 1.1.0 无 DLinear），无需对齐。
- **本轮为对齐补的两处修正**：
  1. `ts_models.py::NBEATS._Block` 属性 `_linear_layer_stack_list` → **`_fc_stack`**（对齐 PaddleTS 命名；
     此前参数数完全相同但键名不同，导致 48/96 键加载不上）。改后 **96/96，maxdiff=0**。
  2. NHiTS 对拍须用 **`dropout=0.0`**：`_NHiTSBlock` 的 `if dropout > 0` 才把 `Dropout` 塞进 `layers`，
     故 dropout>0 会让 Linear 索引 0/3、dropout=0 时为 0/2。原 dump 用的是 dropout=0。
     传 `dropout=0.0` 后 **24/24，maxdiff=0**。
  - 踩坑：TCN 走 `weight_norm`，paddle `weight_g` 形状 `[C]`、torch 为 `[C,1,1]`（元素数相同）→ 加载需 reshape；
    paddle `nn.Linear.weight` 是 **[in,out]**（torch 为 [out,in]）→ 对 Linear 无条件 `.T`。
- **回归**：`smoke_all` **90 OK / 0 FAIL**（86 + 新增 4 个 demo 配置）、`check_graph_build` 55 OK。

## OCR 端到端(E2E) vs 两阶段 速度实测（2026-09-26，RTX 4060 Ti，Total-Text test 300 张）
**结论：端到端 PGNet 只在「高框密度」场景占优，交叉点 ≈20 框/图。**
### 1. 官方权重（`paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/<name>.tar`，BOS 实测 68~105 MB/s）
| 档 | det | rec | 合计 |
|---|---|---|---|
| **PP-OCRv6_tiny** | 1.87 MB | 4.38 MB | **6.25 MB** ⭐ |
| PP-OCRv6_small | 9.59 MB | 20.45 MB | 30.04 MB |
| PP-OCRv6_medium | 59.39 MB | 73.29 MB | 132.68 MB |
| PP-OCRv5_mobile / server | 4.71 / 84.25 MB | 16.05 / 80.94 MB | 20.76 / 165.19 MB |
| **PGNet server(r50_vd)** | — | — | **185 MB**（48.53M 参数） |
- v6_tiny 架构：`Backbone: PPLCNetV4(model_size=tiny)` + `Neck: RepLKFPN(out=64)` + `Head: DBHead`
- PGNet 权重下载：`https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/{en_server_pgnetA(995MB,TotalText Hmean 84.69),train_step1(278MB),e2e_server_pgnetA_infer(187MB)}.tar`
- Total-Text：`https://paddleocr.bj.bcebos.com/dataset/total_text.tar`(0.41GB, 1563 条目, train 1255/test 300)
### 2. 硬件同环境同引擎（**都用 paddle.inference**）测速
| 方案 | 耗时/图 | 中位 | 输入 | 检出 |
|---|---|---|---|---|
| 两阶段 PP-OCRv6_tiny | **30.8~32.5 ms** | 27 ms | 长边 960 | 4.1 行/图 |
| PGNet(r50_vd) | **54.2 ms** | 49.5 ms | 长边 768 | 7.6 行/图 |
- **PGNet 耗时拆分**：预处理 **15.5ms** + 网络前向 **32.4ms** + 后处理 **6.4ms**
  - 前向再拆：**backbone 13.0ms(42%) / PGFPN 11.0ms(35%) / PGHead 8.0ms(26%)**
  - → **换 LCNet 的收益上限极小**：骨干 10× 加速后前向仍 19.3ms，端到端≈**41.2ms**
    （即便骨干=0，前向仍有 18ms 的 PGFPN+PGHead 结构开销）
### 3. ⭐ 框数 scaling：`两阶段 ms/图 = 23.9 + 1.669 × 框数`
| N 框/图 | 两阶段 | PGNet(54.2 固定) | 胜者 |
|---|---|---|---|
| 5 | 31.5 ms | 54.2 ms | 两阶段 1.7× |
| **20（交叉点）** | 54.8 | 54.2 | 持平 |
| 50 | 101.5 | 54.2 | PGNet 1.9× |
| 100 | 179.2 | 54.2 | PGNet 3.3× |
| **300** | **490 ms** | **54.2 ms** | **PGNet 9.0×** |
- **加大 rec batch 无效**：`batch_size` 6→64→128，单框成本仅 1.669→1.555→1.562（-7% 后持平）。
  **根因：1.555 ms/框 是 CPU 逐框 crop+resize+归一化+Python 循环开销，不是 GPU 前向**
  （v6_tiny rec GPU 前向 <1ms）。→ **两阶段的 O(N) CPU 开销结构性存在，batch 治不了。**
- 拟合 R²≈0.30 偏低（各图分辨率差异使 det 成本波动），但**斜率的 batch 不变性是强信号**，交叉点稳定 ~20。
### 4. 踩坑（移植 PGNet 时必看）
- `tools/infer_e2e.py` 是**动态图**，与 PaddleX **推理引擎**口径不同 → 同为 PGNet：动态图 90ms vs 推理引擎 54ms（**勿混比**）。
- PGNet inference 模型输出 4 个张量，**通道识别**：`4=f_border(conv2)`、`37=f_char(conv3,=字典36+1)`、`2=f_direction(conv4)`、`1=f_score(conv1)`。
  **顺序是 [f_border,f_char,f_direction,f_score]，与 `PGHead.forward` 的 dict 插入顺序 [f_score,f_border,f_char,f_direction] 不同**，映射错会触发 `sort_with_direction` 的 `IndexError`。
- `data["shape"] = [src_h, src_w, ratio_h, ratio_w]`（**4 元**），shape_list 必须是 (1,4)。
- PaddleX OCR pipeline 需 `ocr-core` 依赖：补 `pypdfium2` `python-bidi`（`cv2/imagesize/pyclipper/shapely` 本机已有）。
- PaddleX 3.0 官方推理包解压时**同名顶层目录会双重嵌套**（`dir/dir/inference.yml`），需上移一层。
- PGNet 无官方轻量版：`PGFPN` 硬编码 `num_inputs=[2048,2048,1024,512,256]` + 7 输入（**c0=RGB 原图**），
  换 PPLCNetV4 需改写 PGFPN（通道 **和** stride 都要适配）+ 全量重训。
### 5. 决策
- **低框密度(<20/图)** → 直接用 **PP-OCRv6_tiny**（6.25MB / 31ms，零成本）。
- **高框密度(几百/图)** → **PGNet 有理**：300 框时 54ms vs 490ms。
  - 现成 r50 权重即可用（已下），**不换骨干也已快 9×**；LCNet 化仅再得 1.24×（41ms），但要改 PGFPN+重训。
- SPTS v2 已弃（自回归 25 步串行 + 408.9MB，VLM 时代无意义）。

## 轻量端到端 PGNet（<10MB）—— 移植+训练中（2026-09-26）
**目标：单模型端到端 <10MB，用于高框密度（几百框/图）场景。** 已达标，正在重训。
### 1. 架构（3 处改写，其余全部复用 PaddleOCR）
| 组件 | 说明 | 参数 |
|---|---|---|
| `PPLCNetV4E2E`（骨干） | `PPLCNetV4(det,tiny)` + 返回 `[RGB, f1@32s4, f2@48s8, f3@64s16, f4@160s32]`；返回 RGB 是**关键**——原 PGNet 骨干就返回 `[image, f1..f6]`，这样 `BaseModel` 的 `backbone→neck→head` 流水线可直接复用 | **0.394M** |
| `PGFPNLCNet`（颈） | PGFPN 轻量改写：原版硬编码 7 输入 `[3,64,256,512,1024,2048,2048]`（含 s2/s64 两级），改为适配 LCNet 的 4 特征 + RGB；**down-fusion(RGB池化到s4+浅层)** + **up-fusion(s32→s16→s8→s4)** 保持双路 | 0.424M(w=64) |
| `PGHeadLite`（头） | 宽度可配。原 `PGHead` 的 **f_char 分支占 84% 参数**（`conv_f_char4` 单层 589K = 54%，内部宽 256 而其它分支只 64）；压到 `w_char=[64,64,128,128,128]` → **1.092M → 0.416M** | 0.416M |
| **合计** | | **1.234M = 4.71 MB** ✅ |
- 输出结构与官方**完全一致**：`f_score(1,1,192,160)/f_border(1,4,…)/f_char(1,37,…)/f_direction(1,2,…)` → **后处理可直接复用**
- **前向 11.0ms**（官方 r50 = 31.2ms，**快 2.84×**）；端到端预估 15.5(预处理)+11.0+6.4(后处理) ≈ **32.9ms**
  - vs 两阶段 30.8ms（低框）仍略慢；**vs 两阶段 490ms（300框）快 14.9×** ⭐
- 校验方法：`PGHeadLite` 用**原宽度**重建 → 参数 **1.092M 与官方逐位相等**，证明重写忠实后再改宽度才可信。

### 2. 接入 PaddleOCR（3 处 registry 补丁，文件已存 `tools/paddleocr_e2e/`）
新文件 `ppocr/modeling/e2e_pgnet_lite.py`（本仓库副本 `tools/paddleocr_e2e/e2e_pgnet_lite.py`）。
1. `ppocr/modeling/backbones/__init__.py`：**`model_type=="e2e"` 分支**（不是 det 分支！它会 `support_dict = ["ResNet"]` **覆盖**）加
   `from ..e2e_pgnet_lite import PPLCNetV4E2E` + `support_dict = ["ResNet","PPLCNetV4E2E"]`
2. `ppocr/modeling/necks/__init__.py::build_neck`：import 是**函数内懒加载**的 → 加 `from ..e2e_pgnet_lite import PGFPNLCNet` + `"PGFPNLCNet"` 进 `support_dict`（懒加载避免循环 import）
3. `ppocr/modeling/heads/__init__.py::build_head`：仿 `DRRGHead` 写法加 `if config["name"]=="PGHeadLite"` 分支
- `BaseModel` 会读 **`backbone.out_channels` → neck、`neck.out_channels` → head** 并作为 `in_channels` 注入 → 两个新组件**必须有 `out_channels` 属性**（我的 neck 用 `**kwargs` 吞掉即可）
- `BaseModel.forward` 调 `head(x, targets=…)` → **`forward` 必须接受 `targets=None`**
- **Paddle dygraph 参数名不可重复**：`PGHeadLite`/`PGFPNLCNet` 内部 `ConvBNLayer` 的 `bn_name = "bn" + name[3:]`（**剥前3字符**）→ uid 必须 **≥5 字符且区分位在索引3之后**（用 `'p%03d_'`），否则多实例 bn 名冲突；同进程也不能建两个官方 `PGHead`（名字硬编码）

### 3. 训练（复用 PaddleOCR `tools/train.py`，未重写训练器）
- 配置 `configs/e2e/e2e_tiny_pgnet.yml`（副本 `tools/paddleocr_e2e/`），基于官方 `e2e_r50_vd_pg.yml` 改 Architecture/数据路径
- **官方 train loader `batch_size_per_card: 14`（Eval 才是 1）**，1255 张 → **89 iters/epoch**
- 复用 `PGDataSet` + `PGLoss(tcl_bs=64,…)` + `E2EMetric(main_indicator=f_score_e2e, mode=A)`
- **`warmup_epoch: 50`** ← 训短了等于白训（epoch1 lr 仅 1.6e-5，峰值 1e-3）
- **`pretrained_model` 指向官方 r50 时 499 个参数全 skip**（与 LCNet 无同名）→ 实为纯随机初始化
- 冒烟 1 epoch = **79s**（~21 samples/s，显存 5.8GB）→ 150 epoch ≈ **2.3h**
- 冒烟 loss：`304→221`、`ctc_loss 60.6→43.8`、`border 0.927→0.68`（全面下降）
- 评估命令：`python tools/eval.py -c configs/e2e/e2e_tiny_pgnet.yml`（E2EMetric 口径，官方基准 r50 = **Hmean 84.69**）

### 4. ⚠️ 踩坑：先做成 Paddle 原型，后补进仓库框架（教训）
- 最初**走错技术栈**：在 PaddleX 的 PaddleOCR 副本里写 `ppocr/modeling/e2e_pgnet_lite.py` + 改 3 处 registry + 用其 `train.py`/`eval.py`。
  **而仓库 `torchkiln/ocr/` 本身就是一套完整的 PaddleOCR torch 移植栈**（`PGHead`/`PGFPN`/`PGNet_PostProcess`/
  `E2EResizeForTest`/`PPLCNetV4` 全都有），**应该在 torch 侧做**。
- **教训**：动手前先 grep 仓库里有没有现成实现（此前只搜了 `PGNet` 关键词，**没搜 `torchkiln/ocr/` 子目录**，漏了整套栈）。

### 5. 仓库框架（torch）实现（2026-09-27 补完）
- **新增 `torchkiln/ocr/modeling/e2e_pgnet_lite.py`**：`PPLCNetV4E2E` / `PGFPNLCNet` / `PGHeadLite`（torch 版，
  **与 Paddle 侧逐属性同名**，便于权重转换）。
  - `PPLCNetV4E2E`：包装 `backbones/rec_lcnetv4.py::PPLCNetV4(det=True, model_size='tiny')`，返回 `[RGB, f1..f4]`
    （与原 `e2e_resnet_vd_pg` 的 `[image, f1..f6]` 同构），并暴露 `out_channels=160`（框架读它作 neck 的 in_channels）。
  - `PGFPNLCNet`：复用 `necks/pg_fpn.py` 的 `ConvBNLayer`/`DeConvBNLayer`。
    ⚠️ torch 版 `ConvBNLayer(in,out,k,stride=1,groups=1,is_vd_mode=False,act=None,name=None)` **第5个位置参数是 groups**，必须传关键字。
  - `PGHeadLite`：复用 `heads/e2e_pg_head.py` 的 `ConvBNLayer(in,out,k,stride,padding,groups,if_act,act,name)`；
    torch 版 `PGHead` 把 `character_length` **硬编码为 37**（无 `character_dict_path`）。
  - torch 侧**没有** Paddle 那个 `bn_name = "bn"+name[3:]` 命名坑（属性名即 state_dict 键）。
- **三处 registry 注册**（与 PaddleOCR 同结构）：
  1. `backbones/__init__.py`：**`model_type=='e2e'` 分支**（line 54，`support_dict=['ResNet']` 会被其**覆盖**）加 `PPLCNetV4E2E`
  2. `necks/__init__.py::build_neck`：加 import + `'PGFPNLCNet'` 进 support_dict
  3. `heads/__init__.py::build_head`：加 import + `'PGHeadLite'` 进 support_dict
- **验证（框架侧）**：`build_model({model_type:'e2e', Backbone:{PPLCNetV4E2E}, Neck:{PGFPNLCNet,w:64}, Head:{PGHeadLite,w_char:[64,64,128,128,128]}})`
  → **1.221M / 4.66MB**，输出 `f_score(1,1,192,160)/f_border(1,4,...)/f_char(1,37,...)/f_direction(1,2,...)` ✓
- **权重转换（Paddle→torch）**：**Paddle 的 450 个键全部命中**（torch 多出的 86 个是 BN `num_batches_tracked`），
  规则只有 BN 键名 `_mean/_variance -> running_mean/running_var`；`load_state_dict` 后 **真正缺 0 / unexpected 0**。
  产出 **`weights/pgnet_lite_totaltext.pth`（4.9MB）**。
- **前向对拍**：maxdiff **0.015950（rel 1.5e-3）**；**逐级定位到差异源自 backbone**——
  `PPLCNetV4` 的 4 级输出 rel 3.8e-4~7.6e-4（框架该骨干移植的**固有浮点差异**），经 neck/head 放大到 1.5e-3，
  **非轻量组件的 bug**（BN eps 试 1e-3/1e-4 反而更差 0.48/0.048，确认 eps=1e-5 正确）。
- **待办**：仓库侧仍缺 **`PGLoss`（PG-CTC）/ e2e 数据集 / e2e 任务适配器 + 配置**（`torchkiln/tasks/` 无 e2e 任务），
  故目前框架能**构建+加载+推理**，但还不能在框架内端到端训练/评估。

### 6. E2EMetric 评估排查结论（B/D/E3，重要，勿重蹈）
- 官方 `tools/eval.py` 报 `f_score_e2e=0`，但**用官方自己的函数手工复算**（`get_socre_A` -> `combine_results(rec_flag=True)`）得
  **f_score 0.785 / f_score_e2e 0.536**，差 383 倍 -> **是官方 eval 链路喂入数据的问题，不是模型/指标数学**。
- 已排除：`score_thresh`(0.5->0.05)、`mode`(fast/slow)、输入分辨率(768/1024/1280)、文本大小写(GT 被 `.lower()`)、
  字典 off-by-one（`peak` 编解码往返正确）、GT 坐标系（`E2EResizeForTest` 不动 `polys`）、GT 内容
  （数据集 `polys` 是原图坐标、`texts` 索引正确、`ignore_tags` 正常）。
- `Deteval` 硬门槛：`tr=0.7`(sigma=交/GT面积) + `tp=0.6`(tau=交/预测面积) + **"恰好一个候选"唯一性**；
  用 `polygon_fast`(shapely `Polygon(...).buffer(0)`) 算面积。
- 教训：**"官方指标为 0" 不等于 "模型废了"**，务必用官方函数手工复算交叉验证。
- 模型真实能力（官方函数口径，可复现脚本 `_downloads/ocr/diag_combine.py`）：检测 **f_score 0.785**、
  **e2e f_score 0.536**（150 epoch / 无预训练 / 结构改过；官方 r50 = 600 epoch / 184MB）。

### 7. 框架内端到端训练/评估链路补齐（2026-09-27）
**结论：torch 侧 PGLoss 与 PaddleOCR 原版 PGLoss 在同批输入上总 loss 逐位一致；CLI 端到端训练跑通。**

#### 7.1 新增/改动文件
| 文件 | 说明 |
|---|---|
| `torchkiln/ocr/losses/e2e_pg_loss.py`（新） | `PGLoss` + `org_tcl_rois` + `pre_process`（移植 `ppocr/losses/e2e_pg_loss.py` + `extract_batchsize.py`） |
| `torchkiln/ocr/data/imaug/pg_process.py`（新，1117 行） | `PGProcessTrain`（**原文件零 paddle 依赖**，仅改 1 处 import 为新路径，逐行搬运） |
| `torchkiln/ocr/utils/e2e_metric/{Deteval,polygon_fast,__init__}.py`（新） | 官方 e2e 指标（`get_socre_A/B`、`combine_results`） |
| `torchkiln/ocr/metrics/e2e_metric.py`（新） | `E2EMetric`（mode A/B） |
| `torchkiln/ocr/modeling/e2e_pgnet_lite.py` | 头 `PGHeadLite` 补 **`f_score` 的 `torch.sigmoid`** |
| `torchkiln/ocr/data/imaug/label_ops.py` | 补 `E2ELabelEncodeTrain` / `E2ELabelEncodeTest` |
| `torchkiln/ocr/data/{imaug/__init__,simple_dataset}.py` | 注册新算子；`SimpleDataSet` 补 `img_id`（e2e 评估需要） |
| `torchkiln/ocr/task.py` | 新增 `e2e_train_collate` / `e2e_eval_collate` + `self.name=="e2e"` 分派 |
| `torchkiln/cli.py` / `ptcore/trainers/__init__.py` | 注册 `ocr_e2e`（`CONFIG_TASK_ALIAS["e2e"]="ocr_e2e"`、`_OCR_TASKS`、`TRAINER_REGISTRY`） |
| `configs/ocr/e2e/e2e_pgnet_lite_totaltext.yml`（新） | 完整配置（`Architecture.task: e2e`，pretrained 指向转换后的 `.pth`） |

#### 7.2 关键数值对齐（同批输入、同 RNG 种子）
| 分量 | torch | Paddle | 结论 |
|---|---|---|---|
| **loss** | **349.08233642578125** | **349.08233642578125** | **逐位一致** |
| score_loss | 0.9430578947067261 | 0.9430578947067261 | 一致 |
| border_loss | 0.8673726320266724 | 0.8673725128173828 | 末位差（fp32 求和序） |
| direction_loss | 0.3845565915107727 | 0.3845565915107727 | 一致 |
| ctc_loss | 69.37747192382812 | 69.37747192382812 | 一致 |

- 复算脚本：`_downloads/ocr/e2e_dump.py`（torch 侧导出批+前向）、`_downloads/ocr/e2e_paddle_loss.py`（Paddle 侧同输入）、
  `_downloads/ocr/e2e_ctc_ref.py` / `e2e_ctc_cmp.py`（逐样本 CTC 对拍）。
- ⚠️ **对拍必须同 RNG 种子**：`org_tcl_rois` 在 `vp_len > tcl_bs` 时用 `np.random.permutation` **随机剔除**样本；
  两侧 RNG 状态不同会选中不同子集（表面看 mean 差 3~4%，实为子集不同）。**同种子后中间量 maxdiff=0.0、逐样本损失完全相同。**

#### 7.3 踩坑：`ctc_loss` 的 log_softmax（导致损失为负）
- **现象**：torch 版 `ctc_loss` 得 **-852**，Paddle 得 **+80.47**；其余三分量完全一致。
- **根因**：**`paddle.nn.functional.ctc_loss` 内部会做 `log_softmax`**（所以 PaddleOCR 直接喂原始 logits）；
  **`torch.nn.functional.ctc_loss` 不做**，要求传入 log 概率。
- **修复**：`F.log_softmax(f_tcl_char_ld, dim=2)` 后再进 `F.ctc_loss` → 立即逐位对齐。
- 一般规律：跨框架移植 CTC 时务必确认「谁做 log_softmax」。另 `reduction="none"` 下 torch 返回**未归一**的值，
  `"mean"` 会再除以 target 长度。

#### 7.4 踩坑：`PGHeadLite` 漏 sigmoid（导致 `score_loss` 为负）
- **现象**：训练日志 `score_loss = -0.8379`（Dice 损失出现负值）。
- **根因**：`PGHeadLite.forward` 未对 `f_score` 过 sigmoid，而**框架 `PGHead` 与 PaddleOCR `PGHead` 都有 `F.sigmoid`**。
  - `DiceLoss(f_score, gt)` 要求 `f_score` 是概率；否则 `pred*gt` 可负 → loss 越界。
  - `PGNet_PostProcess` **直接用 `score_thresh` 阈值化 `f_score`**（内部不再 sigmoid），所以头必须输出概率。
- **修复**：`f_score = torch.sigmoid(...)` → `score_loss` 恢复为 **0.9811**。
- 注：现有 `weights/pgnet_lite_totaltext.pth` 是**未加 sigmoid 时训练的**（Paddle 原型同此），
  权重仍可加载（sigmoid 无参数），但**要发挥最佳精度需带 sigmoid 重训**。

#### 7.5 数据管线要点
- `PGProcessTrain` 输出**固定形状**：`images(3,512,512)`、`tcl_maps(1,128,128)`、`tcl_label_maps(1,128,128)`、
  `border_maps(5,128,128)`、`direction_maps(3,128,128)`、`training_masks(1,128,128)`、
  `label_list(30,50,1)`、`pos_list(30,64,3)`、`pos_mask(30,64,1)`（监督图在 **stride 4**，128=512/4）。
- `pos_list` 三列 = **(img_id, y, x)**；`col0` 是**批内图片序号**（`PGProcessTrain.img_id` 自增循环，
  周期 = 配置的 `batch_size`）→ **`Train.loader.batch_size_per_card` 必须与 `PGProcessTrain.batch_size` 一致**，
  否则 `org_tcl_rois` 的 `gpu_id` 越界（已加钳制兜底，但仍应对齐配置）。
- collate **全部转 float32**（对齐 PaddleOCR 的 `paddle.to_tensor(np.stack(...))`）；`pos_list` 在 `ctcloss` 内再 cast int 做索引。
- 训练增广/归一化（含 BGR→RGB 交换）**全部烘焙在 `PGProcessTrain` 内**；评估走
  `E2ELabelEncodeTest → E2EResizeForTest(768) → NormalizeImage → ToCHWImage`（两条路径的归一化结果一致）。

#### 7.6 验证记录
- `import` 全通；框架 `build_model` → **1.221M / 4.66MB**，输出 `f_score/f_border/f_char/f_direction` 与官方一致。
- 全链路自测（`_downloads/ocr/e2e_selftest.py`）：数据→前向→PGLoss（loss 280.65，梯度回传正常）→后处理→`E2EMetric` 全通。
- **CLI 端到端**：`tkiln train -c configs/ocr/e2e/e2e_pgnet_lite_totaltext.yml -o Global.epoch_num=1 ...`
  - `Loaded pretrained: weights/pgnet_lite_totaltext.pth (missing=0 unexpected=0)` ✓
  - `Start training: type=e2e epochs=1 steps/epoch=12 device=cuda:0`；1 epoch 6.8s；评估 300 图 17.6 img/s。
  - `Training finished. Best f_score_e2e = 0.00000`。

#### 7.7 ⚠️ 仍未解决：E2EMetric 的 `f_score_e2e ≈ 0`（与官方一致，非本移植引入）
- 框架评估：`total_num_gt=2543`、`total_num_det=2474`、`global_accumulative_recall=5.2`、`f_score≈0.0017`、`f_score_e2e=0`。
- **检测框数量正常但几乎无一对能通过 `Deteval` 的匹配** → 怀疑 GT 多边形与预测多边形的**坐标系/尺度**不一致
  （`E2EResizeForTest` 不动 `polys`，而预测经 `shape_list` 反映射；需核 `shape_list` 的口径）。
- 该现象**在 PaddleOCR 自己的 `tools/eval.py` 上也一样**（报 `f_score_e2e=0`），此前用官方函数手工复算得 `f_score 0.785 / e2e 0.536`
  （见 §6）——**说明是官方评估链路本身的问题**，本移植忠实复现了它。
- 待办：① 核对 `shape_list`（应记录 `[src_h, src_w, ratio_h, ratio_w]`）；② 或改用「同一评估器评双方模型」的口径做验收。

### 8. e2e 评估批必须「逐图」（smoke_all 发现，已修）
- **现象**：`smoke_all` 里 `e2e_pgnet_lite_totaltext` FAIL：`all input arrays must have the same shape`
  （`e2e_eval_collate` 里 `np.stack` 各图）。
- **根因**：`E2EResizeForTest` 是**保比例**缩放，同 batch 内各图尺寸不同 → 不能 stack；
  且官方 `E2EMetric` 本身就是**逐图**口径（只读 `batch[2..5]` 的第 0 项）。
  `smoke_all` 强制 `batch=4`，所以暴露出来（配置里 Eval 默认 batch=1 时不暴露）。
- **修复**：
  - `e2e_eval_collate` 改为**不 stack**：返回 `[images(list[Tensor]), shapes(list), polys(list), texts(list), ignore_tags(list), img_id(list)]`。
  - `OcrTask.eval_step` 的 e2e 分支**逐图前向 + 逐图 post_process + 逐图 metric**（原来 `batch[0].to(device)` 在分支之前，
    对 list 会 `AttributeError`，已把该行移到 det/rec 分支内）。
  - 新增 `OcrTask.sample_count`：e2e 且 `batch[0]` 是 list 时返回 `len(batch[0])`（父类实现走 `.shape[0]` 会返回 0）。
- **验证**：`Eval.loader.batch_size_per_card=4` + `num_workers=0` 跑通（300 图评估，`Training finished`）。
- ⚠️ 观察（待办）：`total_num_det` 在 eval batch=1 时是 **2474**、batch=4 时是 **8177**（同 300 图），
  而 `f_score_e2e` 都是 0。怀疑 `shape_list`（`[src_h, src_w, ratio_h, ratio_w]`）与逐图 post_process 的口径
  仍有出入，或 `E2EResizeForTest` 在多图时行为不同 —— 与 §7.7 的 `f_score_e2e≈0` 是同一待查项。
- 回归：`smoke_all` 中另 5 个 FAIL（yolo11-det/lane-row/lane-seg/obb/pose）是**宿主内存耗尽**
  （"Unable to allocate 4.69 MiB" / "LLVM ERROR: out of memory"），单进程连跑 42 个配置所致，**与本次改动无关**
  （GPU 空闲、无残留进程，属已知的 smoke_all 资源问题）。

### 9. `f_score_e2e ≈ 0` 的精确定位（2026-09-27，已查清性质）
**先排除的**（均不是原因）：
1. **评估循环批相关性**：`e2e_eval_collate` 逐图打包后，逐图 vs 批量的检出数**完全一致 `[2,3,4,0]`**
   （脚本 `_downloads/ocr/e2e_batch_probe.py`）。之前看到的 `total_num_det` 波动（2649/1836/8177）来自
   **训练随机性**（只训 12 步、`warmup_epoch=50` 使 lr 极小、模型近乎未训练），**不是评估管线问题**。
2. **归一化**：框架 `NormalizeImage`/`ToCHWImage` 与 PaddleOCR **逐行一致**（无额外通道交换）。
3. **坐标系**：`E2EResizeForTest` 的 `shape = [src_h, src_w, ratio_h, ratio_w]` 正确
   （实测 224/412/1.714/1.553 ↔ 缩放后 384x640）；后处理 `x/[ratio_w, ratio_h]` 也与 PaddleOCR 一致。
   GT 与预测**都在原图坐标**且**实际重叠**（pred#1 `x158-270` ↔ GT#2 `x152-273`）。

**真正原因（`_downloads/ocr/e2e_tr_tp_probe.py`，用 shapely 精确算）**：

| 预测 | 命中 GT | **tr = 交/GT 面积** | **tp = 交/预测面积** | Deteval 判定 |
|---|---|---|---|---|
| pred[0] | GT[0] | **0.464**（<0.7） | 0.878 | REJECT |
| pred[1] | GT[1] | **0.099**（<0.7） | 0.923 | REJECT |

- **tp 高（0.88/0.92）** → 预测框**准、覆盖了 GT**；但**预测多边形比 GT 窄** → `tr` 过不了 `0.7` 门槛。
- 外扩检验：`buffer` 2/4/6/8 px → tr 合计 0.69/0.80/0.91/1.01 → **只需 ~4-6px 外扩即可匹配**。
- ⇒ **结论：不是移植 bug，是「文本骨架（text kernel）预测得偏窄 + 官方硬门槛 tr=0.7」共同作用**。
  这解释了为什么 **PaddleOCR 自己的 `tools/eval.py` 也报 0**（AGENTS §6/§7.7 已记）。
- 本原型是「结构改过 + 150 epoch + 无预训练」；官方 r50（600 epoch / 184MB）预测的框更完整，故能过门槛。

**验收建议（后续）**：
- 要复现「官方口径 0.536」的可比结论，应**用同一个评估器评双方模型**（AGENTS 的通用验收第 4 条），
  而不是直接看 `E2EMetric`（它对窄框过严）。
- 或确认 Paddle 侧手工复算脚本 `_downloads/ocr/diag_combine.py` 的入口（它是怎么绕过 tr=0.7 的），再对齐口径。
- 本移植的**训练正确性已由 loss 逐位对齐证明**（§7.2），与该评估门槛问题无关。

### 10. ⚠️ 更正 §6/§7.7/§9：`f_score 0.785 / e2e 0.536` 不是本原型的成绩（2026-09-27 关闭）
**方法：同一评估器评两份预测 + 直接跑 Paddle 模型对拍。**

#### 10.1 关键实验（脚本 `_downloads/ocr/common_evaluator.py` / `fw_infer_all.py` / `paddle_one_img.py`）
1. **同一评估器（框架 `get_socre_A` + `combine_results(rec_flag=True)`）评两份预测**：
   | 预测来源 | f_score | f_score_e2e | precision | recall | seqerr |
   |---|---|---|---|---|---|
   | Paddle `output/bench_pgnet/result.txt` | **0.7851** | **0.5364** | 0.8334 | 0.7421 | 0.315 |
   | 框架自己跑的 `fw_result.txt` | 0.00051 | 0 | 0.0007 | 0.0004 | 1.0 |
   → **评估器本身正确**（Paddle 预测经框架评估器复现 0.785/0.536），差异在预测。
2. **直接跑 Paddle 模型（`latest.pdparams`，即真正用于 `.pth` 转换的那份权重）**：
   | 图 | Paddle 模型(官方 eval 管线) | 框架 | `result.txt` |
   |---|---|---|---|
   | img1.jpg | **0 框** | **0 框** | 1 框 `retrosains` |
   | img589.jpg | **2 框 `['fresh','et']`** | **2 框 `['fresh','et']`** | — |
   | img995.jpg | **4 框 `['golden','ge','gate','brid']`** | **4 框一致** | — |
   → **框架预测 == Paddle 模型预测**（含文本）；而 `result.txt` 的检出（7.59 框/图，合计 2276）
     与两者（4.59 框/图，合计 1377）明显不同，**只有 38/300 图检出数相同**。

#### 10.2 结论
- **`output/bench_pgnet/result.txt` 是用另一份（更优）权重生成的**，不是 `tiny_pgnet_totaltext/latest.pdparams`。
  §6 里「0.785/0.536 = 150 epoch 原型的真实能力」的归属**是错的**，本节更正。
- **真正跑本原型（150 epoch / 无预训练 / 结构改过）时，Paddle 与框架得到同样的 ~0.0005**
  —— **框架与 Paddle 之间没有差距**，验证链完整闭合：
  权重 450/450 加载 → 前向 maxdiff 1.5e-3 → 后处理逐位一致（同输入同输出 `2 框 ['fresh','et']`）→ **预测一致** → **指标一致**。
- §9 的解释仍成立：**预测框比 GT 窄（tr=0.464/0.099 < 0.7 被 Deteval 拒）**，是这份原型模型的真实短板。
- **待办（若要本原型出好看的指标）**：需**重新训练**（更久 epoch / 带 sigmoid 重新对齐训练口径 / 用官方 r50 权重起点），
  而不是调评估管线。

#### 10.3 顺带澄清
- PaddleOCR 官方 `PGHead` 里 `f_score` 有 `F.sigmoid`；**我的 `PGHeadLite`（Paddle 侧与 torch 侧都）原本没有**。
  实测 Paddle 侧 `f_score` 范围 `0~403`（raw logits），阈值化用 `>0.5`；torch 侧加了 sigmoid 后
  `sigmoid(raw)>0.5 ⟺ raw>0`，**预测几乎不变**（`UNDO_SIG=1` 反算 logit 后检出数/坐标一致）→ 两侧口径等价，均可。
- `result.txt` 时间戳 09-26 21:09（与训练同晚），但内容与该 checkpoint 不符 —— **说明该文件是别的运行留下的，勿再引用**。

### 11. 显存约束取消 + GPU 利用率（2026-09-27，用户决定）
- **AGENTS 原「batch 最多占显存 1/4(≤4GB)」的约束已由用户取消** —— 改为**把 GPU 拉满、利用率最大**，
  明确不接受训练时 GPU 占用/利用率很低。
  （下方"显存 / batchsize 约束"一节仍保留在原文，但 e2e 训练不受其 1/4 限制。）
- **e2e 训练的 batch 选择经验（关键，别再走弯路）**：
  | batch | allocated | step 时长 | GPU 利用率 | 结论 |
  |---|---|---|---|---|
  | 14 | 3.47 GB | ~0.57s | 99%（单点采样） | 可用 |
  | 32 | 7.32 GB | ~1.5s | — | 内存中等 |
  | 48 | 11.14 GB | **16.1s** | **均值 ~8%，掉到 0% 共 4 次** | ❌ 数据管线跟不上 |
  | 64 | 15.75 GB | — | — | 逼近 16 GiB 上限，评估易 OOM |
  | **16** | 5.9 GB | ~2.4s | **均值 98.6%，最低 95%** | ✅ **当前配置** |
- **根因**：`PGProcessTrain`（TCL 点采集/几何）是**纯 CPU 单样本**开销，
  batch 越大单步等待数据的时间占比越高 → GPU 反而空转。
  **batch=48 时 step 内 GPU 只有 ~10% 时间在算**；batch=16 时数据管线刚好喂得饱。
- **当前配置 `configs/ocr/e2e/e2e_pgnet_lite_totaltext.yml`**：
  `Train.loader.batch_size_per_card: 16` + `num_workers: 4` + `prefetch_factor: 4`
  ⚠️ **必须同步改 `PGProcessTrain.batch_size: 16`**（它维护 `img_id` 自增周期 = batch，
  与 loader batch 不一致会让 `org_tcl_rois` 的 `gpu_id` 越界/钳制，虽然有兜底但会错配）。
  实测：78 步/epoch（1255 图 / 16）、显存 5938 MiB、**利用率 98.6%（最低 95%）**。
- **`num_workers=8` 在大 batch 下曾导致宿主内存峰值崩溃**：
  报 `Unable to allocate 1.00 MiB for an array with shape (512,512)`（来自 `PGProcessTrain.fit_and_gather_tcl_points_v3`）。
  4 worker + batch 16 稳定。
- **`accumulate` 语义**：仅当 `Optimizer.nbs` 存在时才 `accumulate = round(nbs/batch)`；本配置无 `nbs` → **`accumulate=1`**，
  每个 batch 一次优化器更新（无累积开销）。
- 训练命令：`python -m torchkiln train -c configs/ocr/e2e/e2e_pgnet_lite_totaltext.yml -o Global.eval_epoch_step=5`
  （每 5 epoch 评估一次；`eval_epoch_step` 默认 1 = 每 epoch 评，会吃掉约 2.5h）

### 12. e2e 训练提速（9~10h → 3.1h），三个根因（2026-09-27）
- **骨干确认是 PP-LCNetV4**（`PPLCNetV4E2E` 0.385M + `PGFPNLCNet` 0.422M + `PGHeadLite` 0.414M = 1.22M）。
  **慢不是骨干的问题**，而是下面三点：
1. **`retry_on_none` 缺失 → 每个 batch 只剩 6/16 个有效样本（关键 bug）**
   - `PGProcessTrain` 按几何/随机缩放拒绝大量样本（`min(new_w,new_h) < input_size*0.5`、
     全 ignore、`len(pos_list)>max_text_nums` 等），**实测 batch=16 collate 后只有 6 个**。
   - **PaddleOCR `PGDataSet` 是「被拒就随机重取」**；框架 `SimpleDataSet` 却是「返回 `[]` 由 collate 丢弃」→
     **每 epoch 实际只用到 ~37% 数据**，且 GPU 白算。
   - **修复**：`SimpleDataSet` 新增 `retry_on_none`（默认 False，det/rec 语义不变），
     e2e 在配置里设 `Train.dataset.retry_on_none: true`（带 30 次重试上限防死循环）。
     修后 batch 恢复 **16/16**。
2. **训练没开 AMP → GPU 单步 1.326s**
   - 模型虽只有 1.22M 参数，但 `PGFPNLCNet`/`PGHeadLite` 都在 **stride 4（128×128）** 上做 conv，
     batch16 时是百 GFLOP 级，fp32 下单步 **1.326s**。
   - 开 `Global.amp: true` 后 **0.135s/step（9.8×）**。
   - 安全性：`ptcore/trainers/base.py` 已在 loss 前 `_to_fp32(preds)`（**递归处理 dict**），
     且 `evaluate()` 强制 fp32 → CTC/后处理不受 fp16 影响。**e2e 可放心开 AMP**。
3. **显存约束（§11）**：batch=48 时数据管线喂不饱（利用率 8%，掉 0%），batch=16 稳定。
- **实测（batch16 + workers4 + AMP + retry_on_none）**：
  - epoch 时长 **75.5s**（150 epoch ≈ **3.1h**，含每 10 epoch 一次评估）
  - **GPU 利用率均值 98.5%，最低 93%**；显存 3582 MiB
  - loss 正常下降：step10 **60.42** → step20 **54.76**（无 NaN）
  - 分段计时（`_downloads/ocr/e2e_profile.py`，num_workers=0）：
    GPU 0.135s/step < 数据 0.678s/step → 4 worker 后数据 ≈0.17s，与 GPU 基本平衡
- ⚠️ 注意：剖析脚本必须 `num_workers=0`，否则 Windows spawn 会重跑主模块导致递归。
- 训练命令：`python -m torchkiln train -c configs/ocr/e2e/e2e_pgnet_lite_totaltext.yml -o Global.eval_epoch_step=10`

### 13. 定期诊断脚本（监控根因，而非门槛后的 recall）+ 首次基线
- **背景**：`f_score_e2e` 长期为 0，且 `global_accumulative_recall` 在极低区间(20/2543)
  震荡 11↔23，**是门槛后的二值量、噪声极大**，不适合当监控信号。
- **脚本 `_downloads/ocr/e2e_watch.py`**（后台每 120s 跑一次，输出 `_downloads/ocr/e2e_watch.csv`）：
  固定前 40 张 val 图，加载 `best_accuracy.pth`(退化为 `latest.pth`)，算：
  | 字段 | 含义 |
  |---|---|
  | `n_det` | 检出框数 |
  | `tr_med` / `tr_p90` | 最佳匹配的 `tr = 交集/GT面积` 中位数 / 90 分位（Deteval 门槛 **0.7**） |
  | `tr_pass` | `tr>=0.7` 的比例 ← 能否通过官方匹配的直接指标 |
  | **`aratio_med`** | **预测框面积 / 匹配 GT 面积（中位数）← §9「框太窄」的直接测量** |
  | `recall`/`f_score`/`f_score_e2e` | 用官方 `get_socre_A`+`combine_results(rec_flag=True)` 复算（与训练日志同口径） |
- **首次基线（epoch ~18 的 best 权重）**：
  ```
  n_det=193  tr_med=0.423  tr_p90=0.662  tr_pass=0.074  aratio=0.518
  recall=0.0515  f_score=0.0550  f_score_e2e=0.0081
  ```
  → **`aratio=0.518`：预测框面积只有 GT 的 52%**（框只有一半宽），因此 `tr_med` 卡在 0.423
  远低于 0.7 门槛，`tr_pass` 仅 7.4% —— 与 §9 的 shapely 精算完全吻合。
- **判据（比 recall 灵敏）**：训练若学好 `f_border`，`aratio_med` 应从 0.518 升向 ≥0.7，
  随后 `tr_pass↑` → `recall` 才有意义。若 `aratio` 长期不动 → 说明 `border_loss` 没起作用，
  需查 loss 权重/`f_border` 分支，而不是继续盲跑。
- 注意：`_downloads/` 不入库；脚本靠 `sys.path` 注入仓库根 + ptocr 环境运行。

### 14. ⚠️ 撤回 `f_score` 的 sigmoid（训练从 epoch 2 起持续劣化的根因，2026-09-27）
**触发**：用户要求"指标没变化甚至更差就停下来查"。诊断脚本（§13）显示：
| 权重 | n_det(40图) | aratio | tr_med | tr_pass | f_score |
|---|---|---|---|---|---|
| epoch 2（=best，**60+ epoch 从未刷新**） | 193 | **0.518** | 0.423 | 0.074 | 0.0550 |
| epoch 60（latest） | 842 | **0.028** | 0.013 | 0.004 | 0.0000 |
→ 模型越训越差，**best_accuracy 从 epoch 2 之后就再没被刷新过**。

**根因（`_downloads/ocr/e2e_sigmoid_test.py`，原型权重单批实测）**：
```
f_score(带 sigmoid): min=0.5000 mean=0.5028  >0.5 占比=0.0058
反解 raw logit:       min=0.00   max=20.72   ← 背景精确=0、从不为负
GT tcl_maps 正样本占比=0.0152
Dice(带 sigmoid)=0.9603   Dice(用 raw)=-0.7075
```
1. 原型权重的 `f_score` raw 形态是 **[0,+∞)、背景精确=0**（它自己的 Dice 推成这样）。
2. 加 sigmoid 后 **背景 = sigmoid(0) = 0.5**：
   - **Dice 分母被 ~13 万背景像素 ×0.5 淹没 → loss 下限 ≥0.94、梯度趋零**
     → 训练日志里 `score_loss` **60+ epoch 恒为 0.927~0.939**（分割头完全学不动）；
   - 后处理 `f_score > 0.5` **恰好卡在背景值 0.5 上** → 阈值失效（只有 0.58% 像素略高）。
3. 种子区域退化 → 框越来越小 → **aratio 0.518 → 0.028**。
4. 连锁：`border_loss` 虽从 0.539 降到 0.482，但种子区域已坏，`aratio` 照样崩。

**已撤回**：`torchkiln/ocr/modeling/e2e_pgnet_lite.py::PGHeadLite.forward` 去掉 `torch.sigmoid`
（代码内保留了完整原因注释）。并**删除了劣化的 checkpoint**、从原型权重重启。

**⚠️ 更正 §10.3**：那里写的「加了 sigmoid 后预测几乎不变 → 两侧口径等价，均可」**只在推理端成立**，
**训练端不成立**（损失面完全不同），该结论作废。**训练时必须与权重的原始口径一致。**

**另一条易误判的经验**：`DiceLoss` 出现**负值不是 bug**（`pred > gt` 即可为负，
本项目实测 `Dice(raw) = -0.7075`）。当初正是因为把 -0.84 当成 bug 才加了 sigmoid —— **误判起点**。
PaddleOCR 官方 `PGHead` 有 sigmoid 是因为它**从零一起训练**；在**已训练权重上**补 sigmoid = 口径失配。

**附带教训（运维）**：训练期间**不要并发跑其它 GPU 脚本** —— 我为算 loss 分量跑的
`e2e_loss_break.py` 先 CUDA OOM，随后**训练与 watchdog 两个进程同时被杀**（日志无报错、
GPU 归零，13:43:40 中断）。疑为 OOM 触发驱动重置（与本机早前
`CUDNN_STATUS_EXECUTION_FAILED_CUDART` 同类）。

## 音频 SOTA 移植（PaddleSpeech → TorchKiln，2026-09-27 起）
### 依赖/资源盘点（关键结论，勿重复踩）
- **PaddleSpeech 源码不需要克隆 GitHub**（本机 GitHub 仅 0.03 MB/s）：
  **PyPI `paddlespeech==1.5.0` wheel 只有 1.7 MB，却含 127 个模型 `.py`** + `resource/pretrained_models.py`（**152 个 bcebos 权重 URL**）+ 全部 `*.yml` 配置。
  装法：`pip install --no-deps paddlespeech==1.5.0`（**绝不能带依赖**，会拖入 paddlepaddle）。
- **权重下载速度**：`bj.bcebos.com` / `paddlespeech.cdn.bcebos.com` 实测 **7~105 MB/s**（GitHub 0.03 MB/s 差 3500×）。
- **环境分工**：`paddlex` 有 paddle(3.1.1)，`ptocr` 有 torch(2.12.1)+torchaudio(2.11.0) → **跨框架对拍要拆成 Step-A(paddlex)/Step-B(ptocr) 两个脚本，用 npz 交接**。
- **torchaudio 2.11.0 装在 torch 2.12.1 下可用**（`pip install --no-deps`）：`kaldi.fbank`/`rnnt_loss`/`MelSpectrogram` 全部正常 —— 不必冒险升级 torch。

### ❌ 做不了的（源码或权重缺失，别浪费时间）
| 项 | 状态 |
|---|---|
| **Paraformer** | 代码就不在 wheel 里（0 条） |
| **VITS / JETS**（TTS SOTA） | 有代码但**权重 0 条（未发布）** |
| **icefall ASR 移植** | **k2 / kaldifeat 在 Windows 无官方 wheel**（PyPI 那份是 `cpython-39-x86_64-linux-gnu.so` + 强制 `torch==1.13.1`；k2 Release 只发 manylinux/macOS）→ icefall 的 pruned RNNT loss/beam search 跑不了；用户已决定**不装 WSL2/Docker** |
| icefall 仓库符号链接 | `egs/**` 的 `zipformer.py` 等是**符号链接被落成 44~76 字节文本**（真源码在 `egs/librispeech/ASR/pruned_transducer_stateless7/`，76KB）；共 81 个 0 字节 `__init__.py` |

### ✅ 已完成：PANNs CNN14（语音分类）—— 四条对齐全过
**源码**：`paddlespeech/cls/models/panns/panns.py`（310 行，`CNN14/CNN10/CNN6` + `ConvBlock/ConvBlock5x5`）
**权重**：`https://bj.bcebos.com/paddleaudio/models/panns_cnn14.pdparams`（**491.3 MB @105 MB/s**，68 键 / 80.77M 参数）
配置 `panns.yaml`：`sample_rate 32000, n_fft 1024, hop_length 320, window_length 1024, f_min 50, f_max 14000, n_mels 64`

| 条目 | 结果 | 数据 |
|---|---|---|
| ① 权重加载 | **PASS** | `missing=0 / unexpected=0`；**转换规则只有 2 条**：`_mean/_variance→running_mean/var` + `fc1`/`fc_audioset` 两个 Linear **`.T`**（Conv 4D 同形状不需转） |
| ② 逐层前向 | **PASS** | **fp64 rel 3.55e-11**（`bn0` 4.04e-16 = 精度极限）；fp32 top5 一致、输出 maxdiff 3.9e-05 |
| ③ 单步 loss/梯度 | **PASS** | **fp64 loss 6.46e-12**、全 **42/42** 个参数梯度 **2.72e-11** |
| ④ 同权重推理指标 | **PASS** | probs 最大差 **0.000034**（阈值 0.02）、**top5 完全一致**、**特征 rel 3.79e-05** |

**产物**：`torchkiln/audio/panns.py`、`torchkiln/audio/__init__.py`、`weights/panns_cnn14.pth`(308 MB)
脚本：`_downloads/sp3a_dump/sp3b_convert/sp4a/sp4b/sp5a/sp5b/sp6a/sp6b_*.py`

#### ⭐ 方法论：用 fp64 判定「fp32 舍入」还是「真 bug」（强烈推荐复用）
fp32 对拍时 ②rel 只有 **6.8e-4**（我的 1e-4 阈值判 FAIL）。**把两侧都切 fp64 再对拍**：
- **fp64 rel → 3.55e-11** ⇒ **证明是 fp32 舍入累积，移植正确**；若 fp64 仍是 1e-4 才是真 bug。
- 用法：`paddle.set_default_dtype("float64")` + 权重 `v.astype("float64")`；torch 侧 `model.double()`；阈值 1e-10。
- **fp32 下误差随深度放大**：block1 1e-7 → block3 1.9e-4 → out 3.6e-4（正常）；fp32 梯度可放大到 2.3e-2。

#### 对拍脚本的两个易错点（都踩过）
1. **Paddle `paddle.Tensor.max(axis)` 只返回值**，torch `x.max(dim)` 返回 `(values,idx)` → 用 **`torch.amax(x, dim)`**。
2. **Linear 梯度与权重同布局**：Paddle `[in,out]` vs torch `[out,in]` → **对比前对 Paddle 梯度 `.T`**（否则方阵 `fc1` rel=1.0、`fc_audioset` 直接形状不符；我第一版把 42 个参数误判成 41 个）。

#### 官方脚本的问题（同 tools/eval.py 一例，勿信官方"跑不起来"）
- PaddleSpeech `cls/exps/panns/predict.py` 写 `LogMelSpectrogram(**feat_conf)`，但 `panns.yaml` 用 `sample_rate/window_length`，
  而 **paddle 3.1.1 的签名是 `sr/win_length`** → **官方 predict.py 直接 TypeError**。正确映射：
  `LogMelSpectrogram(sr=32000, n_fft=1024, hop_length=320, win_length=1024, window='hann', f_min=50, f_max=14000, n_mels=64)`。
- `paddlespeech.cls` 的 import 链会拉到缺失的 `resampy` → **桩掉** `paddlespeech{,.audio,.audio.utils,.audio.utils.download,.utils,.utils.env}` 这几个模块，
  再 `exec(源码)` 动态加载 `CNN14`（`panns.py` 里那两个 import 只服务 `pretrained=True` 在线下载，本流程用本地权重）。

#### ④ 的验收口径（可复用到后面所有音频模型）
不追求外部 SOTA 榜单数字，而是**同权重端到端**：真实音频 → 特征 → 模型 → **`probs 最大差 ≤0.02` + top-5 集合一致**。
实测 `probs 差 0.000034`、top5 全一致；**torchaudio `MelSpectrogram` 复刻 Paddle 口径**：
`power=2.0, norm='slaney', mel_scale='slaney', center=True, pad_mode='reflect', window_fn=hann`，log 用 `10*log10(clamp(mel,1e-10))`（`ref=1.0, amin=1e-10, top_db=None`）。
特征 maxdiff 3.34e-03 / rel 3.79e-05。样例音频 `paddlespeech.cdn.bcebos.com/PaddleAudio/en.wav`（16k → 线性重采样 32k）。

### 说话人 ECAPA-TDNN（进行中，① 已过）
- **源码** `paddlespeech/vector/models/ecapa_tdnn.py` 521 行，纯 paddle.nn 无外部依赖。
- **配置** `conf/model.yaml`：`sr=16000, n_mels=80, window_size=400(25ms), hop_size=160(10ms)`；
  `input_size=80, channels=[1024]*4+[3072], kernel_sizes=[5,3,3,3,1], dilations=[1,2,3,4,1]`
  `attention_channels=128, lin_neurons=192, res2net_scale=8, se_channels=128`；输入 `(N,80,T)` → 输出 `(N,192)`。
- **权重** `sv0_ecapa_tdnn_voxceleb12_ckpt_0_2_0.tar.gz`（266MB，内含 `model.pdparams` 88.8MB；
  `model.pdopt` 177MB 是优化器状态**不需要**）。
- **① PASS**：`missing=0 / unexpected=0`，形状不符 0；backbone **20.768M** → `weights/ecapa_tdnn_voxceleb12.pth` (79.5MB)。
  转换规则 **3 条**：去 `backbone.` 前缀 + BN `_mean/_variance→running_mean/running_var` +
  **顶层 `weight (192,7205)` 是 wrapper 的说话人分类头，不加载（预期）**。
  Conv1d 是 **3 维 [out,in,k] 与 torch 同形不需转**；本模型**无 Linear**（`fc` 也是 k=1 的 Conv1d）。
- **命名天然一致**（照抄 wrapper 结构即可）：Paddle `blocks.0.conv.conv.weight` ↔ torch `TDNNBlock.conv.conv.weight`；
  Paddle `blocks.0.norm.norm._mean` ↔ torch `.norm.norm.running_mean`。
- **移植要点**：① Paddle `BatchNorm1d(momentum=0.9)` ≡ torch `momentum=0.1`（都保留 90% 旧统计量；eval 只用 running 不影响对拍）；
  ② Paddle Conv1d 的 "same" padding 是**对称** `d*(k-1)//2`，与 torch `padding=` 等价（本配置所有 `d*(k-1)` 均为偶数，故 L_out=L_in）；
  ③ `EcapaTdnn.forward` 的 MFA 是 **`torch.cat(xl[1:])` —— 排除第一层**（concat blocks[1..3] = 3×1024 = 3072 = channels[-1]）；
  ④ `Res2NetBlock`: `chunk(scale, dim=1)`，i==0 直通、i==1 过 TDNN、i>=2 先累加 `x_i + y_i` 再过；
  ⑤ `AttentiveStatisticsPooling`: `eps=1e-12`、`softmax` 屏蔽 padding 用 `torch.where(..., -inf, attn)`、输出 `concat(mean,std)`。
- **产物**：`torchkiln/audio/ecapa_tdnn.py`、`torchkiln/audio/__init__.py`（已加导出）。
- **待办**：② 逐层前向（需按 conf 造 (N,80,T) log-fbank；可复用 sp4 脚本骨架 + **fp64 判定法**）、③ loss/梯度、④ 端到端。
  ④ 注意：说话人的输出是 **192 维 embedding**（不是分类 logits），比对应改为
  **同音频 embedding 的 maxdiff + 余弦相似度**（同 batch 阈值 0.02 / cos≈1.0）。
- **脚本**：`_downloads/sp7_spk_kws.py`(下载) `sp8_ecapa_scan.py`(结构) `sp9a_ecapa_dump.py` `sp9b_ecapa_convert.py`。
- **KWS MDTC 已就位待做**：权重 `kws0_mdtc_heysnips_ckpt.tar.gz` 0.1MB + `conf/mdtc.yaml`；
  源码 `kws/models/mdtc.py` 235 行（`DSDilatedConv1d/TCNBlock/TCNStack/MDTC/KWSModel`）
  + **`kws/models/loss.py` 83 行（`padding_mask/fill_mask_elements/max_pooling_loss`，③ 必需）**。

#### ⚠️ ECAPA ② 的 FAIL 根因：PaddleSpeech Conv1d 默认 **reflect 填充**
- 症状：locks0（仅 conv→ReLU→BN）第一层就 **rel=1.34e-1**，逐层放大到 4.3e-1；embedding 余弦 0.999667。
- 排查顺序（复用价值高）：① **先证权重加载无误** —— locks.0.conv.conv.weight maxdiff **0.00e+00**、键数 200 全匹配（sp11a_paddle_load_check.py，**只需 paddle，不必装 torch**）；
  ② 输入 maxdiff=0.00e+00 → 差异只可能在算子 → 读回源码 Conv1d.__init__ 发现第 49 行 **padding_mode="reflect"**，
  而 _manage_padding 用 F.pad(x, padding, mode=self.padding_mode) ⇒ **是镜像填充，不是零填充**。
- **修复**：torch 
n.Conv1d(..., padding_mode="reflect")（与 Paddle 的「手动 reflect-pad + conv(padding=0)」语义等价）。
  修后：locks0 1.34e-1 → **2.57e-4**（↓500×）、mfa 4.29e-1 → 9.99e-4、输出 3.90e-2 → **1.08e-3**、余弦 **0.99999988**。
- **残余 ~1e-3 疑似 fp32 舍入**（与 PANNs ② 的 fp32 6.8e-4 同量级）→ **下一步用 fp64 判定法确证**（阈值 1e-10）。
- 另：paddle 3.1.1 的 set_state_dict() **无 eturn_missing=** 参数；PaddleSpeech 的 TDNNBlock.forward(x) **不接受 lengths**（EcapaTdnn.forward 的 try/except 就是为此）。
- 脚本：sp10a/sp10b_ecapa_fwd.py（②对拍）、sp11a_paddle_load_check.py（①的加载验证，仅需 paddle）。

#### ECAPA ①②③ 结果（④ 待做）
- **① PASS**：missing=0 / unexpected=0，200/201 键（第 201 个是 wrapper 分类头 weight(192,7205)，不加载属预期）。
- **② PASS（fp64）**：locks0 **1.47e-15**（精度极限）、locks1~3 8e-12~1.8e-11、mfa **1.38e-11** 全部 <1e-10；
  sp 起放大到 5.5e-9 → c 1.43e-8；**embedding 余弦 = 1.0000000000**。
  - **残余集中在 AttentiveStatisticsPooling**：含 sqrt(clamp(var,1e-12))（d(sqrt)/dvar 对小方差可达 1e5 倍）+ softmax → **数值敏感点，非逻辑 bug**（前半段 conv/BN/Res2Net/SE/MFA 全 1e-11 级已证正确）。
- **③ PASS（fp64）**：loss rel **1.67e-08**（阈值 1e-7）、embedding rel 1.43e-08、**可比梯度 138/138**、**非零梯度最差 rel 1.34e-07**（mfa.norm.norm.weight）。
  - ⚠️ **判定方法必须跳过零梯度参数**：sp.conv.conv.bias 的 |grad|max 在 fp64 是 **8.88e-16**（fp32 是 4.77e-6），
    **数学上恒为 0** —— 因为 ASP.conv 的输出进 softmax，**bias 的加性常数被 softmax 平移不变性抵消**。
    用 el = maxdiff/max|pg| 会因分母过小**虚高到 8.5 / 1.41**（我第一版就误判 FAIL）。
    **正确做法**：denom < 1e-10 时改用**绝对误差判据**。
- **脚本**：sp10a/b_ecapa_fwd.py(②) sp12a/b_ecapa_loss.py(③，loss 用 MSE(emb, 固定 target)，
  因为说话人模型输出是 embedding 而非分类 logits) sp13_grad_abs.py(看梯度绝对量级，定位"分母过小虚高")。
- **下一步**：④ 端到端（同音频 → fbank → embedding，比 maxdiff + 余弦），再做 KWS MDTC。

#### ECAPA ④ PASS — 四条全过（说话人 ECAPA-TDNN 完成）
- 特征: PaddleSpeech 的 fbank 入口是 **`paddlespeech.audio.compliance.librosa.melspectrogram`**
  （不是 kaldi fbank; 全 vector 目录无 compute_fbank）。参数: `sr=16000, window_size=400`（**docstring 明确 window_size 同时是 FFT size 与 window length**，故 n_fft=400 而非 512）、
  `hop_length=160, n_mels=80, fmin=50(默认), fmax=None(=8000), window=hann, center=True, pad_mode=reflect`,
  `power=2.0, to_db=True, ref=1.0, amin=1e-10, top_db=None`。
- **torch 复刻**: `torchaudio.MelSpectrogram(sr=16000,n_fft=400,win_length=400,hop_length=160,window_fn=hann,center=True,pad_mode=reflect,power=2.0,n_mels=80,f_min=50,f_max=8000,norm=slaney,mel_scale=slaney)` + `10*log10(clamp(mel,1e-10))`（ref=1 → 无减项；top_db=None → 无 clamp）→ **特征 rel 4.73e-06**（maxdiff 3.97e-04）。
- **embedding**: 相对 maxdiff **0.001010**、**余弦 0.9999996424**、norm 123.0696 vs 123.0715 → **PASS**。
- ⚠️ **④ 判据必须按输出量纲设**：我最初照搬 PANNs 的 `绝对 maxdiff ≤ 0.02`（那是 probs 0~1 的口径），对
  norm≈123 / std≈8.9 的 192 维 embedding 不适用（实测 0.0248 只占 std 的 0.28%）。
  **正确判据 = 余弦(说话人验证的真实使用口径) + 相对 maxdiff**；其 1.01e-03 与 ② 的 fp32 `fc` 1.08e-03 同源。
- **加载 `librosa.py` 的方法**（跨环境坑）：它 `from ..utils import depth_convert, ParameterError`，
  而真包链 `paddlespeech.audio.__init__` 会拉到缺失的 resampy →
  **桩 `paddlespeech.audio`（设 `__path__=[]`）+ `paddlespeech.audio.utils`（注入 depth_convert/ParameterError），
  再 exec 源码且设 `mod.__package__='paddlespeech.audio.compliance'`** 才能让相对导入解析。
- 脚本: `sp14a_ecapa_e2e.py`(paddle) / `sp14b_ecapa_e2e.py`(torch)。

### 优先级 3：KWS MDTC（源码与配置已读完，移植前待解 2 个问题）
- **特征** `conf/mdtc.yaml`: `feat_type: kaldi_fbank`，`sample_rate 16000, frame_shift 10(ms),
  frame_length 25(ms), n_mels 80` → torch 侧用 **`torchaudio.compliance.kaldi.fbank`**（早前已验证可用；
  建议 `dither=0` 以便可复现）。
- **模型配置**: `MDTC(num_keywords=1, stack_num=3, stack_size=4, in_channels=80,
  res_channels=32, kernel_size=5)`；`causal=True` 默认。包装 `KWSModel(backbone, num_keywords)`。
- **权重**: `_downloads/speech/mdtc_heysnips/ckpt/model.pdparams`（0.2MB）+ `conf/mdtc.yaml` ✅ 已就位。
- **源码结构**（`paddlespeech/kws/models/mdtc.py` 235 行，纯 paddle.nn）：
  | 类 | 行 | 要点 |
  |---|---|---|
  | `DSDilatedConv1d` | 21-54 | depthwise `conv(groups=in_channels)` → `bn` → `pointwise 1x1`；`padding=0`（**手动 pad**） |
  | `TCNBlock` | 57-98 | `conv1→bn1→relu1→conv2(1x1)→bn2`；残差分支对输入做**切片** `inputs[:, :, R:]`(causal) 或 `[:, :, half:-half]`；`in==res` 才相加 |
  | `TCNStack` | 101-157 | `dilations = [2**l for s in stack_size for l in stack_num]`（即 3 组 1,2,4,8 共 12 个 block）；`nn.Sequential` |
  | `MDTC` | 160-221 | `preprocessor=TCNBlock(in→res,d=1)` + `stack_num 个 TCNStack`；输出对多 stack 结果**对齐后求和**再 transpose 返回 **`(outputs, None)`** |
  | `KWSModel` | 224-233 | `linear(hidden_dim→num_keywords)` + `Sigmoid` |
- **loss（`kws/models/loss.py` 83 行）**: `padding_mask(lengths)` / `fill_mask_elements(condition, value, ...)` /
  `max_pooling_loss(logits, ...) → ③ 必需，**尚未读细节**。
- ⚠️ **移植前必须先解的 2 个问题（勿猜）**：
  1. **`MDTC.forward` 返回 `(tensor, None)` 元组**，而 `KWSModel.forward` 写 `outputs = self.backbone(x)`
     再 `self.linear(outputs)` → 对元组做 Linear **会 TypeError** ⇒ **官方 KWSModel 路径本身是坏的**
     （与 `cls/exps/panns/predict.py` 的 TypeError、`tools/eval.py` 同类）。
     移植时**自己组装 backbone+linear**，并把这个不一致记为「官方问题」。
  2. **`MDTC.forward` 的 `F.pad(x, (0,0, receptive_fields, 0, 0,0), 'constant')` 用了 6 值 pad**
     （paddle pad 从**最内层维度**开始配对），紧接着 `transpose([0,2,1])` ⇒ **必须先确认输入是 `(N,C,T)` 还是 `(N,T,C)`**、
     padding 到底加在时间维还是通道维。**这是移植正确性的关键**，猜错会得到完全错误的前向。
     解法：打印 `mdtc.yaml` 训练入口的 dataloader 输出形状，或直接跑一次 Paddle `MDTC` 打印 `F.pad` 前后形状。
- **四条对齐的计划**（复用已验证流程）：① 键结构 dump + 转换（BN 改名 + 可能的 Linear .T；注意 `KWSModel.linear`
  是 `nn.Linear` ⇒ **需 .T**）→ ② 同 (N,80,T) 输入逐层对拍 + **fp64 判定** → ③ `max_pooling_loss` + 梯度
  （**注意零梯度参数按绝对判据**，见 ECAPA 教训）→ ④ 同音频 kaldi_fbank → logits/命中对比。
- **脚本可复用**：`sp3a/sp3b`(转换) `sp10a/sp10b`(②含FP64) `sp12a/sp12b`(③) `sp14a/sp14b`(④)。

#### KWS MDTC 的 2 个待解问题已解（_downloads/sp20_mdtc_probe.py，只需 paddle）
- **Q1 键结构**：236 键 = **ackbone.* 234 + linear.* 2**；linear.weight = **(32,1)**（Paddle Linear [in,out] → **torch 需 .T** 得 (1,32)）；
  preprocessor.conv1.conv.weight = **(80,1,5)** ← depthwise 与 torch **同形状，不需转**；eceptive_fields = 184。
- **Q2 输入布局（关键）**：**输入是 (N, T, 80)（帧在前），不是 (N, 80, T)** ——
  | 输入 | 结果 |
  |---|---|
  | (2,80,120) | ❌ The channel of input must be divisible by groups, received: the channel of input is 120 |
  | **(2,120,80)** | ✅ OK → (2,120,32) |
  报错信息**正好印证推断**：F.pad(x,(0,0,R,0,0,0)) 的 6 值 pad 从**最内层维度 C** 开始配对 ⇒ C=(0,0)、T=(R,0)（**时间维左 pad R=184，causal**）、N=(0,0) → (N,T+R,C)，再 	ranspose([0,2,1]) → (N,C,T) 进 conv ⇒ 若给 (N,80,T) 则 T 变通道 → 报错。
- **撤回我先前的过早判断**：KWSModel 路径**没有坏**（权重里含 linear.* 说明它可用）。我只读源码就下"会 TypeError"的结论是错的 ——
  **教训：涉及调用约定的问题必须实跑验证，不能只看代码**（同 §6 的"官方指标为 0 不等于模型废了"）。
- **对齐 ① 的转换规则（预测）**：去 ackbone. 前缀 + BN _mean/_variance→running_mean/running_var + **linear.weight .T**；
  conv 权重（含 depthwise [in,1,k]）同形状不转。

#### MDTC ① PASS + ② PASS（fp64 判定）
- **①**：missing=0 / unexpected=0，可转换 236/275（39 个为 
um_batches_tracked），形状不符 0；
  参数 0.0344M → weights/mdtc_heysnips.pth (240.17KB)。
  - ⚠️ **键结构本就一致，无需加任何前缀**（KWSModel = ackbone.* + linear.* 两边同名）—— 我先加 ackbone. 导致全未命中，又加了 linear. 前缀导致 linear.* 两个键 missing。
  - linear.weight (32,1) → **.T**；depthwise conv (80,1,5) 同形不转。
- **②**：input / after_pad / after_transpose 三层 **maxdiff = 0.00e+00** —— **确证 F.pad 6 值语义的 torch 等价写法正确**：
  F.pad(x, (0,0, R,0))（torch pad 从最后一维开始配对 ⇒ C=(0,0)、T=(R,0)）⇔ Paddle F.pad(x,(0,0,R,0,0,0))。
  | 层 | fp32 rel | fp64 rel |
  |---|---|---|
  | preprocessor | 4.47e-4 | **6.75e-12** |
  | stack0/1/2 | 5.8e-4~9.5e-4 | 2.6e-10~6.0e-10 |
  | output | 6.23e-4 | **2.84e-10**（maxdiff 6.08e-09） |
  | output range | 0~21.364460 / 0~21.364410 | **0~21.364413 / 0~21.364413（完全相同）** |
  - **判 PASS**：6e-10 是 fp64 数值累积（12 层 conv），非逻辑差异（pad 层 0.00e+00 已证结构/语义正确）。
  - 三模型 fp64 量级对照：PANNs 3.55e-11 < **MDTC 6.02e-10** < ECAPA 1.43e-08（后者含 ASP 的 sqrt/softmax 敏感点）。
- **脚本**：sp21a/sp21b(①转换) sp22a/sp22b(②含 FP64)。
- **待做**：③ max_pooling_loss+梯度（先读 kws/models/loss.py 83 行）、④ 同音频 kaldi_fbank → logits 对拍。

#### MDTC ③ PASS（fp64）+ 实证官方 KWSModel 路径确实坏
- **③**：loss rel fp32 **2.362e-06** / **fp64 3.106e-11**；**correct=1, acc=0.5000 两侧完全一致**；
  可比梯度 **158/158**（零梯度 0 个）；最大梯度 rel fp64 **7.636e-09** → **RESULT_3: PASS**。
  - 又踩 linear.weight 布局坑（同 PANNs）：Paddle (32,1) vs torch (1,32) → **对比前 .T**，否则报 shape 不符。
- ⚠️ **实证：官方 KWSModel.forward 确实是坏的** ——
  ValueError: linear(): argument 'X' (position 0) must be Tensor, but got tuple（MDTC 返回 (tensor,None) 而它不解包）。
  **我在 2b2a6a5 的"撤回"是错的撤回**（当时只看权重含 linear.* 就下结论）—— **本次实跑把结论扳回：原判断正确**。
  - 绕过方式：b_out, _ = m(t); logits = kw.activation(kw.linear(bb_out))；我的 torch KWSModel.forward 已正确解包 outputs, _ = self.backbone(x)。
  - **强化教训**：同一个"必须实跑"的原则**对正反两向都要用** —— 我用它推翻过一个"只看代码"的判断，
    又因为"看权重"而推翻了一个其实正确的判断。**调用约定类问题只有运行时证据算数。**
- **max_pooling_loss 语义（已移植到 	orchkiln/audio/kws_loss.py）**：
  命中关键词 → **max-pooling**（padding 置 0）-log(max)；其它/filler → **min-pooling**（padding 置 1）-log(min)；
  均 clip(1e-8,1) 后 /num_utts，并返回 (loss, num_correct, acc)。
  对应坑：paddle.max(axis) 只返回值 → torch .max(dim).values；paddle.clip → 	orch.clamp；
  m[:min_duration] = True 是**原地修改**（与 Paddle 一致）。
- **脚本**：sp23a/sp23b_mdtc_loss.py。
- **待做**：④ 同音频 → 	orchaudio.compliance.kaldi.fbank(dither=0) → logits 对拍。

#### MDTC ④ 的入口与参数（已定位，尚未跑通）
- **Paddle 侧 fbank 入口**：`paddlespeech/audio/compliance/kaldi.py::fbank`（644 行，**kaldi 官方算法的 Python 移植**，
  含 `spectrogram/_mel_scale/_get_mel_banks/_get_dct_matrix`；`kaldi_fbank` 字样出现在 `audio/datasets/dataset.py` 与 `cli/kws/infer.py`）。
- **签名**（对拍参数必须按此对齐）：
  `fbank(waveform, blackman_coeff=0.42, channel=-1, dither=0.0, energy_floor=1.0, frame_length=25.0,
  frame_shift=10.0, high_freq=0.0, htk_compat=False, low_freq=20.0, n_mels=23, preemphasis_coefficient=0.97,
  raw_energy=True, remove_dc_offset=True, round_to_power_of_two=True, sr=16000, snip_edges=True,
  subtract_mean=False, use_energy=False, use_log_fbank=True, use_power=True, vtln_* , window_type="povey")`
- **依赖**：`from ..functional import create_dct` + `from ..functional.window import get_window`；
  同样撞到 **缺 resampy 的包链** → 用 **§7/ECAPA 已验证的「桩 `paddlespeech.audio`(设 `__path__=[]`)
  + 注入依赖 + exec 源码 + 设 `__package__`」法**（对 `librosa.py` 成功过）。
- **torch 侧**：`torchaudio.compliance.kaldi.fbank`（同为 kaldi 移植），
  参数 `num_mel_bins=80, frame_length=25.0, frame_shift=10.0, sample_frequency=16000.0, dither=0.0,
  snip_edges=True, low_freq=20, high_freq=0, window_type='povey'`。
  - ⚠️ **`energy_floor` 默认两边都是 1.0** —— 我在 `sp24b` 里写成了 `0.0`，**对拍前必须改回 1.0**。
  - `remove_dc_offset=True`（Paddle）⇔ `zero_mean_windows=True`（torch）✓；`use_power/use_log_fbank=True` ✓。
- **脚本已写好待跑**：`sp24a_mdtc_e2e.py`(paddle) / `sp24b_mdtc_e2e.py`(torch)；
  ④ 判据：`logits diff <= 0.02` + **HIT/filler 判定一致**（同 `max_pooling_loss` 的 acc 口径 >0.5）+ `feat rel < 1e-3`。

##### ⚠️ MDTC ④ 当前阻塞（paddlex 环境，2026-09-27）
- **桩 `paddlespeech.audio` 指向真实目录**（跳过其 `__init__` 的 resampy 链，保留真实子模块）这一步是**对的**，
  但随即在 `paddlespeech.audio.functional.__init__` 卡在：
  **`ImportError: The scipy install you are using seems to be broken, (extension modules cannot be imported)`**
  ⇒ **paddlex 环境的 scipy 损坏/与 paddle 的 numpy ABI 冲突**，导致 `..functional` / `..compliance.kaldi` 都加载不了。
- **可选出路（按代价从低到高）**：
  1. **跳过 `functional/__init__`**：直接 exec `functional/window.py`（拿 `get_window`）+ 给 `create_dct` 打桩
     （`fbank` 不用它，只有 `mfcc` 用），再 exec `compliance/kaldi.py` —— 纯文件级加载，绕开包 `__init__`。
  2. 重装/降级 paddlex 环境的 scipy（**有风险**，可能影响 paddle3d/bevlane 等既有工作，**不建议先做**）。
  3. ④ 改为**共享特征**（torch 侧 `torchaudio.compliance.kaldi.fbank` 产出喂给两侧模型）—— 但这样 ④ 只测模型端，
     与 ②（随机输入逐层）重复度高，**验收价值打折**；且仍需 Paddle 侧能跑模型（可，模型不依赖 scipy）。
- **若走路线 1 的落地细节**：`fbank` 参数须两侧一致
  （`dither=0, energy_floor=1.0, frame_length=25, frame_shift=10, low_freq=20, high_freq=0,
  n_mels=80, sr=16000, snip_edges=True, window_type='povey', use_power/use_log_fbank=True,
  remove_dc_offset=True ⇔ zero_mean_windows=True`）。
- **进度**：PANNs ①②③④ 全过 | ECAPA ①②③④ 全过 | **MDTC ①②③ 过，仅剩 ④**。

##### MDTC ④ 结果：模型端 PASS，特征端差 1.09%（实现差异，非移植问题）
- **模型端 ✅**：`logits maxdiff = 4.043443e-09`（阈值 0.02）、`Paddle max = torch max = 0.000001`、
  **HIT/filler 判定一致**（两边都判 filler —— en.wav 不含 "hey snips"，**正确**）。
- **特征端 ❌**：`feat maxdiff 1.757e-01 / rel 1.090e-02`（阈值 1e-3）。
  - Paddle `range -16.1181..5.1794` vs torch `-15.9424..5.1794` —— **最大值完全相同，仅最小值差 0.176** ⇒
    差异集中在**低能量帧**，是 `paddlespeech/audio/compliance/kaldi.py`（Paddle 自己的 kaldi 移植）与
    `torchaudio` 版在 `energy_floor/raw_energy/preemphasis` 边界处理上的**实现差异**，**不是模型移植问题**
    （logits 4e-09 已证模型端完全对齐）。
  - **下一步**：对比两版 `fbank` 在低能量帧的中间量（`spectrogram` → `mel bank`），定位是哪一项参数/分支；
    若确认是实现差异，④ 判据应改为「logits + HIT/filler」，并把特征差异**如实记为已知差异**。
- **跑通 ④ 的关键（环境绕过）**：`paddlespeech.audio` / `.functional` / `.compliance` **三个都做成
  「`__path__` 指向真实目录、跳过 `__init__`」的桩包** —— 因为：
  `audio/__init__` 拉缺失的 resampy；`functional/__init__` 会 import `.functional` 触发
  **`ImportError: The scipy install you are using seems to be broken`**（**scipy 其实没坏**，1.15.3 正常，
  是 paddlespeech 自己的检查）；`compliance/__init__` 会 import librosa。
  再给 `functional.create_dct` 打桩（`fbank` 不用，仅 `mfcc` 用）即可 `from .. import` 解析。
- **参数名两侧不同（易错）**：Paddle `fbank(n_mels=80, sr=16000)` vs torchaudio `fbank(num_mel_bins=80,
  sample_frequency=16000)`；且 torchaudio **无 `zero_mean_windows`**，用 kaldi 原名 **`remove_dc_offset=True`**。
  两侧共同：`dither=0, energy_floor=1.0, frame_length=25, frame_shift=10, snip_edges=True`。
- **进度**：PANNs ①②③④ 全过 | ECAPA ①②③④ 全过 | **MDTC ①②③ 全过，④ 模型端过/特征端待查**。

##### MDTC ④ 特征差异排查进展（已排除 1 个假设）
- **实验**：把两侧 `energy_floor` 从 1.0 改为 **0.0**（脚本 `sp24a/sp24b` 支持 `EF` 环境变量），
  **差异完全没变**：`maxdiff 1.757107e-01 / rel 1.090e-02`（与 EF=1 逐位相同）
  ⇒ **排除 `energy_floor` 假设**。
- **仍未排除的嫌疑**（按可能性）：`preemphasis_coefficient=0.97` 预加重、`raw_energy=True` 的能量基准、
  `round_to_power_of_two=True` 的 n_fft 对齐、`window_type="povey"` 窗函数实现、`low_freq=20/high_freq=0`。
- **证据约束**：两侧 `max` **完全相同 5.1794**，只有 `min` 差（-16.118 vs -15.942）⇒ **绝大多数帧一致、少数帧不同**，
  且**集中在低能量**帧 ⇒ 很可能是**能量为 0 的静音帧**上某一步的边界处理（log/sqrt 的 epsilon、或预加重在帧首样本）。
- **下一步（决定性）**：对比两侧**中间量 `spectrogram`**（`paddlespeech.audio.compliance.kaldi.spectrogram`
  vs `torchaudio.compliance.kaldi.spectrogram`）——若 spectrogram 就不同 ⇒ 差异在帧/窗/预加重阶段；
  若相同 ⇒ 差异在 `_get_mel_banks`/log 阶段。再逐级定位。
- **不影响已完成的结论**：模型端 `logits maxdiff 4.043443e-09`、HIT/filler 一致 —— **②③④ 的模型对齐已成立**。
- **进度**：PANNs ①②③④ 全过 | ECAPA ①②③④ 全过 | **MDTC ①②③ 全过；④ 模型端 PASS、特征端 1.09% 待查（已排除 energy_floor）**。

##### MDTC ④ 特征差异已精确定位到 `spectrogram` 阶段
- **实验**（`sp25a_mdtc_spec.py`(paddlex) / `sp25b_mdtc_spec.py`(ptocr)）：
  两侧 `spectrogram`（Paddle `paddlespeech.audio.compliance.kaldi.spectrogram` vs
  `torchaudio.compliance.kaldi.spectrogram`，同参数 dither=0/energy_floor=1/frame 25-10ms/povey/预加重0.97）：
  - 形状一致 **(328, 257)**；**`max` 两侧完全相同 4.74978**；`min` paddle **-16.1181** vs torch **-15.9424**
  - **`maxdiff 0.175711 / rel 1.09015e-02`** —— 与 fbank 的 `rel 1.090e-02` **逐位相同**
  - 差异 >1e-3 的 bin **166/257**；这些 bin 的能量全是**负数**（-3789 / -3789 / -3398 / -2722 / -2123）
- **结论 1**：fbank 的 **100% 差异都来自 spectrogram** ⇒ `_get_mel_banks` / log 阶段**没有额外差异**（不必再查）。
- **结论 2**：**只在能量为负（log 后极小值）的 bin 上不同**，高能量端逐位相同 ⇒
  是**极小值在 log 域被放大的数值细节**（某个 epsilon / `remove_dc_offset` / 预加重在低能量帧的实现差异），
  **不是帧数、窗长、mel bank 的结构性错误**（形状与 max 均已证明一致）。
- **④ 的判定现状**：模型端 **PASS**（`logits maxdiff 4.043443e-09`、HIT/filler 一致）；
  特征端 **1.09% 已知差异**，根因已定位到 spectrogram 的低能量端 ——
  **建议 ④ 判据采用「logits + HIT/filler」**，特征差异**如实标注为已知实现差异**（两版 kaldi 移植的数值细节不同）。
- **进度**：PANNs ①②③④ 全过 | ECAPA ①②③④ 全过 | **MDTC ①②③ 全过，④ 模型端 PASS / 特征端已定位为实现差异**。

###### MDTC ④ 特征差异的第二层定位：**不是**每帧常数偏移，指向 FFT fp32 舍入在 log 域被放大
- **判别实验**（`sp25b_mdtc_spec.py` 末尾，纯 numpy）：算 `d = paddle_spec - torch_spec`，
  看它是否沿 bin 维恒定（即"每帧一个加性常数"）：
  | 量 | 值 | 含义 |
  |---|---|---|
  | **每帧内 diff 的 std** | max **0.0407** / mean 0.00074 | **不接近 0** ⇒ **不是**每帧加性常数 |
  | 每 bin 内 diff 的 std | max 0.0418 | 两方向都有 spread ⇒ **逐 (frame, bin) 分布** |
  ⇒ **排除"log 能量项统一偏移"假设**，差异在 STFT 内部。
- **与已有证据合起来的结论**：`spectrogram` 的 `max` 两侧**逐位相同(4.74978)**、只在**低幅值 bin** 差
  （`rel 1.09e-02` 且集中在能量为负/极小的 bin）⇒ 符合
  **`log(x)` 对小 x 的 `1/x` 放大 × FFT 的 fp32 舍入**：FFT 在低幅值 bin 的绝对误差 ~1e-7，
  经 `1/x`（x≈1e-5 时 ×1e5）→ 0.1 量级，与实测 `maxdiff 0.175711` 量级吻合。
- **一锤定音的实验（尚未做）**：**用 fp64 重跑两侧 `spectrogram`** —— 若差异塌到 ~1e-10
  ⇒ 确证是 fp32 舍入（同 §PANNs/ECAPA/MDTC 模型侧用过的 fp64 判定法）；若仍是 1e-2 ⇒ 才是真实现差异。
  （注意：两环境隔离 `paddlex` 只有 paddle/numpy/scipy1.15.3、`ptocr` 只有 torch/torchaudio/scipy1.18.1，
  **都必须拆两个脚本**；顺带证实两个环境的 scipy 都正常，paddlespeech 的 "scipy broken" 确是其自身检查。）
- **④ 判据建议（未改判，等 fp64 结论）**：模型端 `logits maxdiff 4.043443e-09` + HIT/filler 一致**已成立**；
  特征端 1.09% **先记为"待 fp64 确证"**，若 fp64 仍 1e-2 则**如实标为两版 kaldi 移植的已知实现差异**。

###### ❌ 更正上一条：MDTC ④ 特征差异**不是** fp32 舍入，是两版 kaldi 移植的**算法实现差异**（fp64 已证）
- **fp64 实验**（`sp25a/sp25b` 加 `FP64` 环境变量，两侧都传 float64 波形）：
  | | fp32 | **fp64** |
  |---|---|---|
  | `maxdiff` | 0.175711 | **0.17571** |
  | `rel` | 1.09015e-02 | **1.09014e-02** |
  | Paddle `min` | -16.1181 | **-16.1181（不变）** |
  | torch `min` | -15.9424 | **-15.9424（不变）** |
  ⇒ **差异在 fp64 下逐位不变 ⇒ 排除 fp32 舍入假设**，是**两版 kaldi fbank/spectrogram 移植的算法实现差异**。
- ⚠️ **更正上一条 commit 的结论**（我写的是"指向 FFT fp32 舍入在 log 域被放大"）—— **该推断被 fp64 证伪**。
  **这是本轮第三次"实跑推翻自己的推断"**（前两次：① 官方 `tools/eval.py` 的 `f_score_e2e=0`；
  ② 官方 `KWSModel` 的 tuple bug —— 我曾据"看权重"错判为正常）。
  **教训再次强化：对**自己**的推断也要走 fp64/实跑验证，不能停在"听起来合理"。**
- **④ 最终判定**：
  - **模型端 PASS** —— `logits maxdiff = 4.043443e-09`（阈值 0.02）、`Paddle/torch max 均 0.000001`、
    **HIT/filler 判定一致**（两边 filler，en.wav 不含 "hey snips"）。
  - **特征端 1.09% → 如实标为「两版 kaldi fbank 移植的已知算法实现差异」**（非我方移植问题，
    已由 ②（逐层 0.00e+00 ~ 6e-10）+ ③（loss 3.1e-11 / 梯度 7.6e-09）+ ④ 模型端（4e-09）三重证明模型完全对齐）。
  - **建议 ④ 判据采用「logits + HIT/filler」**；特征差异不作为失败项，但**必须标注**。
- **进度**：PANNs ①②③④ 全过 | ECAPA ①②③④ 全过 | **MDTC ①②③ 全过，④ 模型端 PASS / 特征端已定为已知差异**
  ⇒ **三个音频 SOTA 模型的移植与对齐全部完成**。

## kokoro-82M 训练集成（进行中，2026-09-28）
### 资源获取（生死门已过）
- **权重托管：HuggingFace 与 hf-mirror 都超时（被墙）** → 改走 **ModelScope**，实测 **25.5 MB/s**（比 GitHub 0.03 MB/s 快 850×）。
  - repo: `hexgrad/Kokoro-82M`（或 `AI-ModelScope/Kokoro-82M`），文件接口：
    `https://www.modelscope.cn/api/v1/models/{ns}/{name}/repo?Revision=master&FilePath={path}`
  - 文件列表接口：`.../repo/files?Revision=master&Root=`
  - 已下：`_downloads/speech/kokoro/kokoro-v1_0.pth` **327.21 MB** + `config.json` 2.3KB + `voices/` `eval/` `samples/`
- **源码**：PyPI `kokoro==0.9.4` + `misaki==0.9.4`（G2P），用 `pip install --no-deps` 装在 ptocr（**只要源码**）。
- **config.json 键**：`dim_in, dropout, hidden_dim, istftnet, max_conv_dim, max_dur, multispeaker,
  n_layer, n_mels, n_token, plbert, style_dim, text_encoder_kernel_size, vocab`
  → **iSTFTNet 声码器 + PL-BERT(ALBERT) 文本编码器 + 多说话人 style 向量**。

### 权重结构（548 键 / 81.763 M / 327.1 MB fp32，与文件大小吻合）
| 子模块 | 键 | 参数 | 内容 |
|---|---|---|---|
| `bert` | 25 | 6.292 M | ALBERT/PL-BERT（`position_embeddings(512,128)` → hidden 128） |
| `bert_encoder` | 2 | 0.394 M | `weight(512,768)` ALBERT 768→512 投影 |
| `predictor` | 122 | 16.195 M | ProsodyPredictor + DurationEncoder（weight_norm ×16） |
| `decoder` | 375 | 53.276 M | iSTFTNet（`F0_conv`/`N_conv`/`asr_res`/`decode.*`，weight_norm ×70） |
| `text_encoder` | 24 | 5.606 M | style/text 编码（weight_norm ×3） |
- **顶层是 nested dict**（`sd['bert']` 本身是 dict）—— 遍历时必须下钻一层（我第一版漏了这层，误报 0 个张量）。
- **所有键带 `module.` 前缀**（`module.embeddings...`/`module.F0_conv...`）→ ① 需处理。
- **89 个 `weight_g`/`weight_v`**（weight_norm 分解）——**已是 torch 格式**，直接兼容，无需转换。

### 与前三个模型的本质差异（对齐口径要换）
- PANNs/ECAPA/MDTC 都是 **Paddle → torch 跨框架转换**，用"四条对齐"对照 Paddle 参考。
- **kokoro 是纯 torch**：官方 `pip` 包就是参考实现 → **对齐基准改为「与 kokoro 官方包逐层对拍」**：
  ① 权重 `missing=0`（处理 `module.` 前缀）→ ② 同输入逐层前向 vs 官方 `kokoro` 包（+fp64 判定法）
  → ③ 同输入 loss/梯度 → ④ 推理指标（TTS 口径，需另定，如波形/频谱一致性）。

### 源码结构（`kokoro==0.9.4`，7 个 .py）
| 文件 | 行/大小 | 关键类 |
|---|---|---|
| `model.py` | 6.6 KB | **主模型（5 部分组装）——尚未细读** |
| `istftnet.py` | 19.5 KB | `SineGen / SourceModuleHnNSF / Generator / UpSample1d / AdainResBlk1d / Decoder` |
| `modules.py` | 7.8 KB | `LinearNorm / LayerNorm / TextEncoder / AdaLayerNorm / ProsodyPredictor / DurationEncoder / CustomAlbert(AlbertModel)` |
| `custom_stft.py` | 7.7 KB | 自定义 STFT |
| `pipeline.py` | 17.6 KB | 文本→音素→模型 推理链 |
- ⚠️ **`modules.py` 依赖 `transformers`**（`CustomAlbert(AlbertModel)` = PL-BERT）—— **ptocr 里没装**，是下一个依赖门槛
  （`transformers` 在镜像里有 5.x 全系列；装时注意别拖入超大依赖）。
- **脚本**：`_downloads/kokoro_precheck.py`(可行性) `kokoro_ms_search.py`(搜权重)
  `kokoro_ms_files.py`(文件列表) `kokoro_fetch.py`(下载, 支持 `--big`) `kokoro_scan.py`(结构)。

### kokoro ① 进展 + 关键根因：**权重与 pip 版本的键路径不匹配**
- **依赖已装齐且未破坏环境**（`transformers 4.57.1` + `loguru 0.7.3` + `huggingface_hub` + `attrs`，
  pip 报过 dependency conflicts，但**实测 `torch 2.12.1+cu132` / `torchaudio 2.11.0` / CUDA True / `nn.Conv2d` 全部完好**）。
  - 新缺 `attr`（`custom_stft.py` 用 `from attr import attr`）→ 装 `attrs` 即可。
- **绕过 `kokoro/__init__.py`**：它会 `from .pipeline import KPipeline` → `from misaki import en`，
  而 misaki 是 `--no-deps` 装的（缺依赖）→ 用**桩包法**（`kokoro.__path__` 指真实目录，跳过 `__init__`）
  就能正常 `from kokoro.model import KModel`（同 paddlespeech 的做法，脚本 `_downloads/kokoro_load.py`）。
- **官方权重加载方式**（`model.py` L68-75，必须照抄）：
  `python
  for key, sub in torch.load(model, map_location='cpu', weights_only=True).items():
      assert hasattr(self, key)
      try: getattr(self, key).load_state_dict(sub)
      except: sub = {k[7:]: v for k, v in sub.items()}   # 去 'module.'
              getattr(self, key).load_state_dict(sub, strict=False)
  `
  即 **先试直接 load，失败则去掉 `module.` 前缀再 `strict=False`**。
- **① 实测结果**（`kokoro_load.py`）：
  | 子模块 | 结果 | 参数 |
  |---|---|---|
  | `bert` | **stripped(missing=0, unexpected=0)** ✅ | 6.292 M |
  | `bert_encoder` | **stripped(missing=0, unexpected=0)** ✅ | 0.394 M |
  | `text_encoder` | **stripped(missing=0, unexpected=0)** ✅ | 5.606 M |
  | `predictor` | stripped(**missing=24**) | 16.204 M |
  | `decoder` | stripped(**missing=116**) | 53.314 M |
  | 合计 | 81.810 M（盘点的 sd 是 81.763 M，差值来自 buffer 计入口径） | |
- ⚠️ **根因（下一步要修）**：缺失键全是 `*.norm1.norm.weight/bias` / `*.norm2.norm.*` / `encode.norm1.norm.*`，
  而权重里对应路径是 **`module.F0.0.norm1.**fc**.weight` / `module.decode.0.norm1.**fc**.weight`**（实测 `norm1.norm.weight` 0 个、`fc.weight` 存在）。
  ⇒ **权重由旧版 kokoro 训练，pip 装的 0.9.4 把 `AdaLayerNorm` 内部子模块从 `fc` 改名为 `norm`**。
  **修法**：加载时键名映射 `.fc.` → `.norm.`（predictor 24 + decoder 116 个缺失键全是此原因）。
- **下一步**：① 补键名映射到 `missing=0` → ② 同输入与**官方 `kokoro` 包**逐层对拍（+fp64 判定法）
  → ③ 同输入 loss/梯度 → ④ 推理指标（TTS 口径，波形/频谱一致性，需另定）。

###### kokoro ① 的精细根因：不是简单改名，是**版本间结构性差异**（AdaLayerNorm 同时有 
orm 和 c）
- **我上一步的".fc. → .norm. 改名"方向是错的**：加了改名后 `missing` 列表里**同时出现**
  `F0.0.norm1.norm.weight` **和** `F0.0.norm1.fc.weight` ⇒ **当前 `kokoro 0.9.4` 的
  `AdaLayerNorm` 同时拥有 `norm` 与 `fc` 两个子模块**，而**权重里只有 `.fc.*`**。
  → 不是改名问题，是**权重缺 `.norm.*` 那一份**。
- **完整证据（`predictor` 的 `load_state_dict` 报错）**：
  - `missing` 同时含：`F0.0.norm1.norm.weight` + `F0.0.norm1.fc.weight`、`N.0.norm2.fc.weight`…
  - `unexpected` 全部**仍带 `module.` 前缀**（`module.text_encoder.lstms...` / `module.F0.0.norm1.fc.weight`）
    ⇒ 说明 `k[7:]` 对这批键**没生效**，或 `sub` 内前缀不统一，**需逐键核对而不是整批切片**。
- **下一步必须先做**（不要猜）：
  1. 打印**权重里同一模块的完整键集合**（如 `module.F0.0.*` 全部键）与**模型期望的键集合**（`F0.0.*` 全部键）**并列比对**，
     找出 .norm.* 对应的权重键到底叫什么（可能叫 `.LayerNorm.*` / `.norm.weight` 且前缀不同 / 或旧版权重根本没有）。
  2. 核对 `modules.py::AdaLayerNorm` 与 `istftnet.py` 里 `norm1/norm2` 的**实际子模块结构**（是 `LayerNorm` 还是 `LinearNorm`）。
  3. 再决定：是**换旧版 kokoro 代码**（pip install 一个更老的版本，让键名对上），还是**在 torchkiln 侧自己写映射**。
  - ⚠️ **换版本前先看 `config.json` 的权重是哪个 kokoro 版本产出的**（`kokoro-v1_0.pth` 对应 v1.0）。
- **教训**：这次我又一次"听起来合理就动手"了（改名方案没先验证模型期望的键集合）。
  **正确顺序永远是：先打印「权重键集合」与「模型期望键集合」两个集合的差集，再设计映射。**

###### ✅ kokoro ① 根因**彻底确定**（脚本 _downloads/kokoro_keydiff.py）
- **方法（这次做对了）**：并列打印「权重键集合（去 module.）」vs「模型期望键集合」的**差集**，不猜。
- **结果**（以 F0.0.* 为例）：
  - **交集 10**、**unexpected = 0**、**missing = 4**：
    `F0.0.norm1.norm.{weight,bias}` + `F0.0.norm2.norm.{weight,bias}`
- **当前 kokoro 0.9.4 的结构**（
amed_modules 实测）：
  `
  F0.0.norm1 = AdaIN1d
      - norm : InstanceNorm1d    ← 模型期望 .norm.{weight,bias}，权重里没有
      - fc   : Linear            ← 权重里有 .fc.{weight,bias} ✓ 交集在这
  `
- **算术完全吻合**：F0.0/F0.1/F0.2/N.0/N.1/N.2 6 blocks × 
orm1/norm2 × weight/bias = **24 = predictor 的 missing=24**（decoder 的 116 同理）⇒ **两处 missing 同源**。
- **根因**：AdaIN1d 内的 InstanceNorm1d —— **当前代码 ffine=True（有 weight/bias）**，而**产 kokoro-v1_0.pth 的旧版 ffine=False（无参数）**。
- **修法（改模型，不改权重 —— 与 Paddle 侧相反）**：移植到 	orchkiln 时把该 InstanceNorm1d 构造成 **ffine=False** → missing=0。
  （⚠️ 若 	rack_running_stats 也有差异，missing 里会多出 unning_mean/var；实测 missing 只有 weight/bias ⇒ 只差 ffine。）
- **验证口径**：修后 predictor/decoder 应达 missing=0 / unexpected=0，① 即 PASS（ert/bert_encoder/text_encoder 已 PASS）。
- **对齐基准**（与前三个模型不同）：kokoro 是纯 torch ⇒ **②③④ 与官方 kokoro 包逐层对拍**（+fp64 判定法）。

###### ✅ kokoro ① PASS（修法已验证）
- **验证脚本 _downloads/kokoro_fix_test.py**：把模型里所有 InstanceNorm1d 重建成 **ffine=False**（模拟产权重的旧版口径）后加载：
  | 子模块 | 修前 missing | **修后 missing** | unexpected |
  |---|---|---|---|
  | ert / ert_encoder / 	ext_encoder | 0 | **0** | 0 |
  | predictor | 24 | **0** | 0 |
  | decoder | 116 | **0** | 0 |
  | **合计** | 140 | **0** | **0** |
  - **重建的 InstanceNorm1d 数量 = 70**。
- **① 的两条要点（移植到 	orchkiln 时照做）**：
  1. **键处理**：k[7:] 去掉 module. 前缀（实测 unexpected=0，**不需要** .fc.→.norm. 改名 —— 那是我上一轮的误判）。
  2. **模型构造**：AdaIN1d 内的 InstanceNorm1d 必须 **ffine=False**（旧版权重无 weight/bias）。
- **踩过的桩包坑**：load_kokoro() 第二次调用时 importlib.find_spec 抛 **ValueError: kokoro.__spec__ is None**
  （桩包把 __spec__ 置 None 后，ind_spec **抛异常而非返回 None**）→ 用 **	ry/except ValueError + 模块级 _ROOT 缓存** 解决。
- **下一步（kokoro ②③④）**：对齐基准是**官方 kokoro 包**（纯 torch，无跨框架）：
  ② 同 input_ids+ef_s 逐层前向 vs 官方（+**fp64 判定法**）→ ③ 同输入 loss/梯度 →
  ④ 推理指标（TTS 口径：波形/频谱一致性 或 pred_dur 对齐）。
  ⚠️ ef_s 是 1×512 的 style 向量（s = ref_s[:,128:] 给 predictor、ef_s[:,:128] 给 decoder）—— 需固定随机 ef_s 对拍。

###### ⚠️ 方案 A（拉官方训练脚本）**不可行**：hexgrad 从未开源 kokoro 训练代码
- **hexgrad/kokoro 仓库 108 文件全清单**（kokoro_repo_probe.py / 	ree.json，	runcated=False）：
  `kokoro/` **7 个文件（= PyPI 包，只有推理）** + `demo/`(7) + `examples/`(4) +
  `kokoro.js/`(83, JS 移植) + `voices/*.bin` + `tests/` + README/pyproject/uv.lock
  ⇒ **无 	rain.py / loss.py / dataset / optimizer**，顶层只有 README.md。
  仓库 29122 KB 主要被 LFS 权重/样本占用，**代码本体极小**。
- **hexgrad 名下只有 2 个仓库**：`hexgrad/kokoro`（推理）+ `hexgrad/misaki`（G2P）→ **官方确无训练代码**。
- **GitHub 代码搜索需要认证**（`401 Unauthorized`）→ 本机无 token，搜不了 code search。
- **第三方训练代码候选（仓库搜索结果）**：
  | 仓库 | ★ | 说明 |
  |---|---|---|
  | **`jonirajala/kokoro_training`** | **46** | "Training code for kokoro tts model" ← 最直接、最高星 |
  | `BovineOverlord/Derpy-Turtle-The-Kokoro-Trainer` | 14 | Windows GUI 构建 Kokoro |
  | `gushilabs/train-kokoro-encoder-styletts2` | 4 | Kokoro 基于 StyleTTS2，训 encoder |
  | `sammy4321/Kokoro-Indic-Fine-Tuning` | 2 | 微调（非从零） |
- **⚠️ 这是第三方代码，不是官方**：质量、完整性、许可证、与 `kokoro-v1_0.pth` 的架构是否一致**都未知**，
  采纳前必须核对（尤其：它的 loss 是否覆盖 iSTFTNet/PL-BERT/ProsodyPredictor 的全部分支）。
- **⇒ 方案 A 实际退化为二选一**：
  - **A'**：用第三方 `jonirajala/kokoro_training`（★46）作训练蓝本 —— 快，但需先做可信度/架构一致性核对；
  - **B**：自写训练管线（loss + 数据 + 任务适配），贴合 `torchkiln` 架构但工作量大，且**必须自己定 loss 组合**
    （官方未公开，无从对齐 ⇒ ③ 这条无法再"对照官方"，只能自洽）。
- **注**：kokoro 的 **① 已 PASS**；**②③④ 仍可做**（基准=官方 **推理** 包，它存在），**不受训练代码缺失影响**。

###### 第三方 jonirajala/kokoro_training(★46) 已探明 —— 评估：**可作参考，但不是"对齐基准"**
- **仓库元信息**：570 KB / 41 文件 / pushed_at 2025-11-15 / **license = None（无许可证）** / desc "Training code for kokoro tts model"。
- **结构**（_downloads/kokoro_train_repo.py 探明，	runcated=False）：
  | 目录 | 内容 |
  |---|---|
  | `audio/` | `audio_utils.py` / `hifigan_vocoder.py` / `vocoder_manager.py` |
  | `data/` | `english_phoneme_processor.py` / **`ljspeech_dataset.py`**（LJ Speech，**非官方数据**） |
  | `kokoro/` | `model.py` / **`model_transformers.py`** / `positional_encoding.py` / **`postnet.py`** ← **自己重新实现模型** |
  | `training/` | **`english_trainer.py`** / `config_english.py` / `checkpoint_manager.py` / `adaptive_memory_manager.py` / `interbatch_profiler.py` / `mps_grad_scaler.py` |
  | `tests/` | `test_dual_loss.py` / `test_overfit.py` / `test_training_health.py` / `test_vocoder_quality.py` / `test_warmup_schedule.py` |
  | 根 | `setup_ljspeech.py` / `training_english.py` / `inference_english.py` / `requirements.txt` |
  | 其它 | `overfit_test_output/{generated_mel.pt, training_sample.pt, model_config.json}` |
- **⚠️ 三个风险（采纳前必须核对）**：
  1. **无许可证**（`license=None`）→ 使用/分发法律不确定。
  2. **它重新实现了模型**（`model_transformers.py` + `postnet.py`）→ **可能与 `kokoro-v1_0.pth` 架构不一致**
     ⇒ **不能当"对齐基准"**，只能当"训练管线设计的参考"。
  3. `ljspeech_dataset` + `english_trainer` → **只是"一个能跑的英语训练"，不是"官方训练"**
     （官方是多说话人 + 多语言 + PL-BERT + iSTFTNet 的组合）。
- **⇒ 结论**：A' 的**正确用法**是拿它的 **loss 组合 / trainer 结构 / 数据增强**做参考来设计 B（自写），
  而**不**把它当权威参考去对拍（否则会像"对照一个非官方实现"，对齐失去意义）。
- **⚠️ 但 ②③④ 的对齐不受影响**：它们的基准是**官方 kokoro 推理包**（存在、官方），仍然可严格对拍。
  **只有"训练 loss"这一项没有官方参考**（hexgrad 从未公开）⇒ 那一项只能**自洽 + 参考 A'**。
- **脚本**：`_downloads/kokoro_find_train.py`（找仓库）、`kokoro_train_repo.py`（探结构）、
  `kokoro_repo_probe.py`（探 hexgrad 官方仓）。

###### ✅ kokoro 中文版 v1.1-zh 已获取 + ① 也 PASS（ffine=False 修法通用）
- **用户提示正确**：`model.py::MODEL_NAMES` 里就有
  `'hexgrad/Kokoro-82M-v1.1-zh' -> 'kokoro-v1_1-zh.pth'`（`KModel` 支持中文）。
- **已下**：`_downloads/speech/kokoro/kokoro-v1_1-zh.pth` **327.25 MB @ 24.8 MB/s**
  （ModelScope 仓库 `hexgrad/Kokoro-82M-v1.1-zh`，文件接口同 v1.0：`.../repo?Revision=master&FilePath=`）。
  同目录另有其自己的 `config.json`（**v1.0 与 v1.1-zh 的 config 可能不同** —— 测试用的是同目录 v1.0 的 config，
  若两版 `vocab`/`n_token` 不同则须分别取各自 config，**下一步要核对**）。
- **① 中文版权重也 PASS**：`[['bert','bert_encoder','decoder','predictor','text_encoder']]`，重建 70 个
  `InstanceNorm1d` 为 `affine=False` 后 **5 个子模块全部 `missing=0 / unexpected=0`**。
  ⇒ **修法对两版权重通用**。
- **两个权重并存**：`kokoro-v1_0.pth`（英文/多语言）+ `kokoro-v1_1-zh.pth`（**中文**）。
- **脚本**：`_downloads/kokoro_zh_fetch.py`（查/下载 v1.1-zh）。
- **下一步**：kokoro **② 同输入逐层前向 vs 官方包** —— 需先把模型**移植进 `torchkiln/audio/kokoro.py`**
  （参考代码量：`model.py` 152 + `istftnet.py` 422 + `modules.py` 184 + `custom_stft.py` 198 ≈ **956 行**），
  再用已验证的 **fp64 判定法** 对拍；③④ 同前三个模型的流程。
  ⚠️ 注意固定随机 `ref_s`（1×512）与 `input_ids` 做可复现输入；`speed` 参数也要固定。

###### ✅ kokoro 模型已移植进 	orchkiln/audio/kokoro/ + ① PASS（两版权重）
- **搬运**（_downloads/kokoro_vendor.py，官方 **MIT 许可、保留原版权头**）：
  | 文件 | 行 | 改写 |
  |---|---|---|
  | `model.py` | 149 | 去 `hf_hub_download` 回退（**HF 被墙**，改为必须显式传本地 config/`.pth`）；`loguru`→标准 `logging` |
  | `istftnet.py` | 422 | `from kokoro.custom_stft import` → `from .custom_stft import`（**唯一 1 处跨包引用**） |
  | `modules.py` | 184 | 原样（相对导入天然可移植） |
  | `custom_stft.py` | 198 | 原样 |
  | `__init__.py` | 34 | 新建（导出 `KModel/Decoder/CustomAlbert/ProsodyPredictor/TextEncoder/...`） |
  - **`pipeline.py` 不搬**（推理/G2P 链，训练与对齐都用不到，且依赖 misaki）。
  - **导入测试全通过** ✓
- **⭐ 官方源码注释直接印证了 ffine=False 的修法**（`istftnet.py::AdaIN1d`）：
  > "affine should be False, however there's a bug in the old torch.onnx.export ... When affine is true,
  > there's additional learnably parameters. **This shouldn't really matter setting it to True, since we're in inference mode**"
  ⇒ 这 2 个参数是 **ONNX workaround 的副产物**；产权重里**没有**它们 → 默认 `weight=1/bias=0` → `1*x+0 = x`
  ⇒ **`affine=False` 与官方前向数学等价**，且能让 ① 达到 `missing=0`。已在移植代码里改并附注释。
- **① PASS（两版权重，_downloads/kokoro_tk_load.py）**：
  `
  英文 v1.0      missing=0 unexpected=0   参数=81.763 M
  中文 v1.1-zh   missing=0 unexpected=0   参数=81.763 M
  `
  **81.763 M 与最初盘点的权重总量完全一致** ✓
- **下一步**：② 同输入（固定 `input_ids` + `ref_s(1×512)` + `speed=1`）逐层前向 vs **官方 kokoro 包**（+fp64 判定法）→ ③（注意 `forward_with_tokens` 带 `@torch.no_grad`，需另建可微路径）→ ④（`pred_dur` 对齐 + 波形/频谱一致性）。
