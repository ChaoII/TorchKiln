# FAQ / 环境与坑位

## 环境

**Q:需要哪几个 conda 环境?**
| 环境 | 路径 | 用途 |
|---|---|---|
| `ptocr` | `C:\ProgramData\miniconda3\envs\ptocr` | **训练/评估/推理/导出**(torch 2.12+cu132)|
| `paddlex` | `...\envs\paddlex` | 只用于**导出 Paddle 权重**(`tools/convert/dump_*.py`)|
| `ultralytics` | `...\envs\ultralytics` | 只用于**导出上游 `.pt`**(`tools/convert/dump_ultralytics.py`)|

**Q:网络受限?**
GitHub / huggingface 在本机不可达;`paddle-model-ecology.bj.bcebos.com`(Paddle 官方权重)
与 ModelScope 可达。所以:预训练权重走 Paddle 站点,上游 YOLO `.pt` 需手动放到 `_downloads/upstream/`。

**Q:`~/.pytorchx/ocr/pretrained` 是空的?**
那不是 bug:加载器会优先复用仓库内 `~/.pytorchocr/pretrained/` 的同名权重。

## Windows / 数据加载

**Q:`WinError 1455 页面文件太小` / `OpenBLAS error` / 训练卡死在 DataLoader?**
`num_workers` 太多。每个 worker 约 3.5~4GB 提交内存(本机上限 ~63.8GB),
总数 ≥10 会耗尽。建议 `Train.dataset.loader.num_workers` 设 0~4、`Eval` 设 0。
复现问题时可临时 `$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"`。

**Q:PowerShell 里 `-o` 传列表/引号被吞?**
把命令写进 `.ps1`/`.py` 再执行;内联 python 的引号在 PowerShell 里容易被吞掉。

## 复现性

**Q:OCR 训练结果不可复现?**
OCR 数据管线的 `seed` 必须为 `None`(Paddle 语义),否则增广序列被固定导致不收敛/不一致。

**Q:YOLO 图模型训练报 `KeyError: 'Optimizer'`?**
配置必须带 `Optimizer` 段(这是有意的,防止静默用错学习率)。
`configs/yolo/*_graph.yml` 已自动生成 `Adam + Cosine`。

## 模型 / 权重

**Q:上游 YAML 全部支持吗?**
`pytorchx/cfg/models/**` 共 60 个 YAML(可训练 53 个)全部支持:
`python tools/check_graph_build.py` → `53 configs: 53 OK, 0 FAIL`。
不支持/跳过:`rt-detr`(RT-DETR)、`yoloe-*`、`yolov8-world*`、`sam*`、`*-cls-resnet*`(不支持训练或不在范围内)。

**Q:加载上游 `.pt` 报 `WeightsUnpicklingError` / 找不到类?**
上游 `.pt` pickle 了整个模型对象。用 `tools/convert/dump_ultralytics.py`(ultralytics 环境)先转成纯张量
state_dict;车牌的 `.pt`/`.pth` 用 `tools/convert/convert_plate_weights.py`(内置 Unpickler shim,无需上游代码)。

**Q:加载权重后 `missing/unexpected` 不为 0?**
正常现象:按**名字+形状**匹配,分类头(类别数不同)与少量结构差异块会被跳过,日志会打印统计。
`Head.reg_max` 与权重不一致(例如权重是 DFL-free 的 4 通道而配置写了 16)时回归头会被整体跳过。

**Q:上游 seg/obb/pose 权重无法下载?**
是的(GitHub/HF 不可达),这些只做了**结构级**对齐;det 用用户 `.pt` 做过权重级验证。

## 车牌 / 属性

**Q:车牌检测的 ONNX 和 PyTorch 结果不一致?**
用户 `test_data` 里的 `yolov5plate.onnx` 是**更早版权重**(首层卷积即不同),与 `weights/plate_detect.pt` 不是同一次训练;
我们的实现与上游 PyTorch 代码 **max|diff| = 0**。要部署最新权重请自行导出 ONNX。

**Q:车牌识别是 LPRNet 吗?**
不是。它是 `plateNet.py/colorNet.py` 的 `myNet_ocr_color`:CNN + CTC(78 类)+ 颜色头(5 类),
**没有 RNN**,也不是 LPRNet 的 small-basic-block 结构。仓库里虽有 `LPRNet.py`,但权重不是它。

**Q:属性识别的 loss 特别大(200~300)?**
`MultiLabelLoss(size_sum=True)` 是“逐类求和 × 逐样本平均”,量纲天然是类别数量级;
再叠加 `ratio2weight` 的指数权重。用假数据 demo 时尤其明显,正常训练会下降。

**Q:属性识别前向数值与 Paddle 有 ~0.8 偏差?**
结构/键/形状 146/146 全对齐、权重逐元素相同、top-5 属性 4/5 一致;
偏差已定位到 `blocks3.0.pw_conv` 的裸 1×1 卷积(权重 bit 相同、输入几乎相同),
属待收尾项(见 `docs/MODEL_ZOO.md` §5)。

## 其它

**Q:ONNX 里有 NMS 节点?**
端到端模型(`Head.end2end: true`,v10/v26)导出的 ONNX **不含 NMS**;非端到端模型需要外部 NMS。

**Q:训练中间产物在哪?**
`Global.save_model_dir`:`train.log`(与 PaddleOCR 同格式)、`config.yml`(实际生效配置)、
`best_accuracy.pth`、`latest.pth`。
