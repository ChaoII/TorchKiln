# tools 索引

所有脚本都在**仓库根目录**下执行(`python tools/xxx.py ...`)。环境:`ptocr`(默认),
标注 `paddlex` / `ultralytics` 的必须在对应环境跑。

## 训练闭环

| 脚本 | 作用 |
|---|---|
| `train.py` | 训练入口:`-c <cfg>` 必填;`-o A.B=v` 覆盖任意配置;按 `model_family` 自动分派 Trainer;多卡用 `torchrun` + `-o Global.device=gpu:0,1` |
| `eval.py` | 独立评估:`-c <cfg> --weights <pth>`(强制 `use_ema=false`) |
| `export.py` | 导出 `inference.pth / inference.yml / model.pt / model.onnx / model_slim.onnx`(`--onnx --slim`;slim 依赖已有 onnx) |
| `infer/predict_det.py` / `infer/predict_rec.py` | OCR 检测 / 识别推理;参数 **`--input` 单图**(无 `--image_dir`) |
| `infer/predict_yolo.py` | YOLO 七任务推理(按 `task` 自动可视化);同样 **`--input`** |
| `download_pretrained.py` | 预下载 OCR 官方权重到 `~/.torchkiln/pretrained/` |
| `make_demo_data.py` | 生成占位图 + 标签(仅 smoke 自检) |
| `make_format_examples.py` | 各任务标签/配置片段模板 → `datasets/_format_examples/` |
| `convert/dataset_format.py` | 格式互转(当前仅 `yolo_det` ↔ `ocr_det`) |

> 数据集清单/下载入口是 CLI 子命令 **`tkiln data list|get`**(`torchkiln/datasets.py`),**没有** `tools/download_dataset.py`。
> 完整命令链见 [`docs/TRAINING.md`](../docs/TRAINING.md)。

## 自检与诊断

| 脚本 | 作用 | 期望输出 |
|---|---|---|
| `smoke_all.py` | 全量自检:每配置建模型 + 取 1 batch + 1 步训练 + 1 步评估 | `76 OK, 0 FAIL` |
| `check_graph_build.py` | 只建图 + 前向,校验全部 YAML 图配置(秒级,不需要数据) | `53 configs: 53 OK, 0 FAIL` |
| `check_plate_models.py` | 车牌:上游权重张量对齐 + 与出厂 ONNX 的数值对齐 | 检测 `500/500`、识别 `86/86`,识别 `max|diff| ~1e-5` |
| `test_build_all.py` / `debug_*.py` | 历史调试脚本(建图/序列/对齐排查),保留备查 |

## 配置生成

| 脚本 | 作用 |
|---|---|
| `gen_configs.py` | 由官方配置批量生成 `configs/{det,rec}/*.yml` |
| `make_paddle_cfg.py` | 由本平台配置反向生成 Paddle 风格 cfg(对比用) |

## 权重转换(`tools/convert/`)

| 脚本 | 运行环境 | 作用 |
|---|---|---|
| `dump_all.py` / `dump_paddle_sd.py` | **paddlex** | `.pdparams` → numpy pickle(OCR) |
| `convert_all.py` / `convert_det.py` | ptocr | pickle → torch `.pth`(OCR 全量 / 单个) |
| `dump_ultralytics.py` | **ultralytics** | 上游 YOLO `.pt` → 纯张量 state_dict(附 yaml) |
| `convert_plate_weights.py` | ptocr | 车牌上游 `.pt`/`.pth` → torch `.pth`(内置 Unpickler,无需上游代码) |
| `dump_attribute_paddle.py` | **paddlex** | 属性 `.pdparams` → pickle |
| `convert_attribute_weights.py` | ptocr | 属性 pickle → torch `.pth`(`_mean→running_mean`、Linear 转置) |
| `parity_*.py` / `paddle_forward.py` | 两者配合 | Paddle ↔ PyTorch 逐层数值对比(定位差异用) |

## 约定

* 脚本一律使用**仓库绝对/相对根路径**运行,不要 `cd tools`。
* 自检脚本互不依赖,可单独跑;改完代码**至少跑 `smoke_all.py`**。
* 输出统一写到 `output/`;临时/诊断产物写到 `_downloads/`(已在 `.gitignore` 外,注意不要入库大文件)。
