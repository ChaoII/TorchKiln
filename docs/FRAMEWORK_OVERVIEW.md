# TorchKiln 框架总览与统一口径

> 本文档回答四个问题：**① 框架怎么组织（YAML 配置 → 模型结构）**、
> **② 训练/预测/评估/导出四条链路的口径是否一致**、
> **③ 每个模型的结构与公开指标**、**④ 预训练权重的本地/远程清单**。

---

## 一、整体架构（三层 + 一注册表）

```
torchkiln/                     # 模型与任务层（框架核心）
├── nn/                        #   · 网络结构（graph.py 解析 YAML → nn.Module）
│   ├── graph.py               #     parse_model / build_from_arch（YAML→模型）
│   ├── modules.py             #     Conv/C3/C3k2/SPPF/Detect/Segment/Pose/OBB...
│   └── ...
├── tasks/                     #   · 任务适配器（26 个：数据→前向→loss→指标）
├── data/                      #   · 数据集（按任务分文件）
├── cfg/models/<family>/       #   · **模型结构 YAML**（yolo11/yolov8/yolo26/v10/v9/v5/v3/v6/12/plate）
├── models/                    #   · 模型构建入口（det3d/pc_seg/lane_bev/ts/...）
└── cli.py                     #   · 统一命令行（train/val/export/predict/check）

ptcore/                        # 训练核心层（与模型解耦）
├── trainers/base.py           #   · BaseTrainer：训练循环/评估/EMA/AMP/断点续训/早停
├── trainers/<task>.py         #   · 每个任务一个 Trainer 子类
├── optimizer.py / muon.py     #   · 优化器（Adam/AdamW/SGD/MuSGD(Muon)）
├── ema.py                     #   · ModelEMA（threshold / exponential 两种）
└── task.py                    #   · TaskAdapter 基类

configs/                       # 运行配置层（YAML：数据/超参/路径）
└── {yolo,ts,ocr,audio,pc,lane,plate,video,action,attr,local,_parity}/
```

### 1.1 职责划分（关键设计）

| 层 | 管什么 | 不管什么 |
|---|---|---|
| `torchkiln/cfg/models/**/*.yaml` | **模型结构**（backbone/neck/head 的层与通道） | 数据、超参 |
| `configs/**/*.yml` | **数据 + 训练超参 + 路径**（`Architecture` 指向模型族/规模） | 结构细节 |
| `torchkiln/tasks/*.py` | **任务语义**（张量怎么组 batch、loss 怎么算、指标怎么统计） | 结构、循环 |
| `ptcore/trainers/base.py` | **训练/评估循环**（统一：AMP/fp32、EMA、累积、裁剪、早停、续训） | 任务语义 |

### 1.2 模型构建链路（YAML → 网络）

```
configs/xxx.yml  Architecture.{task, scale, ...}
        │
        ▼  build_trainer → get_task(task).build_model(config)
torchkiln/tasks/<task>.py::build_model
        │
        ▼  （det/seg/pose/obb/cls）  build_arch_model(...) 或 graph.build_from_arch(yaml_file, scale)
torchkiln/nn/graph.py::parse_model
        │   · 读 torchkiln/cfg/models/<family>/<model>.yaml
        │   · 按 scale 做 depth/width 缩放；按家族路由头（legacy vs DWConv）
        ▼
nn.Module（参数即结构）
```

**家族头路由（重要）**：`graph.py` 依据 yaml 文件名判断家族 ——
`v3/v5/v8/v9` 为 **legacy 头**（旧式 Conv 头），`yolo11/12/26` 为 **新 DWConv 头**。

### 1.3 任务与模型数量

| 类别 | 任务数 | 说明 |
|---|---|---|
| 2D 视觉 | 6 | `detect` / `segment` / `pose` / `obb` / `classify` / `attribute` |
| 密集预测 | 2 | `depth`（深度）/ `semantic`（语义分割） |
| 3D / 点云 | 2 | `det3d`（CenterPoint-Pillars）/ `pc_seg`（SqueezeSegV3） |
| 车道 | 3 | `lane_bev`（BEV-LaneDet）/ `lane_row` / `lane_seg` |
| 视频/行为 | 2 | `video_cls` / `pose_action` |
| OCR | 3 | `plate_det` / `plate_rec` / `ocr_e2e`（+ `ocr_det`/`ocr_rec`） |
| 音频 | 3 | `panns_cls`（PANNs）/ `kokoro_tts` / （声纹、KWS 见 `torchkiln/audio/`） |
| 时序 | **5** | `ts_forecast` / `ts_anomaly` / `ts_classify` / `ts_embed` / `ts_rul` |
| **合计** | **26 个任务适配器** | `TRAINER_REGISTRY` 已全部注册 |

### 1.4 模型结构 YAML 库存

| 家族目录 | 数量 | 文件 |
|---|---|---|
| `11/` | 7 | yolo11, -cls, -seg, -pose, -obb, -sem, -lane-row |
| `12/` | 5 | yolo12, -cls, -seg, -pose, -obb |
| `26/` | 9 | yolo26, -p2, -p6, -cls, -seg, -pose, -obb, -depth, -sem |
| `v10/` | 7 | yolov10 n/s/m/l/x/b |
| `v9/` | 8 | yolov9 t/s/m/c/e + c-seg/e-seg |
| `v8/` | 12 | yolov8 ± p2/p6 ± ghost, -cls/-seg/-pose/-obb |
| `v5/` | 2 | yolov5, yolov5-p6 |
| `v3/` | 3 | yolov3, -spp, -tiny |
| `v6/` | 1 | yolov6 |
| `plate/` | 1 | yolov5n-0.5（车牌专用） |

---

## 二、四条链路的口径一致性（已逐项核对）

### 2.1 入口

| 链路 | 入口 | 状态 |
|---|---|---|
| 训练 | `tkiln train -c <cfg>` → `tools/train.py` | ✅ |
| 评估 | `tkiln val -c <cfg> [--weights ...]` → `tools/eval.py` | ✅ |
| 预测 | `tkiln predict [task] -c <cfg>` → `tools/infer/predict_*.py` | ✅ |
| 导出 | `tkiln export -c <cfg> --onnx` → `tools/export.py` | ✅ |
| 结构自检 | `tkiln check -c <cfg>` → `tools/check_graph_build.py` | ✅ |
| 非 task 工具 | `tkiln changepoint ...` → `tools/changepoint.py` | ✅ |

`MODES = (train, val, export, predict, check)`，`TASK_ALIASES` 提供中英文/简写别名。

### 2.2 `BaseTrainer` 的统一口径（对所有 26 个任务生效）

| 口径 | 实现 | 为什么要统一 |
|---|---|---|
| **评估强制 fp32** | `autocast(enabled=False)` | fp16 下 yolo26 的框解码/proto einsum 会失真（实测 mask mAP 0.56 vs 0.64） |
| **cudnn.deterministic** | 默认开启（`Global.cudnn_deterministic` 可关） | 否则同权重评估会“抖动”，cls 在 conf 阈值附近翻转 → mAP 低估 |
| **best_metric 方向可配** | `Global.main_indicator_mode: min\|max`（缺省 max） | RMSE/MAE/loss 类指标「越小越好」，原硬编码 `>=` 会把最优当最差 |
| **梯度裁剪阈值可配** | `Optimizer.clip_grad_norm`（缺省 10.0，`null` 关闭） | 对齐 ultralytics=10；对齐 PaddleSpeech 等不裁剪的上游时需关闭 |
| **EMA** | `use_ema` + `ema_decay_type: threshold\|exponential` | 两种上游（ultra=exponential）都要能复刻 |
| **AMP** | `Global.amp`（loss 前 `_to_fp32`，评估强制 fp32） | 加速但不影响数值口径 |
| **梯度累积** | `accumulate = nbs/batch`（读 `Train.loader`） | 有效 batch 决定 LR 相位 |
| **断点续训 / 早停** | `checkpoints` / `Global.patience` | — |
| **两阶段算法钩子** | loss 提供 `train_step` 时**完全接管**每 batch 更新 | USAD/AnomalyTransformer 需要两次 backward+step |
| **按 batch 评估可关** | `eval_batch_step: null` | 只按 epoch 评估（避免重复触发） |
| **loss 必须返回 dict** | trainer 取 `loss_dict["loss"]` | 统一多分量损失的打印 |

### 2.3 「同权重对比」的标准做法（本仓库验收规范）

1. **权重加载** `missing=0 / unexpected=0`（或仅差函数式 DFL）
2. **同输入逐层前向** maxdiff ~1e-5（浮点级用 **fp64 判定法** 区分“舍入”与“真 bug”）
3. **单步 loss / 梯度**：loss 相对误差 <1e-3；梯度仅算子级微差
4. **同权重推理指标**：差 ≤0.02
5. **端到端训练**：终值差 ≤0.02~0.03（含随机性）

> ⚠️ 对比时**必须**：同一评估器、同一 val 清单、同一权重、同一口径记录。

---

## 三、方向 ① 的收尾：RUL 五轮探索总结

| 轮次 | 方向 | 结论 |
|---|---|---|
| ① 特征工程 | `roll_mean` 有效（-5%）；全量 102 维过拟合 |
| ② 多工况 | FD002/003/004 全覆盖 + **按工况分组归一化** |
| ③ 结构升级 | 新增 BiLSTM+Attn / Transformer；**定位瓶颈=数据量** |
| ④ 迁移学习 | **如实否证**（域差异 > 数据量收益）；副产物：**统一 24 列更优** |
| ⑤ 口径溯源 | **文献 12~18 = 筛选易样本口径**；同口径下本框架**不落后**（RUL≤30 时 RMSE 7.6） |

**最终成绩（三层报法，推荐）**：

| 子集 | 台数 | 全量 RMSE | 近失效段(RUL≤30) | NASA Score | 常数基线 |
|---|---|---|---|---|---|
| FD001 | 100 | 25.44（TRF 24.46） | **7.63** | 13.9 | 40.07 |
| FD002 | 260 | **21.74** | **9.00** | 8.2 | 53.78 |
| FD003 | 100 | **19.28** | — | — | 41.40 |
| FD004 | 249 | 30.10 | — | — | 54.52 |
