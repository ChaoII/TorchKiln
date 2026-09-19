# 检测(Detect)模型与 ultralytics 全量对齐计划

> 目标：把**本仓库(pytorchx)的检测(detect)任务**在**推理**与**训练**两方面与原版 ultralytics 完全对齐。
> 范围：权重库 `\\tsclient\D\项目资料\ultralytics_models` 中所有**检测**权重的「版本 × 规模」。

## 一、范围与清单

权重库可做权重对齐的检测版本（detect 任务，纯 `.pt`）：

| 版本 | 目录 | 规模(检测) | 说明 |
|---|---|---|---|
| yolo11 | `yolo11` | n/s/m/l/x | C3k2 系 |
| yolo12 | `yolo12` | n/s/m/l/x | |
| yolo26 | `yolo26` | n/s/m/l/x | |
| yolov10 | `yolov10` | n/s/m/l/x | |
| yolov8 | `yolov8` | n/s/m/l/x | C2f 系 |
| yolov9 | `yolov9` | t/s/m/c/e | |
| yolov5 | `yolov5` | n/s/m/l/x (+6u 版) | |
| yolov3u | `yolov3u` | u / sppu / tinyu | |

> ⚠️ **v4 / v6 / v7 无官方检测权重**（v6 仅仓库有 `yolov6.yaml` 无权重；v4/v7 库中不存在），无法做权重对齐，暂不纳入。

**规模合计**：yolo11/12/26/v10/v8 各 5 ×5 + v9(5) + v5(5+5 个 6u) + v3u(3) ≈ **48 个检测权重**。

## 二、对齐流程（每个模型×规模）

统一「三步验证」，与已完成的 pose/seg/classification 流程一致：

1. **权重加载对齐**：dump ultralytics 权重 → `build_arch_model(...,"detect")` 加载，要求 **missing=0 / unexpected=0**（检测头的各版本 `Detect` 结构须与 ultra 一致）。
2. **推理对齐**（同权重同输入）：
   - 逐层/逐尺度特征对比（backbone/neck/head 各输出 maxdiff≈0）。
   - 端到端解码后评估 mAP（某数据集上框架 vs ultra，cudnn.deterministic）。
3. **训练对齐**（同权重同输入同标签，forward+backward）：
   - 检测损失（box/cls/dfl）与 ultra `v8DetectionLoss` 对齐，loss 逐位一致。
   - 每层梯度 maxdiff ≤ 算子级阈值（≈1e-4）。

**验收标准**：同权重下推理输出一致（算子级）；loss 逐位一致；梯度无漏层、maxdiff ≤ 阈值。

## 三、执行顺序（由易到难、逐版本推进）

先做「代表规模 n」跑通三步，再扩展该版本其余规模；每完成一个版本更新 AGENTS.md。

1. **yolo11**（n→s/m/l/x）：框架检测基础最熟（C3k2/Detect26 已对齐过），先做。
2. **yolov8**（n→s/m/l/x）：C2f + Detect，结构成熟。
3. **yolo12 / yolo26**（n→s/m/l/x）：C2PSA 系，接续。
4. **yolov10**（n→s/m/l/x）。
5. **yolov9**（t/s/m/c/e）。
6. **yolov5**（n/s/m/l/x）。
7. **yolov3u**（u/sppu/tinyu）。

## 四、前提检查（执行前）

- 确认框架 `tasks/detect.py` 检测头能按版本选对 `Detect` 变体（`Detect/DetectDFL/Detect26/Detect10/Detect12`…）。
- 确认 `det.py` 的 DetLoss/DetMetric/DetPostProcess 对应 ultra 各版本（v8DetectionLoss；v9/v5/v10 用各自 head/assign）。
- 确认仓库 cfg/models 已有各版本检测 yaml（v8/v9/v10/v11/v12/v26/v5/v3 已存在，v3u 需 v3 结构支持）。
- 数据集：用仓库已有 det 数据集（COCO 或 det_demo / dota128）做推理 mAP 与训练对齐。

## 五、状态跟踪

- 已完成：Classification 全 15 个、Segment(package-seg)、Pose(tiger-pose)、OBB(dota128)。
- 进行中：本计划 Detet。
- 每个版本×规模完成 → 更新本节 + AGENTS.md。
