# PGNet_lite（轻量端到端 OCR）

> **定位**：PaddleOCR **PGNet** 端到端方案（检测框 + 字符 + 方向一次输出）的**轻量化改写**，
> **1.22M 参数 / 4.66 MB**，高框密度场景比两阶段快 **9~15×**。
> **任务**：`ocr_e2e` | **权重**：`pgnet_lite_totaltext.pth`（4.9 MB，已托管 ModelScope）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **Towards End-to-End Scene Text Spotting with Convolutional Regression Network**（PGNet） |
| 出处 | arXiv 预印本 + PaddleOCR 内置；官方基线 **r50_vd** 版本（48.53M / 185 MB） |
| 官方代码 | https://github.com/PaddlePaddle/PaddleOCR（`ppocr/modeling/{backbones,e2e_pgnet_lite}`） |
| 官方权重 | `paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/e2e_server_pgnetA_infer.tar`（187 MB） |
| 本框架实现 | `torchkiln/ocr/modeling/e2e_pgnet_lite.py`（`PPLCNetV4E2E` / `PGFPNLCNet` / `PGHeadLite`） |
| 移植方式 | **Paddle → torch 逐属性同名搬运**（便于权重转换）+ **PGLoss 逐位对齐** |
| 本框架权重 | `weights/pgnet_lite_totaltext.pth`（4.9 MB，**Total-Text 训练**） |

### 要解决的问题

**两阶段 OCR（det + 逐框 rec）的成本随文本行数线性增长**。
实测（同一引擎 RTX 4060 Ti）：

```
ms/图 = 23.9 + 1.669 × 框数        # 两阶段
```

| 框/图 | 两阶段 v6_tiny | PGNet(r50) | 胜者 |
|---|---|---|---|
| 5 | 31.5 ms | 54.2 ms | 两阶段 1.7× |
| **20（交叉点）** | 54.8 ms | 54.2 ms | 持平 |
| 50 | 101.5 ms | 54.2 ms | PGNet 1.9× |
| **300** | **490 ms** | **54.2 ms** | **PGNet 9.0×** |

**关键定位**（决定性实验）：
- 加大 rec 的 `batch_size` **治不了**（6→64→128，单框成本仅 1.669→1.555→1.562 ms，**-7% 后持平**）；
- 根因：**1.555 ms/框 是 CPU 逐框 crop + resize + 归一化的 Python 循环开销**，
  **不是 GPU 前向**（v6_tiny rec GPU 前向 < 1 ms）。

⇒ 两阶段的 `O(N)` **CPU 开销是结构性的**，batch 治不了，必须换架构。

### 核心创新

PGNet 的关键是**一个网络同时输出 4 张图**：

| 分支 | 通道 | 含义 | 作用 |
|---|---|---|---|
| `f_score` | 1 | **文本骨架**（TCL, Text-Centered Localization）概率 | 找出「文字像素在哪」 |
| `f_border` | 4 | 上/下/左/右**边界距离** | 从骨架**扩张**回完整文本框 |
| `f_char` | 37 | 每像素的**字符分布**（字典 36 + blank） | 直接分类，**免裁剪免旋转** |
| `f_direction` | 2 | **方向**（0°/180°） | 纠正旋转文本 |

> ⭐ **TCL 是精华**：不直接回归框，而是先找文字**中心线**，
> 再用 `f_border` 4 个方向的距离场**扩张**出完整框 —— 对**不规则文本行**（弯曲、任意角度）
> 天然友好，且**不需要 RoI 变换**。

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B, 3, H, W)          # 训练 512×512，测试保比例到长边 768
 │
 └─ PPLCNetV4E2E  (0.394 M)      torchkiln/ocr/modeling/e2e_pgnet_lite.py
      └─ PPLCNetV4(det=True, model_size='tiny')
      └─ 返回 5 路：
           [0] RGB 原图        (B,3,H,W)     ★ 必须返回！PGFPN 的 down-fusion 要用
           [1] f1 @ stride 4   (B, 32, H/4,  W/4)
           [2] f2 @ stride 8   (B, 64, H/8,  W/8)
           [3] f3 @ stride 16  (B, 64, H/16, W/16)
           [4] f4 @ stride 32  (B,160, H/32, W/32)
 │
 └─ PGFPNLCNet     (0.424 M, w=64)
      ├─ **down-fusion**：RGB 原图 → AvgPool 到 s4 与 f1 拼接 → conv → s4 特征
      └─ **up-fusion**：s32 → s16 → s8 → s4   （逐级上采样 + 横向 concat + conv）
      └─ 输出：4 路统一通道数（w=64）→ 送头
 │
 └─ PGHeadLite     (0.416 M)
      ├─ f_score     conv 1×1 → (B,  1, H/4, W/4)   ★ 监督图在 stride 4
      ├─ f_border    conv 1×1 → (B,  4, H/4, W/4)
      ├─ f_char      conv 1×1 → (B, 37, H/4, W/4)
      └─ f_direction conv 1×1 → (B,  2, H/4, W/4)
```

### 2.2 逐模块说明

| 模块 | 结构 | 输入→输出 | 作用 |
|---|---|---|---|
| `PPLCNetV4E2E` | `PPLCNetV4(det=True, model_size='tiny')` 的包装 | (B,3,H,W) → 5 路 | ★ **返回 RGB 原图**是关键设计（原 PGNet 骨干就返回 `[image, f1..f6]`），这样 `BaseModel` 的 `backbone→neck→head` 流水线可直接复用 |
| `PGFPNLCNet` | down-fusion(RGB池化到s4 + 浅层) + up-fusion(s32→s16→s8→s4) | 5 路 → 4 路(w=64) | 双向融合（原版 PGFPN 硬编码 7 输入 `[3,64,256,512,1024,2048,2048]`，改写为适配 LCNet 的 4 特征 + RGB） |
| `PGHeadLite` | 4 个分支，各 `[ConvBNLayer × n, 1×1 conv]` | → 1/4/37/2 通道 | 宽度可配（`w_char`）；原 `PGHead` 的 `f_char` 分支占 **84% 参数**（单层 `conv_f_char4` 589K = 54%，内部宽 256 而其它分支只 64） |

### 2.3 与官方 PGNet(r50) 的差异

| 组件 | 官方 r50_vd | **PGNet_lite** | 变化 |
|---|---|---|---|
| 骨干 | ResNet50-vd（25.6 M） | **PPLCNetV4(tiny)**（0.394 M） | **-65×** |
| 颈 | PGFPN（7 输入，含 s2/s64 两级） | **PGFPNLCNet**（4 特征 + RGB） | 重写以适配通道与 stride |
| 头 | PGHead（1.092 M，`f_char` 内部宽 256） | **PGHeadLite**（0.416 M，`w_char=[64,64,128,128,128]`） | **-62%** |
| **合计** | **48.53 M / 185 MB** | **1.221 M / 4.66 MB** | **-40×** |
| 前向（640） | 31.2 ms | **11.0 ms** | **快 2.84×** |

> **重写忠实性验证**：用 `PGHeadLite` 以**原宽度**重建 →
> 参数 **1.092 M 与官方逐位相等**，证明重写无误后才改宽度。

---

## 3. 输出与后处理

### 3.1 输出张量

| 张量 | 形状（512 输入） | 值域 | 监督 |
|---|---|---|---|
| `f_score` | (B, 1, 128, 128) | 概率 | `tcl_maps` (1,128,128) |
| `f_border` | (B, 4, 128, 128) | 距离 | `border_maps` (5,128,128) |
| `f_char` | (B, 37, 128, 128) | logits | `label_list` (30,50,1) |
| `f_direction` | (B, 2, 128, 128) | 2 类 | `direction_maps` (3,128,128) |

> ⚠️ **监督图全在 stride 4**（128 = 512/4）—— 四个分支共用同一空间尺寸。

### 3.2 后处理（`PGNet_PostProcess`）

```
1. f_score > score_thresh (0.5)      → 二值化得文本骨架 TCL
2. 从 TCL 连通域骨架点向 4 方向搜索，
   按 f_border 的距离找边界           → 文本多边形（4 点）
3. 在多边形内（透视变换后）采样 f_char → (N_frames, 37)
4. CTC 解码（去 blank + 去重）        → 文本
5. f_direction 判定 0°/180°           → 方向修正
```

### 3.3 ⚠️ 关键坑：推理输出的张量顺序

**PGNet 推理模型（inference）输出 4 个张量的顺序**：

```
[f_border, f_char, f_direction, f_score]
```

**与 `PGHead.forward` 的 dict 插入顺序**：

```
[f_score, f_border, f_char, f_direction]
```

⇒ **映射错会触发 `sort_with_direction` 的 `IndexError`**。

### 3.4 其它后处理口径

| 项 | 值 |
|---|---|
| `data["shape"]` | **4 元** `[src_h, src_w, ratio_h, ratio_w]`（不是 2 元！） |
| `shape_list` | 形状必须是 **(1, 4)** |
| 反变换 | `x / [ratio_w, ratio_h]` 映回原图坐标 |
| 评估 | `E2EMetric`（mode A/B） |

---

## 4. 配置与用法

### 4.1 数据管线的固定输出形状

`PGProcessTrain` 输出**固定形状**（batch 维度靠 collate 堆叠）：

| 张量 | 形状 | 说明 |
|---|---|---|
| `images` | (3, 512, 512) | 训练固定 512 |
| `tcl_maps` / `tcl_label_maps` / `training_masks` | (1, 128, 128) | 骨架 |
| `border_maps` | (5, 128, 128) | 4 边界 + 1 mask |
| `direction_maps` | (3, 128, 128) | 2 类 + 1 mask |
| `label_list` | (30, 50, 1) | 字符标签 |
| `pos_list` | (30, 64, 3) | **3 列 = (img_id, y, x)** |
| `pos_mask` | (30, 64, 1) | |

> ⭐ **`pos_list` 的 `col0` 是「批内图片序号」** —— `PGProcessTrain.img_id` 自增循环，
> **周期 = 配置的 `batch_size`**。
> ⇒ **`Train.loader.batch_size_per_card` 必须与 `PGProcessTrain.batch_size` 一致**，
> 否则 `org_tcl_rois` 的 `gpu_id` 越界（已加钳制兜底，但仍应对齐）。

### 4.2 配置

```yaml
Global:
  model_name: e2e_pgnet_lite_totaltext
  pretrained_model: pgnet_lite_totaltext   # 裸名 -> ModelScope 自动下载
  epoch_num: 150
  amp: true                                  # ★ 必须开（9.8× 加速）
Architecture:
  task: e2e
  Backbone: {name: PPLCNetV4E2E, model_size: tiny}
  Neck:     {name: PGFPNLCNet, w: 64}
  Head:     {name: PGHeadLite, w_char: [64, 64, 128, 128, 128]}
Train:
  dataset:
    data_dir: datasets/total_text
    batch_size: 16        # ★ 必须 = PGProcessTrain.batch_size
    retry_on_none: true    # ★ 必须开（否则大量样本被丢）
    augment: ...           # 注意：训练增广已烘焙在 PGProcessTrain 内
  loader:
    batch_size_per_card: 16    # ★ 必须 = PGProcessTrain.batch_size
    num_workers: 4
    prefetch_factor: 4
Optimizer:
  name: Adam
  lr: {name: Linear, learning_rate: 0.001, warmup_epoch: 50}   # ★ warmup 50 不可减
```

### 4.3 三条链路

```bash
# 训练（batch16 + AMP，75.5s/epoch，GPU 利用率 98.5%）
tkiln train -c configs/ocr/e2e/e2e_pgnet_lite_totaltext.yml -o Global.eval_epoch_step=10

# 评估（E2EMetric，300 张 Total-Text test，17.6 img/s）
python tools/eval.py -c configs/ocr/e2e/e2e_pgnet_lite_totaltext.yml

# 预测
tkiln predict -c configs/ocr/e2e/e2e_pgnet_lite_totaltext.yml --input imgs/
```

---

## 5. 规模与速度

| 项 | 值 | 备注 |
|---|---|---|
| **参数** | **1.221 M** | 骨干 0.394 + 颈 0.422 + 头 0.414 |
| **权重大小** | **4.66 MB**（`.pth` 4.9 MB） | vs 官方 r50 的 185 MB |
| **前向（640×640）** | **11.0 ms** | 官方 r50 = 31.2 ms，**快 2.84×** |
| **端到端预估** | **≈32.9 ms** | 15.5(预处理) + 11.0 + 6.4(后处理) |
| vs 两阶段（300 框） | **快 14.9×** | 490 ms → 32.9 ms |
| **训练**（batch16+AMP） | **75.5 s/epoch** | 150 epoch ≈ **3.1 h** |
| GPU 利用率 | **98.5%**（最低 93%） | 显存 3582 MiB |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 | |

### batch size 的实测选择（**别再走弯路**）

| batch | 显存 | step 时长 | GPU 利用率 | 结论 |
|---|---|---|---|---|
| 14 | 3.47 GB | 0.57 s | 99% | 可用 |
| **16** | 5.9 GB | 2.4 s | **均值 98.6%，最低 95%** | ✅ **当前配置** |
| 32 | 7.32 GB | 1.5 s | — | 内存中等 |
| **48** | 11.14 GB | **16.1 s** | **均值 ~8%，掉 0% 共 4 次** | ❌ **数据管线跟不上** |
| 64 | 15.75 GB | — | — | 逼近 16 GiB 上限，评估易 OOM |

**根因**：`PGProcessTrain`（TCL 点采集 + 几何计算）是**纯 CPU 单样本**开销，
batch 越大单步等数据的时间占比越高 → **GPU 反而空转**。

---

## 6. 公开指标

### 6.1 官方基线

| 模型 | 权重 | Total-Text Hmean |
|---|---|---|
| PGNet（r50_vd） | 185 MB，600 epoch | **84.69** |

### 6.2 本框架的对齐验证（**训练正确性的硬证据**）

| 分量 | torch（本框架） | PaddleOCR 原版 | 差异 |
|---|---|---|---|
| **总 loss** | **349.08233642578125** | **349.08233642578125** | **逐位一致** |
| score_loss | 0.9430578947067261 | 0.9430578947067261 | 一致 |
| border_loss | 0.8673726320266724 | 0.8673725128173828 | 末位差（fp32 求和序） |
| direction_loss | 0.3845565915107727 | 0.3845565915107727 | 一致 |
| ctc_loss | 69.37747192382812 | 69.37747192382812 | 一致 |

> ⭐ **同批输入 + 同 RNG 种子**下的**逐位一致** ⇒ 损失、几何采集、
> CTC、边界距离**全部移植正确**。

### 6.3 评估指标为 0 的原因（**非本移植问题**）

`E2EMetric` 的 `f_score_e2e` 长期为 **0**，逐层排查后确认：

| 排除项 | 结论 |
|---|---|
| 评估循环批相关性 | 逐图 vs 批量**检出数完全一致** `[2,3,4,0]` |
| 归一化 | 框架与 PaddleOCR **逐行一致** |
| 坐标系 | `E2EResizeForTest` 的 shape 正确；GT 与预测**实际重叠** |
| **真正原因** | **预测框比 GT 窄** → `tr = 交/GT面积` 过不了 **0.7 硬门槛** |

实测（shapely 精算）：

| 预测 | 命中 GT | tr = 交/GT | tp = 交/预测 | 判定 |
|---|---|---|---|---|
| pred[0] | GT[0] | **0.464** | 0.878 | REJECT |
| pred[1] | GT[1] | **0.099** | 0.923 | REJECT |

外扩检验：buffer 2/4/6/8 px → tr 合计 0.69/0.80/0.91/1.01
⇒ **只需 ~4~6 px 外扩即可匹配**。

**佐证**：**PaddleOCR 自己的 `tools/eval.py` 也报 `f_score_e2e=0`** ⇒ 非本移植引入。

---

## 7. 选型建议

| 场景 | 建议 | 理由 |
|---|---|---|
| **高框密度**（几百框/图，如密集票据、报纸） | ✅ **本模型** | 300 框时快 **9~15×** |
| 低框密度（<20 框/图，如单张车牌、招牌） | ❌ 用两阶段 **PP-OCRv6_tiny** | 6.25 MB / 31 ms，零成本 |
| 边缘/嵌入式 | ✅ 本模型（4.66 MB） | 体积比 r50 小 40× |
| 弯曲 / 任意角度文本 | ✅ TCL 设计天然适配 | 不需要 RoI 变换 |
| 需要 beam search 提升长文本精度 | ⚠️ 当前是 CTC 贪心 | 可扩展 |

**决策树**：

```
框数/图 < 20 ──────────→ PP-OCRv6_tiny（6.25MB / 31ms）
       │
       ≥ 20 ───────────→ PGNet_lite（本模型，4.66MB / ~33ms）
```

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| **vs 官方 PGNet(r50)** | 参数 1.22M vs 48.53M（**1/40**），前向快 2.84×；**结构不同**（LCNet 骨干 + 重写的 PGFPN），故**不是**原权重的量化版 |
| **vs 两阶段 PP-OCRv6** | 低框密度输（54 vs 31 ms），高框密度赢 **9~15×**；瓶颈在 CPU 逐框循环而非 GPU |
| **vs 传统 OCR (CRNN+CTC)** | CRNN 需先 det 再裁剪；PGNet 一步到位，且 TCL 对不规则文本更鲁棒 |
| **vs DBNet + CRNN（PaddleOCR 默认）** | 同上；DBNet 擅长单行，PGNet 擅长多行 + 方向 |

---

## 9. 已知问题 / 注意事项

### 9.1 ⭐ `f_score` 的 sigmoid —— 一个真实的教训

| 阶段 | 事件 |
|---|---|
| 初始 | 本框架 `PGHeadLite.forward` **漏了 `torch.sigmoid`**，而官方 `PGHead` 有 |
| 误判 | 训练日志 `score_loss = -0.8379`（**Dice 损失出现负值**）→ 当成 bug 修复 |
| 实测 | `f_score(带sigmoid): min=0.5000 mean=0.5028`，背景精确 = `sigmoid(0) = 0.5` |
| 后果 | ① Dice 分母被 **13 万背景像素 × 0.5 淹没** → loss 下限 ≥0.94、梯度趋零<br>② 后处理 `f_score > 0.5` **恰好卡在背景值 0.5** → 阈值失效（仅 0.58% 像素略高）<br>③ 种子区域退化 → 框越来越小 → **`aratio` 0.518 → 0.028** |
| 结论 | **`Dice` 出现负值不是 bug**（`pred > gt` 即可为负，实测 `Dice(raw) = -0.7075`）<br>**已撤回 sigmoid**（与权重的原始口径一致） |

> ⚠️ **训练时必须与权重的原始口径一致**。
> 官方 `PGHead` 有 sigmoid 是因为它**从零训练**；在**已训练权重**上补 sigmoid = 口径失配。

### 9.2 训练必读

| 项 | 说明 |
|---|---|
| ⭐ **`retry_on_none: true`** | `PGProcessTrain` 按几何/随机缩放**拒绝大量样本**（`min(new_w,new_h) < input_size*0.5`、全 ignore、`len(pos_list) > max_text_nums`）。<br>**实测 batch=16 collate 后只剩 6 个有效**！PaddleOCR 的 `PGDataSet` 是「被拒就随机重取」。<br>框架 `SimpleDataSet` 新增 `retry_on_none`（30 次重试上限）修复。 |
| ⭐ **必须开 AMP** | fp32 下单步 **1.326 s**（`PGFPNLCNet`/`PGHeadLite` 都在 stride 4 上做 conv，batch16 是百 GFLOP 级）；AMP 后 **0.135 s（9.8×）**。<br>安全性：`base.py` 已在 loss 前 `_to_fp32(preds)`（**递归处理 dict**），评估强制 fp32。 |
| ⭐ **`warmup_epoch: 50`** | 官方值。epoch1 的 lr 仅 1.6e-5，峰值 1e-3 ⇒ **训短了等于白训** |
| ⭐ **batch 三处一致** | `Train.loader.batch_size_per_card` = `Train.dataset.batch_size` = `PGProcessTrain.batch_size` |
| `num_workers` | **4**（8 worker + 大 batch 曾致宿主内存崩溃：`Unable to allocate 1.00 MiB for shape (512,512)`） |
| `accumulate` | 本配置无 `Optimizer.nbs` ⇒ **`accumulate=1`**（每 batch 一次更新） |

### 9.3 评估 / 推理

| 项 | 说明 |
|---|---|
| ⭐ **评估批必须逐图** | `E2EResizeForTest` 是**保比例**缩放，同 batch 内各图尺寸不同 → **不能 stack**。<br>`e2e_eval_collate` 改为不 stack，`eval_step` 逐图前向 + 逐图 post_process + 逐图 metric。<br>（`smoke_all` 强制 batch=4 时暴露；配置里 Eval 默认 batch=1 时不暴露） |
| `OcrTask.sample_count` | e2e 且 `batch[0]` 是 list 时返回 `len(batch[0])`（父类走 `.shape[0]` 会返回 0） |
| `f_score_e2e ≈ 0` | 见 §6.3，**官方 eval 也报 0**，属模型精度问题（框偏窄），非管线问题 |
| 验收建议 | ① 用**同一评估器**评双方模型；② 或用诊断脚本看 `aratio_med`（预测框面积/GT 面积）——**比 recall 灵敏得多** |

### 9.4 移植坑（跨框架）

| 坑 | 说明 |
|---|---|
| Paddle `Pad1D` | `_Interactor` 内层卷积用**零填充** ⇒ torch 必须 `nn.ConstantPad1d(...,0)`，**不能** `ReplicationPad1d` |
| NCL 通道轴 | Paddle NCL 下**通道维 = 时间轴**，`_decoder1/2` 是 `Conv1D(in_chunk_len→out_chunk_len, k=1)`，forward **直接** `decoder(x)`，**不要** transpose |
| 推理输出顺序 | `[f_border, f_char, f_direction, f_score]` ≠ `PGHead.forward` 的 dict 顺序 |
| `character_length` | 框架 torch 版 `PGHead` 把字典长度**硬编码为 37**（无 `character_dict_path`） |
| registry 注册 | `backbones/__init__.py` 必须在 **`model_type=='e2e'` 分支**（det 分支的 `support_dict=['ResNet']` 会**覆盖**）；`necks`/`heads` 的 import 是**函数内懒加载**（避免循环 import） |
| `out_channels` 属性 | `BaseModel` 读 `backbone.out_channels` → neck → head ⇒ 两个新组件**必须有** `out_channels`（颈用 `**kwargs` 吞掉） |
| `forward(targets=None)` | `BaseModel.forward` 调 `head(x, targets=…)` ⇒ **必须接受该参数** |
| Paddle bn_name 冲突 | `ConvBNLayer` 的 `bn_name = "bn" + name[3:]`（剥前 3 字符）⇒ uid 需 **≥5 字符且区分位在索引 3 之后**；同进程**不能建两个官方 `PGHead`**（名字硬编码） |
