# PP-OCRv6（两阶段 OCR：det + rec）

> **定位**：百度飞桨 OCR 主力版本，**tiny/small/medium** 三档，
> **全档权重已在 ModelScope 托管**；**低框密度场景（<20 框/图）的首选**。
> **任务**：`ocr_det` / `ocr_rec` | **权重**：`PP-OCRv6_{tiny,small,medium}_{det,rec}.pth`

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | **PP-OCRv6: A New Era of PaddleOCR**（技术报告，arXiv:2601.12638 系列的百度官方报告）<br>⚠️ 具体编号请以百度官方发布为准；**核心思想已稳定多年**（DB 检测 + CRNN/SVTR 识别） |
| 机构 | 百度飞桨（PaddlePaddle） |
| 官方代码 | https://github.com/PaddlePaddle/PaddleX |
| 官方权重 | `paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_*.tar` |
| 本框架实现 | 推理链路（`torchkiln/ocr/`），权重来自 PaddleX 官方推理模型 |
| 托管状态 | ✅ ModelScope `pretrained/` 已有 **24 个** PP-OCR 权重 |

### 要解决的问题

OCR 落地有两条路线：**两阶段**（det 找框 → 逐框 rec 识别）与**端到端**（一次输出）。
PP-OCRv6 走**两阶段**路线，目标是**在保持精度的前提下把体积和延迟压到可部署区间**。

**为什么低框密度时两阶段反而更快？**

| 环节 | 成本 |
|---|---|
| det 一次前向 | ~14 ms（与框数无关） |
| rec N 次前向 | GPU 上 **<1 ms/框**（可 batch） |
| **rec 的 CPU 预处理** | **1.555 ms/框**（crop + resize + 归一化的 Python 循环）← **瓶颈** |
| 后处理 | ~6 ms |

⇒ **固定开销约 23.9 ms**（det + 后处理 + Python 启动），
**每框增量 1.669 ms**。框少时固定开销主导 ⇒ 两阶段快；框多时线性项主导 ⇒ 端到端快。

### 核心创新（历代累积）

| 代 | 关键改进 |
|---|---|
| **v3** | DB 文本检测（**可微分二值化**）+ CRNN 识别（**CTC 头**）—— 奠定两阶段范式 |
| **v4** | SVTR 识别头（**单结构模型，超越 CRNN 的 3 段式**） |
| **v5** | 更轻的移动版（4.71 MB det）；PP-OCRv5_server 走高精路线 |
| **v6** | **PP-LCNetV4 骨干 + RepLKFPN 颈 + DBHead**，tiny 档压到 **1.87 MB det** |

---

## 2. 网络结构

### 2.1 v6_tiny 架构（⭐ 推荐档）

```
                    ┌───────────── Backbone ─────────────┐
Input (B,3,H,W) ──→ │ PPLCNetV4(model_size='tiny')       │  1.87 MB
                    │  · Rep 块 + SE + 深度可分离        │
                    │  · 输出 4/8/16/32 四级特征         │
                    └───────────────┬─────────────────────┘
                                    ↓
                    ┌───────────── Neck ─────────────────┐
                    │ RepLKFPN(out=64)                    │  含 4 特征
                    │  · 大核（5×5）卷积 + 局部增强       │  双向融合
                    │  · 近融合 UPF + 远融合 ADF          │
                    └───────────────┬─────────────────────┘
                                    ↓
                    ┌───────────── Head ─────────────────┐
                    │ DBHead                              │  det 输出
                    │  · DB 二值化（可微分）              │
                    │  · probability map (1, H/4, W/4)    │
                    │  · threshold map  (1, H/4, W/4)    │
                    │  · binary map     (1, H/4, W/4)    │
                    └───────────────┬─────────────────────┘
                                    ↓
              ┌───────────── PostProcess ─────────────┐
              │ ① prob > thresh 二值化                │
              │ ② cv2.findContours 提轮廓            │
              │ ③ minAreaRect 得最小外接矩形          │
              │ ④ unclip 扩张（ratio 参数）           │
              │ ⑤ 排序 + 过滤（box_thresh/min_size） │
              └───────────────┬─────────────────────┘
                              ↓  N 个文本框
                    ┌───────────── Rec ────────────────┐
                    │ 逐框 crop + resize + 归一化       │ ★ CPU 瓶颈
                    │  （batch 可调，但收益有限）       │  1.555 ms/框
                    └───────────────┬───────────────────┘
                                    ↓
                    ┌──── CTC 贪心解码 ──────────────┐
                    │ argmax → 去 blank → 去重         │
                    │ → 文本 + 置信度                  │
                    └──────────────────────────────────┘
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| **`PPLCNetV4`** | Rep 块（多分支重参数化）+ SE + 深度可分离卷积 | 轻量骨干；`model_size` 控制 tiny/small/base/large |
| **`RepLKFPN`** | **LKF**（Large Kernel Fusion，大核 5×5）+ **UPF**（近融合）+ **ADF**（远融合，ASPP-like） | 兼顾小目标（近融合保细节）与大文本（远融合保上下文） |
| **`DBHead`** | 概率图 + 阈值图 + 梯度图，训练用可微分二值化 | 文本检测；**DB 的核心是可微分二值化**（分割式，无 NMS） |
| **Rec 头** | CRNN（CTC）或 SVTR | 识别；v6 默认 SVTR 结构 |

### 2.3 与其他版本的对比

| 版本 | det 架构 | 特点 |
|---|---|---|
| PP-OCRv3 | ResNet/MobileNetV3 + DB | 首代；已被 v5/v6 超越 |
| PP-OCRv4 | PPLCNet + LKA neck | 首次引入大核 |
| PP-OCRv5_mobile | MobileNetV3 + **轻量 LKFPN** | 20.76 MB 合计 |
| **PP-OCRv6_tiny** | **PPLCNetV4(tiny) + RepLKFPN(64) + DBHead** | **6.25 MB，30.8 ms** ⭐ |
| PP-OCRv6_medium | PPLCNetV4(base/large) + DBHead | 132.68 MB，高精度 |

---

## 3. 输出与后处理

### 3.1 检测输出

| 张量 | 形状 | 含义 |
|---|---|---|
| probability map | (B, 1, H/4, W/4) | 每像素是文本的概率 |
| threshold map | (B, 1, H/4, W/4) | 自适应二值化阈值（**解决相邻文本粘连**） |
| binary map | (B, 1, H/4, W/4) | 可微分二值化结果（训练用） |

### 3.2 DB 后处理（关键步骤）

| 步骤 | 参数 | 说明 |
|---|---|---|
| ① 二值化 | `thresh`（0.3）、`box_thresh`（0.6） | 概率图阈值化 |
| ② 轮廓提取 | `cv2.findContours` | ★ 分割式，**不用 NMS** |
| ③ 最小外接矩形 | `cv2.minAreaRect` | 得 4 点 + 角度 |
| ④ 扩张 | `unclip_ratio`（1.5~1.8） | 恢复被二值化侵蚀的边缘 |
| ⑤ 过滤 | `min_size` | 去除噪声框 |
| ⑥ 缩放回原图 | `scale × 4` | 从 1/4 分辨率映射回原图 |

> ⭐ **DB 是分割式检测**（不依赖 NMS、不依赖 anchors），
> 这也是它对**不规则/弯曲文本**友好、且相邻文本不易粘连的原因。

### 3.3 识别后处理

| 项 | 说明 |
|---|---|
| 输入尺寸 | 高固定 48（v3+ 常用），宽按比例 |
| 解码 | **CTC 贪心**：argmax → 折叠重复 → 去 blank |
| 输出 | `(text, confidence)`，`confidence` 是字符概率均值 |
| ⚠️ 本框架限制 | **贪心**，未实现 beam search（PGNet 同） |

### 3.4 两阶段的成本公式（实测）

```
ms/图 = 23.9 + 1.669 × 框数
```

| N 框/图 | 两阶段耗时 |
|---|---|
| 5 | 31.5 ms |
| 10 | 40.6 ms |
| **20（交叉点）** | **57.3 ms** |
| 50 | 107.4 ms |
| 100 | 190.8 ms |
| **300** | **524.7 ms** |

> ⚠️ **加大 rec batch 治不了**：batch 6→64→128，单框成本 1.669→1.555→1.562 ms
> （**-7% 后持平**），因为瓶颈是 **CPU Python 循环**。

---

## 4. 配置与用法

### 4.1 权重

```yaml
Global:
  pretrained_model: PP-OCRv6_tiny_det    # 裸名，自动从 ModelScope 下载
```

远程已有（**24 个**）：

| 系列 | 权重名 | 合计体积 |
|---|---|---|
| **v6** | `PP-OCRv6_{tiny,small,medium}_{det,rec}.pth` | 6.25 / 30.04 / 132.68 MB |
| **v5** | `PP-OCRv5_{mobile,server}_{det,rec}.pth` | 20.76 / 165.19 MB |
| **v4 / v3** | `PP-OCRv{4,3}_{*}_{det,rec}.pth` | — |

### 4.2 推理

```bash
# PaddleX 推理管线
python -m paddlex --model_name PP-OCRv6_tiny_det --model_dir ... 
```

框架内（训练/评估）：

```bash
tkiln train   -c configs/ocr/ppocrv6_det.yml
tkiln val     -c configs/ocr/ppocrv6_det.yml --weights output/.../best_accuracy.pth
tkiln predict -c configs/ocr/ppocrv6_det.yml --weights ... --input imgs/
```

### 4.3 完整两阶段流程

```python
# 1) 检测
boxes, scores = det_model(image)                  # 一次前向
# 2) 逐框裁剪（CPU 瓶颈）
crops = [crop_resize_normalize(image, box) for box in boxes]
# 3) 批量识别（GPU 快，但受 CPU 裁剪拖累）
texts, confs = rec_model(batch(crops))
```

> ⚠️ 若要优化：**多进程/多线程做 step 2**（crop+resize 可并行），
> 或换端到端模型（见 `pgnet_lite.md`）。

---

## 5. 规模与速度

| 档位 | det | rec | 合计 | 端到端耗时/图 | 检出 |
|---|---|---|---|---|---|
| **PP-OCRv6_tiny** ⭐ | **1.87 MB** | **4.38 MB** | **6.25 MB** | **30.8~32.5 ms** | 4.1 行/图 |
| PP-OCRv6_small | 9.59 MB | 20.45 MB | 30.04 MB | — | — |
| PP-OCRv6_medium | 59.39 MB | 73.29 MB | 132.68 MB | — | — |
| PP-OCRv5_mobile | 4.71 MB | 16.05 MB | 20.76 MB | — | — |
| PP-OCRv5_server | 84.25 MB | 80.94 MB | 165.19 MB | — | — |

**测试环境**：RTX 4060 Ti 16GB，**`paddle.inference` 推理引擎**（非动态图），
det 输入长边 960，Total-Text test 300 张。

> ⚠️ **动态图 vs 推理引擎口径不同，勿混比**：
> 同为 PGNet，动态图 **90 ms** vs 推理引擎 **54 ms**。

**下载速度参考**：`bcebos.com` 实测 **68~105 MB/s**。

---

## 6. 公开指标

| 项 | 说明 |
|---|---|
| 数据集 | Total-Text（1563 条目，train 1255 / test 300）、CTW1500、ReCTS 等 |
| 官方指标 | 见 PaddleOCR 官方基准（各档在各数据集上的 Hmean） |
| 本框架托管 | **官方推理权重**（转换而来），**未做自训** |
| 定位 | 端到端 PGNet_r50 的 Hmean = **84.69**（Total-Text） |

> ℹ️ 本框架**未对 PP-OCRv6 做逐位数值对齐**（托管的是官方推理权重），
> 故本文不列「框架 vs 官方」的对比数字 —— **不编造未验证的数据**。

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| **低框密度（<20 框/图）** | ✅ **v6_tiny** | 6.25 MB / 31 ms，零成本 |
| 单张图片/招牌/车牌 | ✅ v6_tiny | 检测只需一次前向 |
| 移动端/嵌入式 | ✅ v6_tiny 或 v5_mobile | 体积 6~21 MB |
| 服务器、高精度 | ✅ v6_medium / v5_server | 132~165 MB，精度优先 |
| **高框密度（几百框/图）** | ❌ 改用 **PGNet_lite** | 见 `pgnet_lite.md`，快 9~15× |
| 弯曲/任意角度文本 | ✅ v6 的 DB（分割式） | 不依赖 NMS，粘连少 |
| 需要 beam search | ⚠️ 需自行扩展 | 当前是 CTC 贪心 |

**选型决策树**：

```
框数/图 ?
├─ < 20  ──────────→ PP-OCRv6_tiny ⭐（6.25MB / 31ms）
│
└─ ≥ 20  ──────────→ PGNet_lite ⭐（4.66MB / ~33ms，固定）
```

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| **vs PGNet_lite** | 低框密度两阶段更快（31 vs 33 ms，且精度更成熟）；高框密度 PGNet 快 9× |
| **vs PP-OCRv5** | v6_tiny 更小（6.25 vs 20.76 MB），v6 换了 PPLCNetV4 骨干 + RepLKFPN |
| **vs CRNN+CTC（经典）** | 经典两阶段；PP-OCRv4+ 用 SVTR 识别头（单结构超越 CRNN） |
| **vs DBNet 论文原版** | PP-OCR 是 DB 的工程化（加入阈值图解决粘连、轻量化） |
| **vs 端到端（PGNet）** | 端到端避免 CPU 裁剪；两阶段模块可独立升级（换 det 不影响 rec） |

---

## 9. 已知问题 / 注意事项

| 项 | 说明 |
|---|---|
| ⚠️ **PaddleX 推理包双重嵌套** | 解压时同名顶层目录会套一层（`dir/dir/inference.yml`）⇒ **需上移一层** |
| ⚠️ **PaddleX OCR pipeline 依赖** | 需 `ocr-core`：`pypdfium2`、`python-bidi`（`cv2`/`imagesize`/`pyclipper`/`shapely` 本机已有） |
| ⚠️ **动态图 vs 推理引擎** | 口径差 1.7×（PGNet 90 vs 54 ms）⇒ **勿混比** |
| ⚠️ **CPU 裁剪是瓶颈** | 1.555 ms/框，batch 治不了；优化方向是**多进程 crop** 或**换端到端** |
| ⚠️ **DB 无 NMS** | 相邻文本靠 `threshold map`（自适应二值化）分离；若粘连，调大 `thresh` 或减 `unclip_ratio` |
| ⚠️ **未做数值对齐** | 托管的是官方推理权重（转换而来），本框架**未验证逐位一致** |
| ⚠️ **CTC 贪心** | 未实现 beam search；长文本/易混淆字符精度受限 |
