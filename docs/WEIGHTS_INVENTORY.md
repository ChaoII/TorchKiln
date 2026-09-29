# 预训练权重清单（本地 vs ModelScope 远程）

> **盘点时间**：2026-09-29（含**上传后复核**）
> **远程仓库**：`ChaoII0987/TorchKiln` → `pretrained/`（**公开**，否则 resolve 404）
> **远程 .pth 总数**：**153**
> **本地 `weights/` .pth 总数**：**28**（含新转换的 kokoro 2 个）

---

## 〇、上传后复核结论（2026-09-29）

对 **46 个裸名**（含新上传的 `kokoro_v1_0` / `kokoro_v1_1_zh`）做了端到端实测：

| 检查项 | 结果 |
|---|---|
| 仓库里所有 `pretrained_model` 引用能否解析到远程 | ✅ **全部命中**，0 个失败 |
| 远程 HTTP 连通性（真实请求，框架 `model_url` 那条） | ✅ **12/12 HTTP 200**，0 失败 |
| 裸名展开规则 | `_candidates('X') → ['X.pth', 'X']` ✅ |
| ultralytics 风格名（`yolov5nu` / `yolov3u`） | ✅ **远程已有同名文件**，不会静默回落 |
| `null` 走「从零训练」 | ✅ 设计如此 |
| 路径含分隔符但文件不存在 | ✅ 抛 `FileNotFoundError`（**不静默回落**）|
| 裸名拼错 / 远程没有 | ⚠️ 原本只有 `warning` ⇒ **静默从零训练**。**已修**：升级为 `error` + 明确说明后果 + 4 条排查方向 |

**同步改动**：22 个配置里的**完整 URL** 已统一改为**裸名**
（同一文件、同一解析结果；原写法冗长且把 `master` 分支写死）。

> ⚠️ **仍需你决定**：11 个配置是 `pretrained_model: null`，但远程已有对应权重
> （kokoro / centerpoint / squeezesegv3 / bev_lanedet / panns）。
> **未擅自改** —— 那会把「从零训练」变成「微调」，是**行为变更**。

---

## 一、结论先行

| 判定 | 数量 | 说明 |
|---|---|---|
| ✅ **远程已有、可直接用** | **153** | YOLO 家族 + PP-OCR + 车牌 + 属性 + 3D/点云/车道 + 音频 4 个 + PGNet_lite |
| ❌ **本地有、但不该上传** | **19**（1.67 GB） | **业务/对齐中间产物**（nc=1 实验权重、OBB 转换中间件、业务微调） |

> ⚠️ **上一版本文档的推荐是错的**（曾说「远程只有 3 个、需上传 44 个」），
> 根因是**列表接口分页没取全**。现以 `repo/files?Root=pretrained` 一次性取
> `Data.Files` 的 `Name` 为准。

---

## 二、✅ kokoro（已上传并复核通过）

| 文件 | 大小 | 来源 | 验证 |
|---|---|---|---|
| **`kokoro_v1_0.pth`** | 312.1 MB | 官方 `hexgrad/Kokoro-82M` v1.0（多语言/英文） | 549 张量 / **81.763 M** / **missing=0 unexpected=0** |
| **`kokoro_v1_1_zh.pth`** | 312.1 MB | 官方 `hexgrad/Kokoro-82M-v1.1-zh`（中文） | 同上，**missing=0 unexpected=0** |

**为什么值得上传**：
- 是**唯一的通用 TTS 预训练基座**（仓库此前无任何 TTS 权重）；
- 两版**架构完全相同**（仅 `vocab` 不同：114 vs 171 词表），可互换；
- 转换要点已固化为「**扁平 state_dict + `InstanceNorm1d(affine=False)`**」。

**转换脚本**：`_downloads/kokoro_convert.py`
**关键修法**（框架 `torchkiln/audio/kokoro/istftnet.py` 已内置）：

```
官方 AdaIN1d 的 InstanceNorm1d 是 affine=True（ONNX 导出 workaround 的副产物），
但官方权重里**没有** weight/bias ⇒ 必须 affine=False 才能 missing=0。
官方源码注释自认：affine=True 在推理模式下「不应该有影响」，
本框架实测**端到端波形逐位一致（maxdiff=0.000e+00）**。
```

**用法**：

```yaml
Global:
  pretrained_model: kokoro_v1_0        # 裸名，自动从 ModelScope 下载
Architecture:
  task: kokoro_tts
  Head:
    model: kokoro
    config: torchkiln/audio/kokoro/configs/kokoro-v1_0.json   # 必需（HF 被墙，不回退下载）
```

> ⚠️ kokoro 需 `config.json` 配套（已入库于 `torchkiln/audio/kokoro/configs/`）。
> ⚠️ `Generator` 含 3 处随机噪声（正弦初相 + 两处 `randn_like`）⇒ 复现需固定种子。
> ⚠️ 整链一次反传在部分输入下会触发原生段错误 `0xC0000005` ⇒ 训练需**分段反传**。

---

## 三、✅ 已接入预训练的配置（4 个，均实测 missing=0）

| 配置 | `pretrained_model` | 模型参数 | 加载结果 |
|---|---|---|---|
| `configs/pc/centerpoint-det3d.yml` | `centerpoint_pillars_kitti` | 4.996 M / 223 张量 | **missing=0 / unexpected=0 / 形状全对** |
| `configs/lane/bev_lanedet.yml` | `bev_lanedet_apollo_576x1024` | 43.997 M / 440 张量 | **missing=0 / unexpected=0** |
| `configs/local/apollo_bev_lanedet.yml` | `bev_lanedet_apollo_576x1024` | 43.997 M / 440 张量 | **missing=0 / unexpected=0** |
| `configs/pc/squeezesegv3-semantickitti.yml`（**新增**） | `squeezesegv3_rangenet53_semantickitti` | 25.960 M / 612 张量 | **missing=0 / unexpected=0**（**真实从 ModelScope 下载 105MB 验证**）|

### ⭐ SqueezeSegV3 的类别数坑（新增配置的原因）

原 `configs/pc/squeezesegv3-pcseg.yml` 是 **5 类合成 demo**，
而官方 SemanticKITTI 权重是 **20 类**（19 语义类 + 1 unlabeled）：

| 配置 | 模型形状 | 权重形状 | 结果 |
|---|---|---|---|
| `num_classes: 5` | `heads.* = (5, …)` | `heads.* = (20, …)` | ❌ **10 个张量形状不符** |
| `num_classes: 20` | `heads.* = (20, …)` | `heads.* = (20, …)` | ✅ **missing=0** |

⇒ **只有 20 类能加载官方权重**。故：
- 原 5 类 demo 配置**保持 `null` 不动**（合成数据，改类别数会让指标不可比）；
- **新增** `configs/pc/squeezesegv3-semantickitti.yml`（20 类 + 预训练），
  注释里写清数据来源与标签重映射要求。

> ⚠️ `num_classes` 的真实位置是 **`Architecture.Head.num_classes`**
> （`build_loss` / `build_metric` 都从那里读；写到 `Architecture` 顶层**无效**）。

---

## 四、⚠️ 保持 `null` 的配置及原因

| 配置 | 原因 |
|---|---|
| `configs/audio/kokoro_demo.yml` | demo 用合成数据，**从零更合适** |
| `configs/audio/panns_esc50_*.yml`（4 个） | 对齐实验**故意从零**（否则无法与 PaddleSpeech 逐步对比）|
| `configs/pc/squeezesegv3-pcseg.yml` | 5 类 demo，权重是 20 类（**形状不匹配**）|
| `configs/pc/pointpillars-det3d.yml`、`pointpillars-seg.yml` | ⚠️ **架构不同**（PointPillars ≠ CenterPoint），**不能混用** |
| `configs/ts/*`（34 个） | 时序模型**无官方预训练权重**（paddlets 不发布）|

---

## 五、❌ 不该上传的（19 个 / 1.67 GB）—— 业务与对齐中间产物

> **上传标准**：只上传「**官方权重转换**」产物（供用户微调的通用基座），
> **不上传**业务对齐过程中的中间产物。

### 3.1 `*-seg_nc1.pth`（15 个 / 1640.3 MB）—— 对齐实验用

```
yolov8{n,s,m,l,x}-seg_nc1.pth    yolo11{n,s,m,l,x}-seg_nc1.pth    yolo26{n,s,m,l,x}-seg_nc1.pth
```

**是什么**：为做「同起点单步对齐」，用 ultra `SegmentModel(alg.yaml, nc=1) + load(COCO)`
重建的**单类别**权重。

**为什么不上传**：
- `nc=1` 对 COCO 80 类的用户**毫无微调价值**（分类头只认 1 个类）；
- 属于**过程产物**，不是官方发布物；
- 体积 1.6 GB，会把仓库撑大。

### 3.2 `*_obb_fw.pth`（3 个 / 21.8 MB）—— 去掉函数式 DFL 的中间件

```
v8n_obb_fw.pth (6.0 MB)    v26n_obb_fw.pth (5.4 MB)    yolo11n_obb_fw.pth (10.4 MB)
```

**是什么**：官方 OBB 权重去掉 ultralytics 那个**函数式** `model.23.dfl.conv.weight`
（它在 `state_dict` 里是死键，框架侧 DFL 是 `Identity`）后的转换结果。

**为什么不上传**：
- 框架加载官方 OBB 权重时**自动跳过该键**即可（`load_state_dict` 的
  `unexpected=1` 已实测无害），**不需要**人工转换版；
- **OBB 微调基座直接用远程已有的** `yolo11n-obb.pth`（10.4 MB）/ `yolo26n-obb.pth`（5.4 MB）。

### 3.3 `yolo11n_seg_ft_fw.pth`（1 个 / 11.1 MB）—— 业务微调产物

**是什么**：在 `datasets/package-seg` 上微调后的 seg 权重（对齐实验用）。

**为什么不上传**：这是**特定业务数据**的产物，不是通用基座。

---

## 六、远程已有的（153 个）—— 已验证可直接下载加载

### 4.1 YOLO 检测家族（123）

| 家族 | 任务 | 权重示例 | 本框架验证 |
|---|---|---|---|
| **yolo11** | det/seg/pose/obb/cls | `yolo11n.pth` (5.2 MB) / `yolo11n-obb.pth` (10.4 MB) / `yolo11n-seg.pth` (5.7 MB) | n/s/m/l/x 全 missing=0；OBB mAP 0.8005↔ultra 0.821 |
| **yolo26** | det/seg/sem/depth/obb/cls | `yolo26n.pth` (5.2 MB) / `yolo26n-obb.pth` (5.4 MB) | 全家族 missing=0；E2E loss 27.138552↔27.138544 |
| **yolov8 / v9 / v10 / v12** | det/seg/pose/obb/cls | 各 n/s/m/l/x | 全部 missing=0（仅差函数式 dfl） |
| **yolov3u / v5u** | det | 各 n/s/m/l/x | missing=0 |

> ✅ 端到端训练 mAP 与 ultra 对齐（v11 差 0.009 / v8 0.005 / v26 0.013 / v12 0.006 / v10 0.026 / v9c 0.005）。

### 4.2 PP-OCR（24）

| 系列 | 权重 | 体积 |
|---|---|---|
| PP-OCRv6 | `{tiny,small,medium}_{det,rec}.pth` | 6.25 / 30.04 / 132.68 MB |
| PP-OCRv5 | `{mobile,server}_{det,rec}.pth` | 20.76 / 165.19 MB |
| PP-OCRv4 / v3 | 各 det/rec | — |

### 4.3 车牌 / 属性（3）

| 文件 | 大小 | 说明 |
|---|---|---|
| `plate_detect.pth` | 1.0 MB | yolov5n-0.5 车牌检测 |
| `plate_rec_color.pth` | 0.7 MB | CRNN+CTC 车牌识别 |
| （属性头） | — | PP-LCNet 多标签属性 |

### 4.4 3D / 点云 / 车道（3）

| 文件 | 大小 | 验证 |
|---|---|---|
| `centerpoint_pillars_kitti.pth` | 19.2 MB | missing=0（191 张量）；KITTI BEV mAP(Mod) 均值 67.8 vs Paddle3D 71.87 |
| `squeezesegv3_rangenet53_semantickitti.pth` | 99.7 MB | missing=0（524 张量）；前向 maxdiff ~1e-5 |
| `bev_lanedet_apollo_576x1024.pth` | 168.1 MB | missing=0（372 张量）；F1 0.8378 vs Paddle3D 0.7776 |

### 4.5 音频（3）+ OCR 端到端（1）

| 文件 | 大小 | 验证（四条全过） |
|---|---|---|
| `panns_cnn14.pth` | 308.1 MB | fp64 前向 **3.55e-11** / loss **6.46e-12** / probs 差 **3.4e-05** |
| `ecapa_tdnn_voxceleb12.pth` | 79.5 MB | fp64 **1.47e-15** / 余弦 **0.9999996** |
| `mdtc_heysnips.pth` | 0.2 MB | fp64 **6.02e-10** / logits **4.04e-09** |
| `pgnet_lite_totaltext.pth` | 4.9 MB | PGLoss 逐位一致（349.08233642578125 ↔ 同值） |

---

## 七、下载与加载机制

```python
# 框架自动下载（config 里写裸名，不带 .pth）
Global:
  pretrained_model: kokoro_v1_0     # -> ~/.torchkiln/pretrained/kokoro_v1_0.pth
```

| 环境变量 | 作用 |
|---|---|
| `PYTORCHOCR_HOME` | 缓存根目录（默认 `~/.torchkiln/`） |
| `PYTORCHOCR_ALLOW_LOCAL_REPO` | 允许读本地镜像 `\\tsclient\E\TorchKiln` |
| `PYTORCHOCR_AUTO_DOWNLOAD` | 关掉自动下载（离线环境） |

> ⚠️ **仓库必须公开**：框架用公开 URL 无 token 下载，
> 私有仓库会导致 `resolve 404`（git 返回 401）。
> ⚠️ 权重文件名**不带** `_ptocr` / `_state` 后缀（配置 URL 与缓存均为裸名 `.pth`）。

---

## 八、上传操作（需 ModelScope token）

```bash
pip install modelscope
modelscope login --token <YOUR_TOKEN>

# 逐个上传（保持裸名 .pth）
for f in kokoro_v1_0 kokoro_v1_1_zh; do
  modelscope upload ChaoII0987/TorchKiln "E:/TorchKiln/weights/$f.pth" "pretrained/$f.pth"
done

# 验证可下载
python -c "from torchkiln.ptcore.pretrained import resolve_pretrained; \
           print(resolve_pretrained('kokoro_v1_0'))"
```

> ⚠️ 私有仓库会让 resolve 失败 ⇒ **保持公开**或提供 token。
> ⚠️ `kokoro` 还需 `config.json`（已入库，不需上传）。
