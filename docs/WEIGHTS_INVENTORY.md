# 预训练权重清单：本地 vs ModelScope 远程

> 盘点时间：2026-09-29 · 本地目录 `weights/` · 远程 `ModelScope: ChaoII0987/TorchKiln → pretrained/`

## 一、远程已托管（3 个）

| 文件 | 大小 | 对应模型 |
|---|---|---|
| `centerpoint_pillars_kitti.pth` | 19.6 MB | 3D 检测 CenterPoint-Pillars |
| `squeezesegv3_rangenet53_semantickitti.pth` | 99.7 MB | 点云语义分割 SqueezeSegV3 |
| `bev_lanedet_apollo_576x1024.pth` | 168 MB | BEV 车道线 BEV-LaneDet |

> 用法：config 里写裸名（不带 `.pth`），框架会从 ModelScope 自动下载到 `~/.torchkiln/pretrained/`。

---

## 二、⭐ **本地有、远程无 → 需要上传（44 个 / 约 3.92 GB）**

### 2.1 音频（4 个 / 约 412 MB）—— **建议优先上传**

| 文件 | 大小 | 模型 | 备注 |
|---|---|---|---|
| `panns_cnn14.pth` | 323.1 MB | PANNs CNN14 语音分类 | AudioSet 预训练，80.77M |
| `ecapa_tdnn_voxceleb12.pth` | 83.3 MB | ECAPA-TDNN 说话人 | VoxCeleb1+2，20.77M |
| `mdtc_heysnips.pth` | 0.2 MB | MDTC 关键词 | HeySnips，0.034M |
| `pgnet_lite_totaltext.pth` | 5.1 MB | PGNet_lite 端到端 OCR | 轻量 e2e，1.22M |

### 2.2 OBB 专用（5 个 / 约 40 MB）—— **建议上传**

| 文件 | 大小 | 说明 |
|---|---|---|
| `yolo11n_obb_fw.pth` | 10.9 MB | yolo11n-OBB（DOTA 15 类，去掉函数式 dfl） |
| `v8n_obb_fw.pth` | 6.3 MB | yolov8n-OBB（框架格式） |
| `v26n_obb_fw.pth` | 5.6 MB | yolo26n-OBB（框架格式，reg_max=1） |
| `v8n-obb.pt` | 6.6 MB | yolov8n-OBB（ultralytics 原始） |
| `yolo26n-obb.pt` | 5.9 MB | yolo26n-OBB（ultralytics 原始） |

### 2.3 检测（2 个 / 约 11 MB）

| 文件 | 大小 | 说明 |
|---|---|---|
| `yolo26n.pt` / `yolo26n_det.pt` | 5.5 MB ×2 | yolo26n 检测（COCO） |

### 2.4 分割 nc=1 对齐实验（32 个 / 约 3.1 GB）—— **可按需上传**

> 这批是**为对齐验证生成**的 nc=1 权重（框架 `.pth` + ultra `.pt` 成对），
> 用于跨端同权重评估。**非通用模型**，若不对外发布对齐实验可**不上传**。

| 家族 | 档位 | 每档两文件（fw / ultra） |
|---|---|---|
| yolo11-seg | n/s/m/l/x | `yolo11{n,s,m,l,x}-seg_nc1.pth` + `_ultra.pt` |
| yolov8-seg | n/s/m/l/x | `yolov8{n,s,m,l,x}-seg_nc1.pth` + `_ultra.pt` |
| yolo26-seg | n/s/m/l/x | `yolo26{n,s,m,l,x}-seg_nc1.pth` + `_ultra.pt` |
| 合计 | 15 档 × 2 | 大小 11.6 MB(n) ~ 287 MB(x) |

- 另有 `yolo11n_seg_ft_fw.pth`（11.6 MB，package-seg 微调产物）。

### 2.5 临时/中间文件（1 个）

| 文件 | 大小 | 说明 |
|---|---|---|
| `.yolo26n.pt.edace82c...part` | 3.5 MB | **下载残留**，可删除 |

---

## 三、上传建议（按价值排序）

| 优先级 | 内容 | 大小 | 理由 |
|---|---|---|---|
| **P0** | 音频 4 个 + PGNet_lite | ~412 MB | **对外可用**：语音分类/说话人/KWS/轻量 e2e OCR，官方权重已转好且四条对齐全过 |
| **P0** | OBB 5 个 | ~40 MB | **对外可用**：DOTA OBB 是常见需求，已转为框架格式 |
| **P1** | 检测 2 个 | ~11 MB | yolo26n COCO 检测，通用 |
| **P2** | 分割 nc=1 32 个 | ~3.1 GB | **对齐实验用**，非通用模型；建议**不上传**或单独放 `_parity/` 目录 |

**最小上传集（推荐）**：**11 个文件 / 约 458 MB**
```
panns_cnn14.pth  ecapa_tdnn_voxceleb12.pth  mdtc_heysnips.pth  pgnet_lite_totaltext.pth
yolo11n_obb_fw.pth  v8n_obb_fw.pth  v26n_obb_fw.pth  v8n-obb.pt  yolo26n-obb.pt
yolo26n.pt  yolo26n_det.pt
```

---

## 四、上传方式

```python
# ModelScope 上传（需 token）
from modelscope.hub.api import HubApi
api = HubApi()
api.login('你的_token')
for f in [...]:
    api.upload_file(
        repo_id='ChaoII0987/TorchKiln',
        file_path=f'weights/{f}',          # 本地路径
        path_in_repo=f'pretrained/{f}',    # 远程路径（必须 pretrained/ 前缀）
    )
```

> ⚠️ **仓库必须是 public**（框架用公开 URL 下载，无 token）；私有会导致 resolve 404。
> 上传后验证：`resolve_pretrained('<裸名>')` 应能下载并加载 `missing=0/unexpected=0`。
