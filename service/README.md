# TorchKiln 训练服务

把 TorchKiln 包成 HTTP 服务，供 AIStation 这类平台调用。**目标：让平台侧不再维护任何
超参映射表、不再解析控制台日志。**

## 为什么这样切

| 痛点（平台侧现状） | 本服务的做法 |
|---|---|
| 正则解析容器控制台日志取指标，框架一改格式就**静默失效** | 训练侧产出 `metrics.jsonl` 机器契约，服务只 tail 转发 |
| 加一个模型要改「后端映射表 + 前端表单 + 重新发布」 | `GET /models` + `/models/{name}/schema`，**加模型 = 丢一个 YAML** |
| 超参靠从命令行反推 | `JobSpec` 声明式 + 原样存库，天生可复现 |
| 客户端超时重试 → 重复排队烧卡 | `Idempotency-Key`，重复提交返回同一 `job_id` |
| 刷新页面曲线丢失 / 重连指标翻倍 | SSE `Last-Event-ID` + `seq` 游标补发 |
| 服务重启 → 一排"运行中"变僵尸 | 作业库落 SQLite + 重启按文件契约接管 |
| 任务卡在 running（训练没开始就失败，没 end 事件） | 「进程退出 且 无 end 事件」= `setup_failed` |

## 分工边界（重要）

| 关注点 | 归属 |
|---|---|
| 模型/算法/trainer、配置 schema、GPU 排队与并发、容器生命周期、指标产出 | **本服务** |
| 数据集与标注、用户/权限/审批/审计/计费、业务状态机、模型注册与上线、指标历史存储 | **AIStation** |

两条铁律：
1. **指标真值永远是 `<output_dir>/metrics.jsonl`**，本服务只转发不二次存储（避免双写口径不一致）。
2. **单一排队点**：GPU 排队只在服务端。AIStation 侧只做提交节流，不要再排一次队。

## 快速开始

```bash
pip install -r service/requirements.txt

# 本地：直接用现有 conda 环境
export TKILN_PYTHON=/path/to/envs/ultraly/bin/python   # 起训练进程用的解释器
export TKILN_REPO_ROOT=/path/to/TorchKiln
export TKILN_DATA_ROOT=/data/torchkiln                 # 产物根（容器内即共享卷）
export TKILN_SERVICE_TOKEN=<自选 token>                 # 不设则不校验（仅限本地）
export TKILN_MAX_CONCURRENT=1                          # ⚠️ 单卡必须 1

python -m uvicorn service.main:app --host 0.0.0.0 --port 8000
```

交互式文档：`http://127.0.0.1:8000/docs`

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `TKILN_PYTHON` | 当前解释器 | 起训练进程用的 Python |
| `TKILN_REPO_ROOT` | 仓库根 | `configs/...` 相对它解析 |
| `TKILN_DATA_ROOT` | `output/_service` | 作业产物根 |
| `TKILN_DB` | `<data_root>/service.db` | 作业状态库（SQLite） |
| `TKILN_MAX_CONCURRENT` | `1` | 并发训练数。**单卡务必 1** |
| `TKILN_SERVICE_TOKEN` | 空 | 非空则要求 `Authorization: Bearer <token>` |
| `TKILN_POLL_INTERVAL` | `1.0` | tail 指标文件轮询间隔（秒） |
| `TKILN_LOG_RING` | `2000` | 内存保留日志行数 |
| `TKILN_HEARTBEAT` | `15` | SSE 心跳（秒） |
| `TKILN_CANCEL_GRACE` | `10` | 取消时 SIGTERM 到 SIGKILL 的宽限秒数 |

---

## API 契约

### 0. 基础

```
GET  /healthz
GET  /api/v1/info                 # 框架版本、指标契约说明、能力声明
```

### 1. 模型自描述（供前端动态生成下拉框与参数表单）

```
GET /api/v1/models?task=&model_family=&name=&subdir=
GET /api/v1/models/{model_name}
GET /api/v1/models/{model_name}/schema?o=Global.epoch_num=300
```

`/models` 每项：
```json
{ "model_name": "yolov8-det", "config_path": "configs/yolo/yolov8-det.yml",
  "task": "detect", "model_family": "yolo", "algorithm": "yolov8", "scale": "n",
  "main_indicator": "mAP50-95", "main_indicator_mode": "max",
  "epoch_num": 100, "has_pretrained": false }
```

`/schema`（`yolov8-det` 实测 64 个可覆盖参数 / 8 个分组）：
```json
{
  "schema_version": 1, "model_name": "yolov8-det", "task": "detect",
  "main_indicator": "mAP50-95", "main_indicator_mode": "max",
  "groups": ["Global", "Optimizer", "Architecture", "Loss", "Metric", "Train", "Eval", "PostProcess"],
  "managed_keys": ["Global.checkpoints", "Global.save_model_dir"],
  "params": {
    "Global.epoch_num": {
      "key": "Global.epoch_num", "label": "训练轮数", "group": "Global",
      "type": "int", "default": 100, "min": 1, "max": 2000, "widget": "slider"
    },
    "Global.use_ema": { "label": "启用 EMA", "type": "bool", "default": true, "widget": "switch" }
  }
}
```

`widget` 取值：`switch` / `slider` / `number` / `text` / `path` / `list`。
`managed_keys` 是**平台托管、不该进用户表单**的键（输出目录、续训权重）。
中文字段名走「内置常见键表 + 配置内 `_ui` 段覆盖」：

```yaml
_ui:
  Global.epoch_num: { label: "训练轮数", min: 1, max: 500 }
```

### 2. 训练作业

```
POST /api/v1/train/jobs                    # 202 + JobCreated
GET  /api/v1/train/jobs?status=&user_id=&limit=&offset=
GET  /api/v1/train/jobs/{job_id}
POST /api/v1/train/jobs/{job_id}/cancel
```

请求头：`Idempotency-Key`（**强烈建议必带**）、`X-User-Id`、`X-Tenant`

请求体（`JobSpec`，**声明式，原样存库**）：
```json
{
  "spec_version": "1.0",
  "model_name": "yolov8n-det",
  "params": { "Global.epoch_num": 100, "Global.use_ema": true },
  "dataset": { "data_dir": "/data/ds", "train_list": "/data/ds/train.txt",
               "val_list": "/data/ds/val.txt" },
  "resources": { "gpu": 1, "gpu_memory_gb": 12, "shm_size": "4g", "priority": 5 },
  "seed": 1024,
  "labels": { "dataset_version": "v3" }
}
```

`params` 用点分键（就是 `-o Key.Sub=value` 的形态，**零映射**）。服务端会强制注入
`Global.save_model_dir` 与 `Global.metrics_sink=True`，外部覆盖无效。

响应 `202`：
```json
{ "job_id": "job_20260929173737_b61eb0fe", "status": "queued",
  "idempotent_hit": false, "created_ts": 1790674657.1 }
```

作业状态：`queued` → `running` → `succeeded` / `failed` / `cancelled`

| `exit_reason` | 含义 |
|---|---|
| `finished` / `early_stop` / `already_complete` | 正常结束（含早停），`status=succeeded` |
| `error` | 训练中抛异常，`error` 字段带异常摘要 |
| `setup_failed` | **训练没开始就失败**（配置/权重/数据集/依赖），无指标文件 |
| `killed` | 训练中被强杀（OOM/SIGKILL），有指标但无 `end` |
| `cancelled` | 用户取消 |
| `no_end_event` | 退出码 0 但没有 `end`，状态未知（应视为异常） |
| `orphaned` / `lost_on_restart` | 服务重启导致的状态收敛 |
| `spawn_failed` / `internal_error` | 服务自身问题 |

⚠️ **消费方要点**：不能只等 `end` 事件。判定规则是
**「进程退出 且 无 end 事件」= setup 失败**——否则任务会永远卡在 running。

### 3. 日志与指标

```
GET /api/v1/train/jobs/{id}/logs?tail=200
GET /api/v1/train/jobs/{id}/logs/stream?tail=50        (SSE)
GET /api/v1/train/jobs/{id}/metrics?offset=-1&limit=5000
GET /api/v1/train/jobs/{id}/metrics/stream?offset=      (SSE)
```

指标事件（与 `metrics.jsonl` 逐字一致）：
```
event: step     id: 0
data: {"ts":...,"seq":0,"type":"step","epoch":1,"global_step":10,"lr":0.01,
       "loss":4.35,"comps":{"loss_cls":1.2},"ips":12.3,"mem_reserved":2048,"eta_sec":420}

event: eval     id: 1
data: {"type":"eval","epoch":10,"main_indicator":"mAP50-95","main_value":0.795,
       "metrics":{"mAP50":0.9,"mAP50-95":0.795},"is_best":true}

event: best     id: 2
data: {"type":"best","main_value":0.795,"checkpoint":"best_accuracy.pth"}

event: end      id: 8
data: {"type":"end","exit_reason":"finished","best_metric":0.795,"duration_sec":3600.2}
```

**断点续传**：每帧带 SSE `id: <seq>`。客户端重连时带 `Last-Event-ID`（或 `?offset=`），
服务端**先补发缺口再切实时**——已消费过的不会重发。作业结束时服务端会推一帧
`event: end` 后**主动收流**，客户端看到流结束即知作业已终结，不必再轮询。

---

## 产物目录布局

```
<TKILN_DATA_ROOT>/jobs/<job_id>/
  ├── metrics.jsonl     指标真值（append-only，服务只 tail 它）
  ├── meta.json         模型/任务/参数量/主指标及方向/运行环境/覆盖项
  ├── config.yml        训练时的完整配置快照
  ├── train.log         训练侧日志
  ├── service.log       服务捕获的 stdout/stderr
  ├── best_accuracy.pth / final.pth / latest.pth / epoch_N.pth
```

## 验证脚本

```bash
python service/_sse_check.py   http://127.0.0.1:8000 <token> yolov8n-det
python service/_edge_check.py  http://127.0.0.1:8000 <token>
python service/_recover_check.py
```

- `_sse_check.py`：实时指标流 + `Last-Event-ID` 断点续传 + 日志流
- `_edge_check.py`：取消 / setup 失败 / 并发闸门 / 非法请求
- `_recover_check.py`：强杀服务 → 重启 → 接管进行中的训练
