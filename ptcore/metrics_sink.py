"""结构化指标输出（JSONL）—— 供外部平台（AIStation 等）实时消费。

为什么不能靠"解析控制台日志"拿指标：
  1. 日志格式随版本变。第三方框架小版本一改，正则就**静默失效**（曲线断掉、best 取空，
     而且不报错——最难查的一类故障）；
  2. Python stdout 在管道下有缓冲（4~8KB 才 flush），曲线会一卡一卡；容器日志驱动还可能轮转；
  3. 日志行没有稳定的 step 对齐与时间戳，做不了 step 级曲线 / 吞吐 / 显存统计；
  4. 容器一挂，日志可能没了，也没有"断点续读"的概念。

指标必须是**有 schema、可断点续读**的机器契约。约定落在 ``Global.save_model_dir``
（即容器共享卷，宿主机可直接看到）下的两个文件：

  ``metrics.jsonl``  append-only，一行一个 JSON 事件；每行 flush，崩溃不坏；
                     消费方 ``tail -f`` 即可实时拿；断点续读只需记住 ``seq``。
  ``meta.json``      训练启动即写一次：模型/任务/参数量/主指标及方向/运行环境/配置快照路径。
                     原子替换写入，避免消费方读到半截。

事件 ``type``：

  ``step``  每 ``Global.print_batch_step`` 个 step 一条训练态
           (epoch/global_step/lr/loss/各 loss 分量/ips/显存/eta)
  ``eval``  每次评估一条（全部指标 + fps + 主指标）
  ``best``  主指标刷新最优时一条
  ``end``   训练结束一条（best 值/轮次/退出原因），无论正常结束/早停/异常都会写

关闭：环境变量 ``TKILN_METRICS=0``，或配置 ``Global.metrics_sink: false``。
关闭时返回 :class:`_NullSink`，调用点无需写 ``if``，热路径零开销。

⚠️ **契约边界（消费方必须知道）**：
  - ``end`` 事件只在 ``BaseTrainer.train()`` 收尾时写（含正常结束 / 早停 / 训练中抛异常）。
  - **训练还没开始就失败**（配置错误、权重下载失败、数据集路径不存在、模型构建报错——
    都发生在 ``__init__`` 里）**不会**有 ``metrics.jsonl``，只有进程非零退出码。
    所以服务侧判定规则应是：**"进程退出 且 没有 end 事件" = setup 失败**，
    不能只等 ``end``，否则任务会永远卡在 running。
  - 同理，断电/OOM kill 等 SIGKILL 也拿不到 ``end``——容器重启后按"最后 seq + 进程状态"判定。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import platform
import socket
import time

import numpy as np

__all__ = [
    "MetricSink",
    "build_sink",
    "sink_enabled",
    "runtime_env",
    "clean_value",
]

#: 环境变量里表示"关"的取值
_FALSY = {"0", "false", "no", "off", "none", ""}

#: torch **懒加载**：本模块也被 `config_schema` 依赖（`tkiln list/schema` 走这条路径），
#: 而 top-level import torch 要 ~5s。指标清洗在没装 torch 的环境也要能工作。
_TORCH = []


def _torch():
    if not _TORCH:
        try:
            import torch

            _TORCH.append(torch)
        except Exception:  # noqa: BLE001
            _TORCH.append(None)
    return _TORCH[0]


def sink_enabled(default=True):
    """环境变量 ``TKILN_METRICS`` 优先于配置默认值。

    Args:
        default: 配置里 ``Global.metrics_sink`` 的值（缺省 True）。
    """
    raw = os.environ.get("TKILN_METRICS")
    if raw is not None:
        return str(raw).strip().lower() not in _FALSY
    return bool(default)


def clean_value(v):
    """把配置/指标值转成**严格合法**的 JSON 值。

    - numpy 标量/张量 -> 原生数字（外部平台可能是 JS，``NaN``/``Infinity`` 不是合法 JSON，
      统一写 ``null``）；
    - 浮点保留 6 位有效小数，避免 jsonl 被 17 位尾数撑大；
    - 递归处理 dict/list/tuple/set。
    """
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return round(f, 6)
    if isinstance(v, np.ndarray):
        return v.tolist()
    torch = _torch()
    if torch is not None and torch.is_tensor(v):
        t = v.detach()
        if t.numel() == 1:
            return clean_value(t.float().cpu().item())
        return t.float().cpu().tolist()
    if isinstance(v, dict):
        return {str(k): clean_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean_value(x) for x in v]
    if isinstance(v, str):
        return v
    return str(v)


def finite_or_none(v):
    """标量版 :func:`clean_value`：``inf``/``NaN`` -> ``None``（best 初值可能是 inf）。"""
    out = clean_value(v)
    return out if isinstance(out, (int, float, type(None))) else None


def runtime_env():
    """运行环境快照，进 ``meta.json`` 便于复现与排障。"""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    torch = _torch()
    if torch is not None:
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        try:
            env["cudnn"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            pass
        try:
            env["cuda_available"] = bool(torch.cuda.is_available())
            env["gpu_name"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            )
        except Exception:  # noqa: BLE001
            pass
    return env

class _NullSink:
    """禁用时的空实现，保证调用点是 ``self.metrics_sink.step(...)`` 而无需 if 判断。"""

    enabled = False
    jsonl_path = None
    meta_path = None

    def write_meta(self, meta):
        pass

    def emit(self, type_, **fields):
        pass

    def step(self, **fields):
        pass

    def eval(self, **fields):
        pass

    def best(self, **fields):
        pass

    def end(self, **fields):
        pass

    def close(self):
        pass


class MetricSink(object):
    """append-only 的 JSONL 指标写入器。

    ⚠️ **写盘失败绝不能拖垮训练**——所有 I/O 都吞掉异常（只丢指标，不中断训练）。
    """

    enabled = True

    def __init__(self, out_dir, filename="metrics.jsonl"):
        self.out_dir = out_dir
        self.jsonl_path = os.path.join(out_dir, filename)
        self.meta_path = os.path.join(out_dir, "meta.json")
        self._seq = 0
        self._meta_written = False
        self._fh = None
        os.makedirs(out_dir, exist_ok=True)
        # newline="\n" 固定换行，消费方在 Windows 宿主上 tail 也不会被 \r\n 干扰
        self._fh = open(self.jsonl_path, "a", encoding="utf-8", newline="\n")

    # ---------------------------------------------------------------- 元信息
    def write_meta(self, meta):
        """写 ``meta.json``（幂等，只写第一次；原子替换）。"""
        if self._meta_written:
            return
        self._meta_written = True
        payload = clean_value(dict(meta or {}))
        payload.setdefault("schema_version", 1)
        payload.setdefault("metric_events", ["step", "eval", "best", "end"])
        payload["metrics_jsonl"] = os.path.basename(self.jsonl_path)
        try:
            tmp = self.meta_path + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(
                    payload, f, ensure_ascii=False, indent=2, sort_keys=False
                )
            os.replace(tmp, self.meta_path)
        except (OSError, TypeError, ValueError):
            # 元信息写不出去不影响训练
            pass

    # ---------------------------------------------------------------- 事件流
    def emit(self, type_, **fields):
        """写一行 JSON 事件。``fields`` 里值为 ``None`` 的键会被丢弃（保持行紧凑）。"""
        if self._fh is None:
            return
        rec = {"ts": round(time.time(), 3), "seq": self._seq, "type": type_}
        for k, v in fields.items():
            if v is None:
                continue
            rec[k] = clean_value(v)
        try:
            self._fh.write(
                json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n"
            )
            # 每条都 flush：消费方 tail -f 要能立刻看到，且崩溃时不丢已写指标
            self._fh.flush()
        except (OSError, TypeError, ValueError):
            return
        self._seq += 1

    def step(self, **fields):
        """训练态一条（与日志同节奏：``Global.print_batch_step``）。"""
        self.emit("step", **fields)

    def eval(self, **fields):
        """评估结果一条。"""
        self.emit("eval", **fields)

    def best(self, **fields):
        """主指标刷新最优一条。"""
        self.emit("best", **fields)

    def end(self, **fields):
        """训练结束一条，然后关闭文件。"""
        self.emit("end", **fields)
        self.close()

    # ---------------------------------------------------------------- 其它
    def close(self):
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.flush()
                fh.close()
            except (OSError, ValueError):
                pass


def build_sink(out_dir, enabled=True, meta=None):
    """工厂：返回可用 sink 或 :class:`_NullSink`（任何失败都降级为不写指标）。

    Args:
        out_dir: 产物目录（``Global.save_model_dir``）。
        enabled: 是否启用；DDP 下只应让 rank0 传 True。
        meta: 非空时顺带写 ``meta.json``。
    """
    if not enabled:
        return _NullSink()
    try:
        sink = MetricSink(out_dir)
    except (OSError, ValueError):
        return _NullSink()
    if meta:
        sink.write_meta(meta)
    return sink
