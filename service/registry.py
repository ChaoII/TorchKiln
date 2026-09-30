"""模型注册表：把 ``configs/**.yml`` 暴露成 HTTP 接口。

这是"外部平台不硬编码任何超参映射表"的落点——AIStation 前端的下拉框与参数表单
全部由这两个接口动态生成：**加模型 = 丢一个 YAML**，平台侧无需发版。
"""
from __future__ import absolute_import

import json
import os


def _ensure_path(repo_root):
    import sys

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def list_models(repo_root, task=None, family=None, name=None, subdir=None):
    _ensure_path(repo_root)
    from ptcore.config_schema import list_models as _list

    rows = _list()
    if task:
        rows = [r for r in rows if (r.get("task") or "") == task]
    if family:
        rows = [r for r in rows if (r.get("model_family") or "") == family]
    if name:
        low = name.lower()
        rows = [r for r in rows if low in (r.get("model_name") or "").lower()]
    if subdir:
        rows = [r for r in rows if subdir in (r.get("config_path") or "")]
    return rows


def describe_model(repo_root, model_name, config_path=None, overrides=None):
    _ensure_path(repo_root)
    from ptcore.config_schema import describe_config, resolve_config

    if config_path:
        path = config_path
        if not os.path.isabs(path):
            path = os.path.join(repo_root, path)
    else:
        path, _how = resolve_config(model_name)
    return describe_config(path, overrides=overrides or None)


def get_summary(repo_root, model_name):
    """单个模型的一行摘要（用于 GET /models/{name}）。"""
    from ptcore.config_schema import resolve_config

    _ensure_path(repo_root)
    path, _how = resolve_config(model_name)
    info = describe_model(repo_root, model_name, config_path=path)
    return {
        "model_name": info.get("model_name"),
        "config_path": info.get("config_path"),
        "task": info.get("task"),
        "model_family": info.get("model_family"),
        "algorithm": info.get("algorithm"),
        "scale": info.get("scale"),
        "main_indicator": info.get("main_indicator"),
        "main_indicator_mode": info.get("main_indicator_mode"),
        "defaults": info.get("defaults"),
        "data": info.get("data"),
        "groups": info.get("groups"),
        "param_count": len(info.get("params") or {}),
    }


#: 训练环境运行时版本缓存。``import torch`` 要几十秒（本机实测 23s），且版本在
#: 进程生命周期内不变，所以首次探测后就地缓存，后续 ``GET /info`` 直接返回。
_RUNTIME_CACHE = None

#: 子进程只负责"把版本读出来"，**不做任何格式化**——格式化规则只在主进程的
#: ``_fmt_cudnn`` 里存在一份，避免两边各写一套、悄悄跑偏。
_PROBE_CODE = r"""
import json, sys

out = {"python": None, "torch": None, "cuda": None, "cudnn_raw": None,
       "device": None, "device_count": 0}
try:
    out["python"] = ".".join(str(v) for v in sys.version_info[:3])
except Exception:
    pass
try:
    import torch

    out["torch"] = getattr(torch, "__version__", None)
    out["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        out["cudnn_raw"] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            out["device_count"] = int(torch.cuda.device_count())
            out["device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
except Exception:
    pass
print(json.dumps(out))
"""


def _fmt_cudnn(version):
    """cuDNN 版本整数转人类可读。

    编码规则在 cuDNN 8 处换过一次：
      - cuDNN >= 8 是 5 位 ``major*10000 + minor*100 + patch``（90107 -> 9.1.7）
      - cuDNN <= 7 是 4 位 ``major*1000 + minor*100 + patch``（7605  -> 7.6.5）
    只按 5 位算会把老版本读成 0.76.5 这种天书，故按位数分支。
    """
    if version >= 10000:
        return "{}.{}.{}".format(version // 10000, (version // 100) % 100,
                                 version % 100)
    return "{}.{}.{}".format(version // 1000, (version // 100) % 10, version % 100)


def _empty_runtime():
    return {"python": None, "torch": None, "cuda": None, "cudnn": None,
            "device": None, "device_count": 0}


def _probe_runtime_in_subprocess(timeout=180.0):
    """在**独立子进程**里读 torch 版本。

    为什么不能在本进程 import torch：本机实测 ``import torch`` 要 23 秒，且
    期间**长期持有 GIL**。放线程池也一样会把事件循环饿死——训练进行中这 23 秒
    里 SSE 心跳与日志/指标转发全部停摆，平台侧会误判服务已死。子进程有独立
    解释器和独立 GIL，主进程只是一次 ``subprocess.wait``，事件循环照常跑。
    """
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            capture_output=True, text=True, timeout=timeout,
        )
        lines = (proc.stdout or "").strip().splitlines()
        if lines:
            data = json.loads(lines[-1])
            if isinstance(data, dict):
                info = _empty_runtime()
                info["python"] = data.get("python")
                info["torch"] = data.get("torch")
                info["cuda"] = data.get("cuda")
                info["device"] = data.get("device")
                try:
                    info["device_count"] = int(data.get("device_count") or 0)
                except (TypeError, ValueError):
                    info["device_count"] = 0
                raw = data.get("cudnn_raw")
                if raw:
                    try:
                        info["cudnn"] = _fmt_cudnn(int(raw))
                    except (TypeError, ValueError):
                        pass
                return info
    except Exception:  # noqa: BLE001
        pass
    return _empty_runtime()


def runtime_info():
    """训练环境的运行时版本（Python / torch / CUDA / cuDNN / GPU）。

    这几个值只有**装了 torch 的训练进程**才报得出来——平台侧后端本身不装 torch，
    所以由本服务上报，平台只做汇总展示。

    任何一步失败都降级成 ``None``：无 torch 的单测环境、CPU-only 构建、驱动缺失
    都不该把 ``GET /info`` 打挂。
    """
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE
    _RUNTIME_CACHE = _probe_runtime_in_subprocess()
    return _RUNTIME_CACHE


def warm_runtime_info():
    """启动期预热：把 ``import torch`` 的几秒开销挪到服务启动，而不是首个 ``/info``。"""
    try:
        runtime_info()
    except Exception:  # noqa: BLE001
        pass


def framework_info(repo_root):
    """服务与框架的版本/能力声明，供平台侧做兼容性判断。"""
    _ensure_path(repo_root)
    try:
        from torchkiln import __version__

        version = __version__
    except Exception:  # noqa: BLE001
        version = "0.0.0"
    from ptcore.config_schema import CONFIG_ROOT

    return {
        "framework": "torchkiln",
        "framework_version": version,
        "spec_version": "1.0",
        "api_version": "v1",
        "runtime": runtime_info(),
        "config_root": os.path.relpath(CONFIG_ROOT, repo_root).replace("\\", "/"),
        "metrics": {
            "file": "metrics.jsonl",
            "meta": "meta.json",
            "events": ["step", "eval", "best", "end"],
            "note": "指标真值在 output_dir 的 JSONL 文件里，服务只做转发不做二次存储",
        },
        "job": {
            "idempotency_header": "Idempotency-Key",
            "cancel": "POST /api/v1/train/jobs/{job_id}/cancel",
            "streams": ["logs/stream", "metrics/stream"],
        },
    }
