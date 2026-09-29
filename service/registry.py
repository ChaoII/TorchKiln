"""模型注册表：把 ``configs/**.yml`` 暴露成 HTTP 接口。

这是"外部平台不硬编码任何超参映射表"的落点——AIStation 前端的下拉框与参数表单
全部由这两个接口动态生成：**加模型 = 丢一个 YAML**，平台侧无需发版。
"""
from __future__ import absolute_import

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
