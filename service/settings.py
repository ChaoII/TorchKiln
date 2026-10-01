"""服务配置：全部来自环境变量，便于容器化（12-factor）。

| 变量 | 默认 | 说明 |
|---|---|---|
| ``TKILN_PYTHON`` | 当前解释器 | 起训练进程用的 Python（容器内与宿主 conda 环境分别指定） |
| ``TKILN_REPO_ROOT`` | 仓库根 | TorchKiln 代码根，``-c configs/...`` 相对它解析 |
| ``TKILN_DATA_ROOT`` | ``<repo>/output/_service`` | 作业产物根（容器内即共享卷挂载点） |
| ``TKILN_DB`` | ``<data_root>/service.db`` | 作业状态库（SQLite，重启可恢复） |
| ``TKILN_MAX_CONCURRENT`` | ``1`` | 同时运行的训练数。**单卡务必 1**，显存不够会 OOM |
| ``TKILN_SERVICE_TOKEN`` | 空 | 非空则要求 ``Authorization: Bearer <token>`` |
| ``TKILN_POLL_INTERVAL`` | ``1.0`` | tail ``metrics.jsonl`` 的轮询间隔（秒） |
| ``TKILN_LOG_RING`` | ``2000`` | 每个作业内存里保留的日志行数（供断线补发） |
| ``TKILN_HEARTBEAT`` | ``15`` | SSE 心跳间隔（秒），防代理掐断长连接 |
| ``TKILN_CANCEL_GRACE`` | ``10`` | 取消时先 SIGTERM、宽限多少秒再 SIGKILL |
"""
from __future__ import absolute_import

import os

#: service/ 的上一级即仓库根
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPO = os.path.dirname(_HERE)


def _env_str(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name, default):
    try:
        return int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default


class Settings(object):
    """一次性读全 env；测试里可直接 ``Settings(repo_root=..., data_root=...)`` 覆盖。"""

    def __init__(self, **overrides):
        import sys

        self.python_exe = overrides.get("python_exe") or _env_str(
            "TKILN_PYTHON", sys.executable)
        self.repo_root = os.path.abspath(
            overrides.get("repo_root")
            or _env_str("TKILN_REPO_ROOT", _DEFAULT_REPO)
        )
        self.data_root = os.path.abspath(
            overrides.get("data_root")
            or _env_str("TKILN_DATA_ROOT", os.path.join(self.repo_root, "output", "_service"))
        )
        self.db_path = overrides.get("db_path") or _env_str(
            "TKILN_DB", os.path.join(self.data_root, "service.db"))
        # 单卡显存有限：默认严格串行。多卡机器可调大，或改成按 GPU 池分配。
        self.max_concurrent = overrides.get("max_concurrent") or _env_int(
            "TKILN_MAX_CONCURRENT", 1)
        self.service_token = overrides.get("service_token") or _env_str(
            "TKILN_SERVICE_TOKEN", "")
        self.poll_interval = overrides.get("poll_interval") or _env_float(
            "TKILN_POLL_INTERVAL", 1.0)
        self.log_ring = overrides.get("log_ring") or _env_int("TKILN_LOG_RING", 2000)
        # ---- 构建身份（用于判断 job 镜像是否过期）----
        #: 构建时注入的 TorchKiln git 修订号。Dockerfile 用 ARG/ENV 写入；
        #: 本地直接跑（不构建镜像）时为 ``unknown``——那不是过期，别误报。
        self.code_revision = overrides.get("code_revision") or _env_str(
            "TKILN_REVISION", "unknown")
        #: 构建时源码工作区是否有未提交改动。是 ``True`` 时镜像不可复现，
        #: 但**不**据此拒绝运行——本地开发构建带改动是常态。
        raw_dirty = (overrides.get("code_dirty")
                     or _env_str("TKILN_DIRTY", ""))
        self.code_dirty = str(raw_dirty).strip().lower() in ("1", "true", "yes", "on")

        self.heartbeat = overrides.get("heartbeat") or _env_float(
            "TKILN_HEARTBEAT", 15.0)
        self.cancel_grace = overrides.get("cancel_grace") or _env_float(
            "TKILN_CANCEL_GRACE", 10.0)

    def ensure_dirs(self):
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path) or self.data_root, exist_ok=True)
