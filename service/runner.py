"""训练执行器：把声明式 JobSpec 变成一个训练进程，并消费它的 metrics.jsonl。

职责边界（与 AIStation 的分工）：
  - 本模块只管 **执行 + 产出**：起进程、吐日志、tail 指标、判终态；
  - **不存指标历史**（真值永远是 ``<output_dir>/metrics.jsonl``），
  - 不做业务状态机/审批/权限/计费（那是 AIStation 的事）。

⚠️ 终态判定规则（对应 ptcore/metrics_sink 的契约说明）：
  训练**开始后**结束一定会写 ``end`` 事件；但**训练还没开始就失败**（配置错、
  权重下载失败、数据集路径不存在、模型构建报错——都发生在 trainer 构造阶段）
  **不会有任何指标文件**，只有进程非零退出码。
  所以：**"进程退出 且 没有 end 事件" = setup 失败**。
  如果只等 end，任务会永远卡在 running——这是最容易踩的坑。
"""
from __future__ import absolute_import

import asyncio
import json
import os
import signal
import time

from .streams import sse_event

#: 指标文件里允许出现的终态原因 -> 作业终态
_EXIT_REASON_TO_STATUS = {
    "finished": "succeeded",
    "already_complete": "succeeded",
    "early_stop": "succeeded",
    "error": "failed",
}


# ------------------------------------------------------------------ argv 构造
def build_train_argv(spec, config_abs_path, output_dir, python_exe, repo_root):
    """把 JobSpec 翻译成 ``python -m torchkiln train ...`` 参数列表。

    约定：
      - 平台注入的键（``Global.save_model_dir`` 等）由服务端强制覆盖，
        防止外部把权重写到任意路径；
      - ``params`` 里的点分键原样转成 ``-o k=v``，**零映射**——这就是 TorchKiln
        自研带来的最大好处，平台侧不需要维护任何超参映射表；
      - ``dataset`` 只做「路径注入」，不关心数据从哪来。
    """
    argv = [python_exe, "-m", "torchkiln", "train",
            "-c", os.path.relpath(config_abs_path, repo_root).replace("\\", "/")]

    opts = []

    def add(key, value):
        if value is None:
            return
        if isinstance(value, bool):
            opts.append("{}={}".format(key, "true" if value else "false"))
        elif isinstance(value, (list, tuple)):
            # 列表走 JSON 字面量，与 ptcore.config._parse_value 的解析一致
            opts.append("{}={}".format(key, json.dumps(list(value), ensure_ascii=False)))
        else:
            opts.append("{}={}".format(key, value))

    ds = spec.dataset
    if ds is not None:
        add("Train.dataset.data_dir", ds.data_dir)
        if ds.train_list:
            add("Train.dataset.label_file_list", [ds.train_list])
        if ds.val_list:
            add("Eval.dataset.data_dir", ds.data_dir)
            add("Eval.dataset.label_file_list", [ds.val_list])

    for key, value in (spec.params or {}).items():
        add(key, value)

    if spec.seed is not None:
        add("Global.seed", spec.seed)

    # 平台侧托管、不可被外部覆盖的键（放在最后，覆盖前面的同名项）
    add("Global.save_model_dir", output_dir)
    # 指标 JSONL 是本服务的立身之本，不允许外部关掉
    add("Global.metrics_sink", True)

    if opts:
        argv += ["-o"] + opts
    return argv


# ------------------------------------------------------------------ 指标文件
def read_metrics_since(path, offset_bytes):
    """从字节偏移增量读取 JSONL。

    Returns:
        ``(events, new_offset)``；文件被截断/重建时自动从头读（offset 归零）。
    """
    if not path or not os.path.isfile(path):
        return [], offset_bytes
    try:
        size = os.path.getsize(path)
        if offset_bytes > size:
            # 文件被截断（重跑/清理），只能从头开始
            offset_bytes = 0
        if size == offset_bytes:
            return [], offset_bytes
        with open(path, "rb") as f:
            f.seek(offset_bytes)
            raw = f.read()
        new_offset = offset_bytes + len(raw)
    except OSError:
        return [], offset_bytes

    events = []
    # 最后一行可能还没写完（正在 flush 的中间态），留给下一轮
    *lines, tail = raw.split(b"\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            continue   # 坏行跳过，不让消费方整体崩掉
    if tail.strip():
        try:
            events.append(json.loads(tail.decode("utf-8")))
            new_offset += len(tail) + 1
        except (ValueError, UnicodeDecodeError):
            pass
    return events, new_offset


def read_last_event(path, event_type="end"):
    """取文件里最后一条指定类型的事件（终态判定用）。"""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            try:
                f.seek(max(0, os.path.getsize(path) - 65536))  # 末尾 64KB 足够
            except OSError:
                f.seek(0)
            chunk = f.read()
    except OSError:
        return None
    found = None
    for line in reversed(chunk.split(b"\n")):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if rec.get("type") == event_type:
            return rec
    return None


def pid_alive(pid):
    """跨平台判活（恢复用）。"""
    if not pid:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                return True
            finally:
                kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


# ------------------------------------------------------------------ 运行器
class TrainingRun(object):
    """一个正在跑的（或跑过的）训练作业。"""

    def __init__(self, job_id, spec, argv, output_dir, cwd, python_exe,
                 env=None, log_ring=None):
        self.job_id = job_id
        self.spec = spec
        self.argv = argv
        self.output_dir = output_dir
        self.cwd = cwd
        self.python_exe = python_exe
        self.env = env
        self.log_path = os.path.join(output_dir, "service.log")
        self.metrics_path = os.path.join(output_dir, "metrics.jsonl")
        self.log_ring = log_ring
        self.proc = None
        self.pid = None
        #: 子进程 stdout（stderr 已合并进来）。**必须由 jobs.py 在 spawn 之后显式
        #: 赋值**——曾经漏赋值导致 ``_pump_logs`` 首行 AttributeError 被静默吞掉，
        #: 结果 service.log 永远 0 字节、SSE 日志流与 /logs 接口全空。
        self.stdout = None
        self._metrics_offset = 0
        self.cancel_requested = False
        self.exit_code = None
        self.end_event = None

    def metrics_since(self, offset_bytes):
        return read_metrics_since(self.metrics_path, offset_bytes)


def _spawn_env(base_env, extra=None):
    env = dict(base_env or os.environ)
    # 强制打开指标 sink：服务依赖它，拿不到就是故障
    env["TKILN_METRICS"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")   # 日志实时，别被 4KB 缓冲卡住
    if extra:
        env.update(extra)
    return env


def send_signal(proc, hard=False):
    """给训练进程发终止信号（POSIX 下按进程组杀，避免 DataLoader worker 变孤儿)。。

    宽限等待由调用方用 ``asyncio.wait_for`` 控制；这里只负责发信号。
    """
    if proc is None or proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            proc.kill() if hard else proc.terminate()
        elif hard:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass   # 进程已经没了，视为终止成功


def classify_exit(end_event, exit_code, cancel_requested, has_metrics_file):
    """把 (end 事件, 退出码, 是否取消) 收敛成 (status, exit_reason, error)。

    这是整套判定规则的**唯一**实现，实时路径与重启恢复路径共用它——
    避免两条路径给出不一致的结论。
    """
    if cancel_requested:
        return "cancelled", "cancelled", None

    if end_event is None:
        # 没有 end：训练没真正开始（构造阶段就炸了）或被 SIGKILL
        if exit_code not in (0, None):
            hint = ("训练未开始即失败（trainer 构造阶段异常：配置/权重/数据集/依赖问题）"
                    if not has_metrics_file else "训练进程被强杀，未写出 end 事件")
            return "failed", "setup_failed" if not has_metrics_file else "killed", hint
        return "failed", "no_end_event", "进程退出但没有 end 事件，状态未知"

    reason = end_event.get("exit_reason") or "finished"
    status = _EXIT_REASON_TO_STATUS.get(reason, "failed")
    err = end_event.get("error")
    if reason == "error":
        return "failed", "error", err or "训练异常终止"
    return status, reason, err


def sse_metric_frame(event):
    """指标事件 -> SSE 帧；``id`` 用 seq，客户端据此做 Last-Event-ID 续传。"""
    return sse_event(event, event=event.get("type", "step"), event_id=event.get("seq"))


def sse_log_frame(line):
    return sse_event(line, event="log")
