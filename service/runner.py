"""作业执行器：把声明式 JobSpec 变成一个进程，并消费它的 metrics.jsonl。

支持两种 kind（分派见 :func:`build_argv`）：

  ``train``  ``python -m torchkiln train ...``
  ``eval``   ``python -m torchkiln val ...``，产物是**指标**而非权重

两者共用同一套进程管理 / 日志泵 / 指标契约 / 终态判定——这不是为了少写代码，
而是**强制它们共享同一个可观测性契约**：只要走这个服务的作业，外部就都能用
同一套 HTTP 接口与 SSE 事件消费。评估若绕开契约自己 print 日志，调用方就得
为它单独写一套解析，而日志格式恰恰是最容易在小版本改动中静默失效的东西。

职责边界（与 AIStation 的分工）：
  - 本模块只管 **执行 + 产出**：起进程、吐日志、tail 指标、判终态；
  - **不存指标历史**（真值永远是 ``<output_dir>/metrics.jsonl``），
  - 不做业务状态机/审批/权限/计费（那是 AIStation 的事）。

⚠️ 终态判定规则（对应 ptcore/metrics_sink 的契约说明）：
  作业**开始后**结束一定会写 ``end`` 事件；但**还没开始就失败**（配置错、
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


def build_eval_argv(spec, config_abs_path, output_dir, python_exe, repo_root):
    """把 JobSpec 翻译成 ``python -m torchkiln val ...`` 参数列表。

    与训练的几处**刻意不同**：

    - ``--weights`` 走**命令行参数**而不是 ``-o Global.pretrained_model=``。
      ``tools/eval.py`` 会把它写进 ``Global.pretrained_model``，两种写法等价，
      但用参数更明确：评估权重是被评估对象的身份，不该混在"可调超参"里。
    - 依然强制 ``Global.save_model_dir`` 与 ``Global.metrics_sink``：
      评估的产物就是指标，不落到本服务的 output_dir 就没人读得到。

    ⚠️ **必须同时注入 ``Train.dataset.*``**，即使评估根本不迭代训练集。

    原因：``build_trainer`` 在 ``BaseTrainer.__init__`` 里**无条件**构造两个数据集
    （``train_dataset, eval_dataset = task.build_datasets(...)``），构造阶段就会去
    ``open(label_file)``。只注入 ``Eval.dataset.*`` 的话，训练集会退回配置里的默认
    路径（实测报 ``FileNotFoundError: datasets/seg_demo/train.txt``），
    作业在 trainer 构造阶段就死掉——表现为 ``setup_failed``。

    ``tools/eval.py`` 已把 ``Global.epoch_num`` 设为 0，训练集构造完不会被迭代，
    所以指向哪份清单并不影响评估结果。这里给的是**评估那份**（而不是另找 train.txt）：
    调用方只需要导出一份清单，契约更小，也不会出现「train.txt 不在挂载里」的失败。
    """
    argv = [python_exe, "-m", "torchkiln", "val",
            "-c", os.path.relpath(config_abs_path, repo_root).replace("\\", "/")]

    if spec.weights_path:
        argv += ["--weights", str(spec.weights_path)]

    opts = []

    def add(key, value):
        if value is None:
            return
        if isinstance(value, bool):
            opts.append("{}={}".format(key, "true" if value else "false"))
        elif isinstance(value, (list, tuple)):
            opts.append("{}={}".format(key, json.dumps(list(value), ensure_ascii=False)))
        else:
            opts.append("{}={}".format(key, value))

    ds = spec.dataset
    if ds is not None:
        add("Eval.dataset.data_dir", ds.data_dir)
        if ds.val_list:
            add("Eval.dataset.label_file_list", [ds.val_list])
            # 见 docstring：trainer 构造期会打开这两个文件，缺一个就 setup 失败。
            # 给同一份清单即可——epoch_num=0 时它不会被迭代。
            add("Train.dataset.data_dir", ds.data_dir)
            add("Train.dataset.label_file_list", [ds.val_list])

    for key, value in (spec.params or {}).items():
        add(key, value)

    if spec.seed is not None:
        add("Global.seed", spec.seed)

    add("Global.save_model_dir", output_dir)
    add("Global.metrics_sink", True)

    if opts:
        argv += ["-o"] + opts
    return argv


def _posix_join(base, *parts):
    """用 ``/`` 拼路径，**不用** ``os.path.join``。

    ⚠️ 服务跑在 Linux 容器里，但 spec 的路径常常由 Windows 上的平台侧构造；
    ``os.path.join("/workspace/jobs/job_1", "predict_results")`` 在 Windows 上
    会得到 ``/workspace/jobs/job_1\\predict_results``。它在自己的宿主上看着
    完全正常，传进容器就找不到文件——这类错没有任何本地信号。
    """
    out = str(base).rstrip("/\\")
    for p in parts:
        seg = str(p).strip("/\\")
        if seg:
            out = out + "/" + seg
    return out


def build_predict_argv(spec, config_abs_path, output_dir, python_exe, repo_root):
    """把 JobSpec 翻译成 ``python -m torchkiln predict ...`` 参数列表。

    与 eval 的三处**刻意不同**：

    - ``--input`` 传**图片目录**（``DatasetRef.data_dir`` 不适用——预测不吃清单），
      所以走 ``JobSpec.input_dir`` 这个专用字段；
    - ``--output`` 是**结果图目录**，指向本作业的 ``output_dir``，于是结果图直接
      落在挂载出来的共享卷上，宿主无需再从容器里拷；
    - ``--metrics-dir`` **必须显式传**：``--output`` 的语义随输入而变（单图=文件、
      批量=目录），从它反推 metrics.jsonl 该写哪是不可靠的。而外部调度正是靠那个
      文件里的 ``end`` 事件判终态，写错地方等于每次预测都被判失败。

    不注入 ``Global.save_model_dir``：预测的产物是 ``--output`` 指定的图，
    与训练/评估的权重落盘无关。
    """
    argv = [python_exe, "-m", "torchkiln", "predict",
            "-c", os.path.relpath(config_abs_path, repo_root).replace("\\", "/")]

    if spec.weights_path:
        argv += ["--weights", str(spec.weights_path)]
    if spec.input_dir:
        argv += ["--input", str(spec.input_dir)]
    argv += ["--output", _posix_join(output_dir, "predict_results")]
    argv += ["--metrics-dir", output_dir]

    opts = []
    for key, value in (spec.params or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            opts.append("{}={}".format(key, "true" if value else "false"))
        elif isinstance(value, (list, tuple)):
            opts.append("{}={}".format(key, json.dumps(list(value), ensure_ascii=False)))
        else:
            opts.append("{}={}".format(key, value))

    if opts:
        argv += ["-o"] + opts
    return argv


#: ``kind`` -> argv 构造器。新增作业种类时只改这一处。
_ARGV_BUILDERS = {
    "train": build_train_argv,
    "eval": build_eval_argv,
    "predict": build_predict_argv,
}


def supported_job_kinds():
    """本构建实际支持的作业种类（``_ARGV_BUILDERS`` 的键）。

    刻意从**分派表**推导而不是从 ``JOB_KINDS`` 常量读：后者是「协议允许什么」，
    前者是「这份代码真的能跑什么」。加了新种类但忘了重建镜像时，两者会不一致——
    而 ``/healthz`` 报出去的正应该是后者。
    """
    return sorted(_ARGV_BUILDERS)


def build_argv(spec, config_abs_path, output_dir, python_exe, repo_root):
    """按 ``spec.kind`` 分派到对应的 argv 构造器。

    分派只在这一处——``jobs._run_job`` 不该知道有哪些种类，否则每加一种都要改
    两个文件（这类分散历史上就出过「加了新种类却忘了改分发点」的问题）。
    """
    kind = spec.validate_kind() if hasattr(spec, "validate_kind") else (spec.kind or "train")
    builder = _ARGV_BUILDERS.get(kind)
    if builder is None:  # validate_kind 已经报过更清楚的错，这里只是兜底
        raise ValueError("unsupported job kind: {!r}".format(kind))
    return builder(spec, config_abs_path, output_dir, python_exe, repo_root)


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
class JobRun(object):
    """一个正在跑的（或跑过的）作业。

    名字里没有 "train" 是因为它服务**所有** kind：训练与评估的进程管理、
    日志泵、指标契约、终态判定完全一致，差别只在 ``argv`` 怎么拼
    （见 :func:`build_argv`）。原先叫 ``TrainingRun``，加评估种类时就得跟着改
    一堆误导性的类型标注。

    ⚠️ 不管哪种 kind，**终态都依赖 ``metrics.jsonl`` 里的 ``end`` 事件**——
    「进程退出且没有 end」会被 ``classify_exit`` 判成 ``no_end_event / failed``。
    所以新增种类时，CLI 侧必须同样写契约（见 ``tools/eval.py``）。
    """

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


def classify_exit(end_event, exit_code, cancel_requested, has_metrics_file, kind="train"):
    """把 (end 事件, 退出码, 是否取消) 收敛成 (status, exit_reason, error)。

    这是整套判定规则的**唯一**实现，实时路径与重启恢复路径共用它——
    避免两条路径给出不一致的结论。

    ``kind`` 只影响**提示文案**（"训练未开始即失败" vs "评估未开始即失败"）。
    它有默认值 ``"train"``，所以旧调用点不需要改；而判定逻辑本身与 kind 无关——
    那是刻意的：终态规则对所有种类必须一致，否则「什么算失败」会有两套解释。
    """
    if cancel_requested:
        return "cancelled", "cancelled", None

    if end_event is None:
        # 没有 end：作业没真正开始（构造阶段就炸了）或被 SIGKILL
        if exit_code not in (0, None):
            # ⚠️ 文案里**不能写死某一种**——三种 kind 都会走这里。
            #   这条提示是用户看到的第一现场，出现无关的字眼会让人怀疑自己看错了
            #   对象（比如预测作业报「评估进程被强杀」）。
            what = {"train": "训练", "eval": "评估", "predict": "预测"}.get(
                (kind or "train"), "作业")
            hint = (f"{what}未开始即失败（模型构造阶段异常：配置/权重/数据集/依赖问题）"
                    if not has_metrics_file else f"{what}进程被强杀，未写出 end 事件")
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
