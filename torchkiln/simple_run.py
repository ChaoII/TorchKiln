"""单作业进程入口（简单模式）。

AIStation 侧以如下方式启动::

    python -m torchkiln.simple_run <spec.json> --callback-url <url> \
        --token <t> [--task-id <int>] [--repo-root <root>]

本进程读平台侧写好的 :class:`service.schemas.JobSpec`，按 ``kind`` 分派：

  ``train``    进程内 ``build_trainer(..., event_sink=...).train()``
  ``eval``     进程内 ``trainer.evaluate()``（``epoch_num=0`` / ``use_ema=False``，
               对齐 ``tools/eval.py`` 语义），eval 指标由本层补发
  ``predict``  进程内 :class:`torchkiln.infer_api.Runtime` 逐张推理，
               发 ``predict`` 进度事件（与 ``tools/infer/predict_yolo.py`` 同字段）
  ``export``   子进程 ``python tools/export.py``（短任务，stdout 透传）

事件回调（HTTP loopback）
--------------------------
事件通过后台 daemon 线程 POST 到 ``--callback-url``（header ``X-Torchkiln-Token``，
body = ``{"task_id": task_id, "seq": ..., "ts": ..., "type": ..., **fields}``），
与 ``metrics.jsonl`` 契约同形。发送失败重试 3 次（1s/2s/4s 退避），仍失败仅
``logging.warning`` 后丢弃——**事件可丢，训练循环绝不被 HTTP 阻塞**。

终态约定（与 metrics.jsonl 一致）
----------------------------------
  - 终态一律是 ``end`` 事件（``exit_reason``：finished / early_stop /
    already_complete / error）。train 路径由 trainer 的 finally 自动 emit；
    eval / predict / export 路径由本层补发；
  - ``failed`` 只用于**无法构造 end** 的启动即失败（spec 文件不存在 / JSON
    非法 / kind 未知——都发生在作业进程真正可用之前）；spec 已解析后
    任何 kind 的失败（含配置解析 / trainer 构造失败）一律发 ``end(error)``；
  - **不发双终态事件**：``terminal_emitted`` 标记保证失败兜底只在尚未发出
    终态时才补。

退出码：成功 0；任何失败 1（与 AIStation 侧 returncode 判定一致）。

实现形态：**同步**——训练循环是同步的，HTTP 发送由后台 daemon 线程承载，
本模块不引入 asyncio。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2

from ptcore.config import parse_args_to_config
from ptcore.config_schema import resolve_config
from ptcore.factory import build_trainer
from torchkiln.infer_api import Runtime

__all__ = ["HttpEventSink", "build_http_events", "SimpleRun", "main"]

#: 事件发送失败后的重试退避（秒）。共 3 次重试（1 次首发 + 3 次重试 = 4 次尝试）。
RETRY_DELAYS = (1.0, 2.0, 4.0)

#: 目录批量输入时认的图片后缀（与 ``tools/infer/predict_yolo.py`` 一致）
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

_SENTINEL = object()


# ---------------------------------------------------------------- 事件回调
class HttpEventSink:
    """把事件经 HTTP POST 推给平台侧（body 契约见模块 docstring）。

    线程模型：``emit()`` 只做同步入队（无界 ``queue.Queue`` + ``put_nowait``，
    永不阻塞/上抛，训练循环零开销）；后台 daemon 线程串行 POST，失败按
    :data:`RETRY_DELAYS` 退避重试，仍失败仅 warning 后丢弃。
    """

    def __init__(self, callback_url, token, task_id=None):
        self.callback_url = callback_url
        self.token = token
        self.task_id = task_id
        self._q = queue.Queue()
        self._seq = 0
        self._lock = threading.Lock()
        #: 是否已发出终态事件（end/failed）；供调用方做「不补双终态」判定
        self.terminal_emitted = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="torchkiln-event-sink")
        self._thread.start()

    # -- EventSink 协议 --------------------------------------------------
    def emit(self, type, fields):  # noqa: A002 - 与 EventSink.emit 契约一致
        """入队一条事件。

        ``fields`` 两种形态都接受：
          - 普通调用（simple_run 自补的 end/predict 等）：纯载荷 dict，
            seq/ts 由本 sink 自增注入；
          - :class:`ptcore.events.CallbackList` 广播：payload 里已带
            seq/ts/type，原样透传（不重复注入）。
        值为 ``None`` 的字段丢弃（与 ``Event.to_dict()`` 一致）。
        """
        fields = dict(fields or {})
        if "seq" not in fields or "ts" not in fields or "type" not in fields:
            with self._lock:
                self._seq += 1
                seq, ts = self._seq, time.time()
            fields = {"seq": seq, "ts": ts, "type": type,
                      **{k: v for k, v in fields.items()
                         if k not in ("seq", "ts", "type")}}
        body = {k: v for k, v in fields.items() if v is not None}
        if body.get("type") in ("end", "failed"):
            self.terminal_emitted = True
        try:
            self._q.put_nowait({"task_id": self.task_id, **body})
        except Exception:  # noqa: BLE001 - emit 契约：绝不向调用方上抛
            logging.warning("事件入队失败已丢弃: %s", body.get("type"),
                            exc_info=True)

    def close(self, timeout=15):
        """放哨兵并 join 工作线程（收尾调用；幂等）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._q.put_nowait(_SENTINEL)
        except Exception:  # noqa: BLE001
            pass
        if self._thread is not threading.current_thread():
            self._thread.join(timeout)

    # -- 后台发送 ----------------------------------------------------------
    def _run(self):
        while True:
            body = self._q.get()
            if body is _SENTINEL:
                return
            self._post_with_retry(body)

    def _post_with_retry(self, body):
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        last = None
        for delay in (0.0,) + RETRY_DELAYS:
            try:
                req = urllib.request.Request(
                    self.callback_url, data=data, method="POST",
                    headers={"Content-Type": "application/json",
                             "X-Torchkiln-Token": self.token})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                if delay:
                    time.sleep(delay)
        logging.warning("事件回调失败，重试 3 次后丢弃: url=%s type=%s err=%s",
                        self.callback_url, body.get("type"), last)


def build_http_events(callback_url, token, task_id=None):
    """工厂：返回 :class:`HttpEventSink`（供 ``build_trainer(..., event_sink=)``）。"""
    return HttpEventSink(callback_url, token, task_id)


# ---------------------------------------------------------------- 参数工具
def _opt_str(value):
    """spec.params 的值 -> ``-o`` 的字符串形态（与 service.runner 一致）。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return str(value)


def _build_opts(spec):
    """spec.params 点分键 -> ``["k=v", ...]``；seed 追加 ``Global.seed``。"""
    opts = []
    for key, value in (spec.params or {}).items():
        s = _opt_str(value)
        if s is not None:
            opts.append("{}={}".format(key, s))
    if getattr(spec, "seed", None) is not None:
        opts.append("Global.seed={}".format(spec.seed))
    return opts


def _train_dataset_opts(spec):
    ds = spec.dataset
    if ds is None:
        return []
    opts = []
    if ds.data_dir:
        opts.append("Train.dataset.data_dir={}".format(ds.data_dir))
    if ds.train_list:
        opts.append("Train.dataset.label_file_list={}".format(ds.train_list))
    if ds.data_dir:
        opts.append("Eval.dataset.data_dir={}".format(ds.data_dir))
    if ds.val_list:
        opts.append("Eval.dataset.label_file_list={}".format(ds.val_list))
    return opts


def _eval_dataset_opts(spec):
    """eval 作业的数据集注入（对齐 ``service.runner.build_eval_argv``）。

    trainer 构造期会无条件打开 Train + Eval 两份清单，只给 Eval 的话训练集
    会退回配置默认路径导致 setup 失败；epoch_num=0 时训练集不会被迭代，
    指向 val 那份清单即可。
    """
    ds = spec.dataset
    if ds is None:
        return []
    opts = []
    if ds.data_dir:
        opts.append("Eval.dataset.data_dir={}".format(ds.data_dir))
    if ds.val_list:
        opts.append("Eval.dataset.label_file_list={}".format(ds.val_list))
        if ds.data_dir:
            opts.append("Train.dataset.data_dir={}".format(ds.data_dir))
        opts.append("Train.dataset.label_file_list={}".format(ds.val_list))
    return opts


def _list_images(input_path):
    """待推理图片列表：单文件 -> ``[(路径, None)]``；目录 -> 递归 + 相对路径。

    语义与 ``tools/infer/predict_yolo._list_images`` 一致（递归、按相对路径
    排序、相对路径用 ``/``）。
    """
    if not os.path.isdir(input_path):
        return [(input_path, None)]
    items = []
    for root, dirs, files in os.walk(input_path):
        dirs.sort()
        for name in sorted(files):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            abspath = os.path.join(root, name)
            rel = os.path.relpath(abspath, input_path).replace(os.sep, "/")
            items.append((abspath, rel))
    items.sort(key=lambda it: it[1])
    return items


# ---------------------------------------------------------------- 作业分派
class SimpleRun:
    """一次单作业进程的生命周期：读 JobSpec -> 分派 kind -> 推事件 -> 退出码。"""

    #: simple_run 比 service 的 JOB_KINDS 多一个 export（不改 service 侧协议）
    KINDS = ("train", "eval", "predict", "export")

    def __init__(self, spec, callback_url, token, repo_root, task_id=None):
        self.spec = spec
        self.callback_url = callback_url
        self.token = token
        self.repo_root = str(repo_root)
        self.task_id = task_id

    def run(self):
        """同步执行整个作业，返回退出码（0 成功 / 1 失败）。"""
        sink = build_http_events(self.callback_url, self.token, self.task_id)
        try:
            kind = (self.spec.kind or "train").strip().lower()
            if kind not in self.KINDS:
                self._emit_failed(
                    sink, "unsupported job kind: {!r}（可选：{}）".format(
                        self.spec.kind, ", ".join(self.KINDS)))
                return 1
            try:
                rc = self._dispatch(kind, sink)
            except Exception as exc:  # noqa: BLE001
                rc = 1
                # 只有终态尚未发出时才补 failed（trainer 的 finally 可能已经
                # 发过 end(error)）——绝不发双终态事件。
                if not getattr(sink, "terminal_emitted", False):
                    self._emit_failed(
                        sink, "{}: {}".format(type(exc).__name__, exc))
            return rc
        finally:
            sink.close()

    # -- 内部 --------------------------------------------------------------
    @staticmethod
    def _emit_failed(sink, msg):
        sink.emit("failed", {"exit_reason": "setup_failed", "error": msg})

    def _resolve_config_path(self):
        name_or_path = self.spec.config_path or self.spec.model_name
        if not name_or_path:
            raise ValueError("spec 缺少 config_path / model_name，无法解析配置")
        path, _how = resolve_config(
            name_or_path, config_root=os.path.join(self.repo_root, "configs"))
        return path

    def _load_config(self, opts):
        return parse_args_to_config(self._resolve_config_path(), opts)

    def _dispatch(self, kind, sink):
        if kind == "train":
            return self._run_train(sink)
        if kind == "eval":
            return self._run_eval(sink)
        if kind == "predict":
            return self._run_predict(sink)
        return self._run_export(sink)

    def _run_train(self, sink):
        try:
            opts = _train_dataset_opts(self.spec) + _build_opts(self.spec)
            config = self._load_config(opts)
            trainer = build_trainer(config, event_sink=sink)
            try:
                trainer.train()
            except BaseException:  # noqa: BLE001 - end(error) 已由 trainer finally 发出
                return 1
            return 0
        except Exception as exc:  # noqa: BLE001
            # 配置解析 / trainer 构造失败：trainer 的 finally 不会执行，
            # 但 end 完全可以由本层构造（与其余 kind 一致）→ end(error)，不发 failed
            if not getattr(sink, "terminal_emitted", False):
                sink.emit("end", {"exit_reason": "error",
                                  "error": "{}: {}".format(
                                      type(exc).__name__, exc)})
            return 1

    def _run_eval(self, sink):
        try:
            return self._eval(sink)
        except Exception as exc:  # noqa: BLE001 - 失败也补 end（能构造）
            if not getattr(sink, "terminal_emitted", False):
                sink.emit("end", {"exit_reason": "error",
                                  "error": "{}: {}".format(
                                      type(exc).__name__, exc)})
            return 1

    def _eval(self, sink):
        opts = _eval_dataset_opts(self.spec) + _build_opts(self.spec)
        config = self._load_config(opts)
        g = config.setdefault("Global", {})
        if self.spec.weights_path:
            g["pretrained_model"] = self.spec.weights_path
        # 对齐 tools/eval.py：只评估不训练
        g["epoch_num"] = 0
        g["use_ema"] = False
        g.setdefault("save_model_dir", "output/_simple_eval")
        trainer = build_trainer(config, dump_config=False, event_sink=sink)
        metrics = trainer.evaluate()
        if not metrics:
            # evaluate() 不发任何事件，终态由本层补
            sink.emit("end", {"exit_reason": "error",
                              "error": "未配置 Eval 数据集，无可评估内容"})
            return 1
        main_key = getattr(trainer, "main_indicator", "hmean")
        value = float(metrics.get(main_key, 0.0) or 0.0)
        fps = float(metrics.get("fps", 0.0) or 0.0)
        # 与 base.evaluate_and_save 的 eval 事件同字段
        sink.emit("eval", {
            "main_indicator": main_key,
            "main_value": value,
            "main_indicator_mode": getattr(trainer, "_indicator_mode", "max"),
            "is_best": False,
            "fps": fps,
            "metrics": {k: v for k, v in metrics.items() if k != "fps"},
        })
        sink.emit("end", {"exit_reason": "finished",
                          "main_indicator": main_key, "main_value": value})
        return 0

    def _run_predict(self, sink):
        try:
            return self._predict(sink)
        except Exception as exc:  # noqa: BLE001 - 失败也补 end（能构造）
            if not getattr(sink, "terminal_emitted", False):
                sink.emit("end", {"exit_reason": "error",
                                  "error": "{}: {}".format(type(exc).__name__, exc)})
            return 1

    def _predict(self, sink):
        if not self.spec.weights_path:
            raise ValueError("predict 作业需要 spec.weights_path")
        if not self.spec.input_dir:
            raise ValueError("predict 作业需要 spec.input_dir")
        opts = _build_opts(self.spec)
        device = "cuda:0" if int(self.spec.resources.gpu or 0) > 0 else "cpu"
        # 模型只加载一次（在循环外）；先按 --repo-root 解析出绝对路径再传给
        # Runtime.load（它的 resolve_config 用 ptcore 内置 CONFIG_ROOT，
        # wheel 部署下可能指错目录）
        runtime = Runtime.load(
            self._resolve_config_path(),
            self.spec.weights_path, device=device, overrides=opts)
        items = _list_images(self.spec.input_dir)
        if not items:
            raise FileNotFoundError(
                "No readable image under: {} (looked for {})".format(
                    self.spec.input_dir, ", ".join(IMAGE_EXTS)))
        out_dir = ((self.spec.params or {}).get("Global.output")
                   or os.path.join("output", "predict_results"))
        started = time.time()
        total, done = len(items), 0
        for img_path, rel in items:
            img0 = cv2.imread(img_path)
            if img0 is None:
                logging.warning("跳过不可读图片: %s", img_path)
                continue
            result = runtime.infer(img0)
            for line in (getattr(result, "notes", None) or []):
                logging.info(line)
            vis = runtime.visualize(img0, result)
            # 保持输入相对路径（扩展名统一 .jpg），避免 train/val 同名图互相覆盖
            stem = os.path.splitext(rel or os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, stem + ".jpg")
            out_parent = os.path.dirname(os.path.abspath(out_path))
            if out_parent:
                os.makedirs(out_parent, exist_ok=True)
            cv2.imwrite(out_path, vis)
            done += 1
            # 进度事件，与 predict_yolo 的 sink.predict 同字段
            sink.emit("predict", {"images_total": total, "images_done": done})
        if done == 0:
            raise RuntimeError(
                "All {} input image(s) were unreadable".format(total))
        elapsed = round(time.time() - started, 3)
        sink.emit("end", {"exit_reason": "finished",
                          "images_total": total, "images_done": done,
                          "elapsed_sec": elapsed, "output_dir": out_dir})
        return 0

    def _run_export(self, sink):
        try:
            return self._export(sink)
        except Exception as exc:  # noqa: BLE001 - 失败也补 end（能构造）
            if not getattr(sink, "terminal_emitted", False):
                sink.emit("end", {"exit_reason": "error",
                                  "error": "{}: {}".format(
                                      type(exc).__name__, exc)})
            return 1

    def _export(self, sink):
        if not self.spec.weights_path:
            raise ValueError("export 作业需要 spec.weights_path")
        config_path = self._resolve_config_path()
        out_dir = ((self.spec.params or {}).get("Global.output")
                   or os.path.join("output", "export"))
        cmd = [sys.executable, "tools/export.py", "-c", config_path,
               "--weights", str(self.spec.weights_path),
               "--save-dir", str(out_dir)]
        opts = _build_opts(self.spec)
        if opts:
            cmd += ["-o"] + opts
        # stdout/stderr 默认继承（透传）
        proc = subprocess.run(cmd, cwd=self.repo_root)
        if proc.returncode == 0:
            sink.emit("end", {"exit_reason": "finished", "output_dir": out_dir})
            return 0
        sink.emit("end", {"exit_reason": "error",
                          "error": "export exited with code {}".format(
                              proc.returncode)})
        return 1


# ---------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="TorchKiln simple-run 单作业进程入口（简单模式）")
    parser.add_argument("spec", help="JobSpec JSON 文件路径")
    parser.add_argument("--callback-url", required=True,
                        help="事件回调地址（HTTP loopback）")
    parser.add_argument("--token", required=True,
                        help="回调鉴权（X-Torchkiln-Token 头）")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--repo-root", default=None,
                        help="仓库根（缺省取本文件上一级目录）；"
                             "用于 sys.path 注入与配置解析/子进程 cwd")
    args = parser.parse_args(argv)

    repo_root = (Path(args.repo_root).resolve() if args.repo_root
                 else Path(__file__).resolve().parents[1])
    repo_root_s = str(repo_root)
    if repo_root_s not in sys.path:
        sys.path.insert(0, repo_root_s)

    # 启动即失败（spec 不存在/JSON 非法/字段错误）时也要发终态事件：
    # 此时 trainer 不存在、构造不出 end，按约定用 failed。
    sink = build_http_events(args.callback_url, args.token, args.task_id)
    try:
        try:
            from service.schemas import JobSpec   # service 包不在 wheel 里，延迟导入

            raw = json.loads(Path(args.spec).read_text(encoding="utf-8"))
            spec = JobSpec(**raw)
        except Exception as exc:  # noqa: BLE001
            sink.emit("failed", {"exit_reason": "setup_failed",
                                 "error": "{}: {}".format(
                                     type(exc).__name__, exc)})
            return 1
    finally:
        sink.close()

    return SimpleRun(spec, args.callback_url, args.token, repo_root_s,
                     args.task_id).run()


if __name__ == "__main__":
    sys.exit(main())
