"""作业管理：提交（幂等）、排队、取消、查询、重启恢复。

**单一排队点原则**：GPU 是本平台最稀缺的资源，排队与并发准入**只在这里发生**。
AIStation 侧（``concurrency.py``）只做"提交节流"（防队列无限膨胀），
不要在平台侧再排一次队——两个队列 = 队头阻塞 + 状态对不齐。

重启恢复的价值：训练动辄几小时，服务重启（升级/崩溃/OOM）不该让平台侧那一排
"运行中"变成僵尸。恢复逻辑复用 ``metrics.jsonl`` 契约——**文件在、状态就在**：
  - 进程还活着 -> 继续 tail 指标文件
  - 进程没了   -> 读最后一条 end 事件按同一套规则收敛
"""
from __future__ import absolute_import

import asyncio
import json
import os
import time

from . import runner
from .runner import (JobRun, classify_exit,
                     pid_alive, read_last_event, send_signal)
from .schemas import TERMINAL, JobSpec
from .store import JobStore, row_to_dict
from .streams import Hub, LogRing


class JobManager(object):
    def __init__(self, store: JobStore, settings, hub=None, log_ring=None):
        self.store = store
        self.settings = settings
        self.hub = hub or Hub()
        self.log_ring = log_ring or LogRing(settings.log_ring)
        self.runs = {}              # job_id -> JobRun（仅本进程启动的）
        self._configs = {}          # job_id -> 解析后的配置绝对路径
        self._queue = None
        self._workers = []
        self._pending = []          # 已入队待跑的 job_id
        self._tasks = {}            # job_id -> asyncio.Task（日志泵/指标轮询）

    # ------------------------------------------------------------ 生命周期
    async def start(self):
        self.settings.ensure_dirs()
        await self.recover()
        n = max(1, int(self.settings.max_concurrent))
        self._queue = asyncio.Queue()
        self._workers = [
            asyncio.ensure_future(self._worker(i)) for i in range(n)
        ]

    async def stop(self):
        for t in self._workers:
            t.cancel()
        for t in self._tasks.values():
            t.cancel()
        self._workers = []
        self._tasks = {}

    # ------------------------------------------------------------ 提交
    async def submit(self, spec, user_id=None, tenant=None, idempotency_key=None):
        """提交作业。返回 ``(job_dict, idempotent_hit)``。

        幂等：同一 ``(tenant, user_id, Idempotency-Key)`` 重复提交返回**同一个** job_id
        且 ``idempotent_hit=True``，并且**不会重复入队**。
        这是分布式训练服务的头号事故防线——客户端超时重试若重复排队，
        就是白烧几小时卡和一份显存。
        """
        # 先解析配置：非法请求要在落库**之前**就失败，别留一条永远排不动的僵尸作业
        config_abs = self._resolve_config(spec)

        row, hit = self.store.create(
            spec, user_id=user_id, tenant=tenant, idempotency_key=idempotency_key)
        if hit:
            # 幂等命中：原样返回既有作业，不改状态、不入队
            return row_to_dict(row), True

        job_id = row["job_id"]
        output_dir = os.path.join(self.settings.data_root, "jobs", job_id)
        self.store.update(job_id, output_dir=output_dir)
        self._configs = getattr(self, "_configs", {})
        self._configs[job_id] = config_abs
        await self._enqueue(job_id)
        return row_to_dict(self.store.get(job_id)), False

    def _resolve_config(self, spec):
        """把 model_name / config_path 解析成仓库内的配置绝对路径。"""
        import sys

        repo = self.settings.repo_root
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from ptcore.config_schema import resolve_config

        if spec.config_path:
            path = spec.config_path
            return path if os.path.isabs(path) else os.path.join(repo, path)
        if not spec.model_name:
            raise ValueError("JobSpec 需要 model_name 或 config_path 其一")
        path, _how = resolve_config(spec.model_name)
        return os.path.abspath(path)

    async def _enqueue(self, job_id):
        await self._queue.put(job_id)
        self._pending.append(job_id)

    # ------------------------------------------------------------ worker
    async def _worker(self, idx):
        while True:
            job_id = await self._queue.get()
            # 一出队就从 _pending 移除：它已经开跑了，不该再占队列位置号
            # （否则排队作业的 position 会把正在跑的那个也算进去）。
            # 同时 _pending 必须在这里收缩，否则会随跑过的作业数无限增长。
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
            try:
                row = self.store.get(job_id)
                if row is None:
                    continue
                if row["status"] in TERMINAL:
                    continue
                if self.store.cancel_requested(job_id):
                    self.store.mark_finished(job_id, "cancelled", exit_reason="cancelled")
                    continue
                await self._run_job(row)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # worker 绝不能因为单个作业异常而死掉
                self.store.mark_finished(
                    job_id, "failed", exit_reason="internal_error",
                    error="{}: {}".format(type(exc).__name__, exc))
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

    async def _run_job(self, row):
        job_id = row["job_id"]
        spec = JobSpec.model_validate_json(row["spec_json"])
        output_dir = row["output_dir"] or os.path.join(
            self.settings.data_root, "jobs", job_id)
        os.makedirs(output_dir, exist_ok=True)

        config_abs = self._configs.get(job_id) or self._resolve_config(spec)
        # 按 spec.kind 分派到 train / eval 的 argv 构造器（分派只在 runner 一处）
        argv = runner.build_argv(
            spec, config_abs, output_dir,
            self.settings.python_exe, self.settings.repo_root)
        env = runner._spawn_env(os.environ, {
            "PYTHONPATH": self.settings.repo_root,
            "TKILN_METRICS": "1",
        })

        run = JobRun(job_id, spec, argv, output_dir, self.settings.repo_root,
                          self.settings.python_exe, env=env, log_ring=self.log_ring)
        self.runs[job_id] = run

        # 让调用方能从 argv 复现命令（排障必备）
        self.store.update(job_id, output_dir=output_dir, log_path=run.log_path,
                          metrics_path=run.metrics_path)

        self._publish(job_id, "log", " ".join(argv))
        self._publish(job_id, "log", "[service] cwd={}".format(self.settings.repo_root))
        try:
            run.proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.settings.repo_root,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (OSError, ValueError) as exc:
            self.store.mark_finished(
                job_id, "failed", exit_reason="spawn_failed",
                error="无法启动训练进程: {}".format(exc))
            return

        run.pid = run.proc.pid
        # ⚠️ 必须显式取出 stdout 挂到 run 上：_pump_logs 只认 run.stdout。
        #    漏掉这行 pump 会在首行 AttributeError，静默死亡 → 日志全丢。
        run.stdout = run.proc.stdout
        self.store.mark_running(job_id, run.pid, run.log_path, run.metrics_path)

        pump = asyncio.ensure_future(self._pump_logs(run))
        poller = asyncio.ensure_future(self._poll_metrics(run))
        self._tasks[job_id] = pump

        try:
            await run.proc.wait()
        finally:
            poller.cancel()
            try:
                await poller
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            # 等日志泵把管道读完再收尾：进程已退出 → stdout 到达 EOF → 循环
            # 自然结束 → 文件 close 落盘。限时兜底，防止管道异常时整体卡死。
            try:
                await asyncio.wait_for(pump, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                pump.cancel()
            # 最后再拉一次，确保 end 事件一定被消费到
            try:
                await self._drain_metrics(run)
            except Exception:  # noqa: BLE001
                pass
            run.exit_code = run.proc.returncode
            run.end_event = read_last_event(run.metrics_path, "end")
            status, reason, error = classify_exit(
                run.end_event, run.exit_code, run.cancel_requested,
                os.path.isfile(run.metrics_path), kind=spec.kind)
            self.store.mark_finished(job_id, status, exit_reason=reason,
                                     exit_code=run.exit_code, error=error)
            self._publish(job_id, "log", "[service] job {} -> {} ({})".format(
                job_id, status, reason))
            self._publish(job_id, "end", {
                "__end__": True,
                                    "job_id": job_id, "status": status, "exit_reason": reason,
                "exit_code": run.exit_code, "error": error,
                "best": (run.end_event or {}),
            })
            self._tasks.pop(job_id, None)
            self.log_ring.drop(job_id)

    # ------------------------------------------------------------ 泵
    async def _pump_logs(self, run: JobRun):
        """把子进程 stdout 逐行读到日志文件 + 内存环形缓冲 + SSE。"""
        try:
            with open(run.log_path, "a", encoding="utf-8", newline="\n") as fh:
                while True:
                    raw = await run.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    fh.write(line + "\n")
                    fh.flush()
                    self.log_ring.append(run.job_id, line)
                    self._publish(run.job_id, "log", line)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:  # noqa: BLE001
            # 绝不能静默吞掉：日志泵一死，service.log / SSE / /logs 会同时变空，
            # 而训练本身照常跑完——用户看到的只有「什么都没有」，无从排查。
            # 把异常本身作为一行日志落到环形缓冲与文件里，让故障可见。
            msg = "[service] 日志泵异常中止: {}: {}".format(type(exc).__name__, exc)
            try:
                with open(run.log_path, "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(msg + "\n")
                self.log_ring.append(run.job_id, msg)
                self._publish(run.job_id, "log", msg)
            except Exception:  # noqa: BLE001
                pass

    async def _poll_metrics(self, run: JobRun):
        """按字节偏移增量 tail metrics.jsonl 并广播。"""
        while True:
            events, run._metrics_offset = run.metrics_since(run._metrics_offset)
            for ev in events:
                seq = ev.get("seq")
                if isinstance(seq, int):
                    self.store.bump_metrics_seq(run.job_id, seq)
                self._publish(run.job_id, "metric", ev)
            await asyncio.sleep(max(0.2, float(self.settings.poll_interval)))

    async def _drain_metrics(self, run: JobRun):
        """收尾：把文件里剩下的事件（含 end）一次性消费掉。"""
        deadline = time.time() + 5.0
        while time.time() < deadline:
            events, run._metrics_offset = run.metrics_since(run._metrics_offset)
            for ev in events:
                seq = ev.get("seq")
                if isinstance(seq, int):
                    self.store.bump_metrics_seq(run.job_id, seq)
                self._publish(run.job_id, "metric", ev)
            if any(e.get("type") == "end" for e in events):
                return
            if not events:
                await asyncio.sleep(0.2)

    def _publish(self, job_id, kind, payload):
        self.hub.publish(job_id, kind, payload)

    # ------------------------------------------------------------ 取消
    async def cancel(self, job_id):
        row = self.store.get(job_id)
        if row is None:
            return None
        if row["status"] in TERMINAL:
            return {"job_id": job_id, "status": row["status"], "terminated": False}

        changed = self.store.request_cancel(job_id)
        run = self.runs.get(job_id)
        if run is not None and run.proc is not None:
            run.cancel_requested = True
            self._publish(job_id, "log", "[service] 收到取消请求，先发终止信号…")
            send_signal(run.proc, hard=False)
            try:
                await asyncio.wait_for(run.proc.wait(),
                                       timeout=float(self.settings.cancel_grace))
            except asyncio.TimeoutError:
                self._publish(job_id, "log", "[service] 宽限期到，强制杀死")
                send_signal(run.proc, hard=True)
            # _run_job 的 finally 会把状态收敛成 cancelled
            await asyncio.sleep(0.1)
        else:
            # 还没起进程（排队中）
            self.store.mark_finished(job_id, "cancelled", exit_reason="cancelled")
        final = self.store.get(job_id)
        return {"job_id": job_id, "status": final["status"], "terminated": bool(changed)}

    # ------------------------------------------------------------ 查询
    def info(self, job_id):
        d = row_to_dict(self.store.get(job_id))
        if d and d.get("status") == "queued":
            d["queue_position"] = self._queue_position(job_id)
        return d

    def list(self, status=None, user_id=None, tenant=None, limit=50, offset=0, kind=None):
        """列出作业。``kind`` 非空时只返回该种类的作业。

        ⚠️ 过滤在**取回之后**做，不下推到 SQL：``spec_json`` 是 JSON 列，按里面的
        字段过滤要么写 JSONB 表达式（依赖特定 PG 版本）要么加冗余列。作业列表的
        单页上限本来就只有几百条，内存过滤的代价可以忽略；而下推一旦出错是
        **静默少返回数据**，比慢更糟。

        ⚠️ 此时 ``total`` 已是过滤前的数字，与 ``items`` 对不上。已按过滤后的长度
        重算——宁可总数看起来偏小，也不要出现"第 2 页翻出空列表"的错位现象。
        """
        total, rows = self.store.list(status=status, user_id=user_id, tenant=tenant,
                                     limit=limit, offset=offset)
        items = []
        for r in rows:
            d = row_to_dict(r)
            if not d:
                continue
            # ⚠️ row_to_dict 把 spec_json 还原成 **JobSpec 对象**（不是 dict），
            #    所以这里必须用属性访问——写成 .get() 会 500（已踩过一次）。
            if kind and (getattr(d.get("spec"), "kind", None) or "train") != kind:
                continue
            if d.get("status") == "queued":
                d["queue_position"] = self._queue_position(d["job_id"])
            items.append(d)
        if kind:
            total = len(items)
        return {"total": total, "items": items}

    def _queue_position(self, job_id):
        try:
            return self._pending.index(job_id) + 1
        except ValueError:
            return None

    def metrics_since_seq(self, job_id, after_seq=-1, limit=5000):
        """按 seq 补发历史指标（SSE 重连/首次拉取用）。"""
        row = self.store.get(job_id)
        path = row["metrics_path"] if row else None
        if not path or not os.path.isfile(path):
            return []
        out = []
        try:
            with open(path, "rb") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        ev = json.loads(raw.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(ev.get("seq"), int) and ev["seq"] > after_seq:
                        out.append(ev)
                    if len(out) >= limit:
                        break
        except OSError:
            return out
        return out

    def log_tail(self, job_id, n=200):
        """优先读内存环形缓冲；服务重启后回落到落盘日志。"""
        lines = self.log_ring.tail(job_id, n)
        if lines:
            return lines
        row = self.store.get(job_id)
        if not row or not row["log_path"] or not os.path.isfile(row["log_path"]):
            return []
        try:
            with open(row["log_path"], "r", encoding="utf-8", errors="replace") as f:
                return [ln.rstrip("\r\n") for ln in f.readlines()[-n:]]
        except OSError:
            return []

    # ------------------------------------------------------------ 重启恢复
    async def recover(self):
        """启动时收敛所有非终态作业。

        判定完全依赖**文件契约**而非内存态：
          - 有 pid 且进程还活着 -> 继续按 offset tail 指标文件（训练没白跑）
          - 进程没了 -> 读最后一条 end 事件，用与实时路径**同一套**规则收敛
        """
        rows = self.store.active_rows()
        if not rows:
            return
        for row in rows:
            job_id = row["job_id"]
            if pid_alive(row["pid"]) and os.path.isdir(row["output_dir"] or ""):
                # 训练还活着：只接管指标流与日志观察，不重复起进程
                run = JobRun(
                    job_id, JobSpec.model_validate_json(row["spec_json"]), [],
                    row["output_dir"], self.settings.repo_root,
                    self.settings.python_exe, log_ring=self.log_ring)
                run.pid = row["pid"]
                run._metrics_offset = 0
                self.runs[job_id] = run
                poller = asyncio.ensure_future(self._poll_metrics(run))
                self._tasks[job_id] = poller
                asyncio.ensure_future(self._watch_recovered(run))
                self._publish(job_id, "log",
                              "[service] 服务重启，已接管进行中的作业 {}".format(job_id))
            else:
                end = read_last_event(row["metrics_path"], "end")
                has_file = os.path.isfile(row["metrics_path"] or "")
                status, reason, error = classify_exit(
                    end, None, bool(row["cancel_requested"]), has_file,
                    kind=JobSpec.model_validate_json(row["spec_json"]).kind)
                if end is None and not has_file and row["status"] == "queued":
                    status, reason, error = "failed", "lost_on_restart", \
                        "服务在作业启动前重启，作业未真正开始"
                self.store.mark_finished(job_id, status, exit_reason=reason,
                                         error=error)
                self._publish(job_id, "end", {
                    "__end__": True,
                                        "job_id": job_id, "status": status, "exit_reason": reason,
                    "error": error, "recovered": True,
                })

    async def _watch_recovered(self, run: JobRun):
        """接管模式下：轮询到 end 事件就收敛作业状态。"""
        while True:
            await asyncio.sleep(max(0.5, float(self.settings.poll_interval)))
            events, run._metrics_offset = run.metrics_since(run._metrics_offset)
            for ev in events:
                seq = ev.get("seq")
                if isinstance(seq, int):
                    self.store.bump_metrics_seq(run.job_id, seq)
                self._publish(run.job_id, "metric", ev)
                if ev.get("type") == "end":
                    status, reason, error = classify_exit(
                        ev, None, bool(self.store.cancel_requested(run.job_id)),
                        True, kind=run.spec.kind)
                    self.store.mark_finished(run.job_id, status, exit_reason=reason,
                                             error=error)
                    self._publish(run.job_id, "end", {
                        "__end__": True,
                                            "job_id": run.job_id, "status": status,
                        "exit_reason": reason, "error": error, "recovered": True,
                    })
                    self._tasks.pop(run.job_id, None)
                    return
