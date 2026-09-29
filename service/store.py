"""作业状态存储（SQLite）。

为什么不用内存 dict：
  - 训练动辄几小时，服务重启（升级/崩溃/OOM）**不能丢作业记录**，否则平台侧
    那一排"运行中"永远是僵尸；
  - 幂等键必须跨重启生效（否则重试会重复烧卡）；
  - 指标消费游标（``metrics_seq``）要持久化，否则重启后要么丢历史、要么重复推。

⚠️ SQLite 只是**作业元信息**的落地，**指标本身不入库**——指标真值永远是
``<output_dir>/metrics.jsonl``，谁需要谁 tail。这样避免了"服务库里一份、
文件里一份"的双写口径不一致。
"""
from __future__ import absolute_import

import json
import os
import sqlite3
import threading
import time
import uuid

from .schemas import JobSpec, TERMINAL

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    status           TEXT NOT NULL,
    spec_json        TEXT NOT NULL,
    user_id          TEXT,
    tenant           TEXT,
    labels_json      TEXT NOT NULL DEFAULT '{}',
    idempotency_key  TEXT,
    output_dir       TEXT,
    log_path         TEXT,
    metrics_path     TEXT,
    pid              INTEGER,
    exit_reason      TEXT,
    exit_code        INTEGER,
    error            TEXT,
    metrics_seq      INTEGER NOT NULL DEFAULT -1,
    created_ts       REAL NOT NULL,
    started_ts       REAL,
    finished_ts      REAL,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_ts);
-- 同一 (tenant,user) 下幂等键唯一；未带幂等键的作业传 NULL，不受约束
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idem
    ON jobs(tenant, user_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
"""

_COLUMNS = (
    "job_id, status, spec_json, user_id, tenant, labels_json, idempotency_key, "
    "output_dir, log_path, metrics_path, pid, exit_reason, exit_code, error, "
    "metrics_seq, created_ts, started_ts, finished_ts, cancel_requested"
)


def new_job_id():
    return "job_{}_{}".format(time.strftime("%Y%m%d%H%M%S"), uuid.uuid4().hex[:8])


class JobStore(object):
    """线程安全的轻量作业库。所有写操作都在一把锁内 + 单连接。"""

    def __init__(self, db_path):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # WAL：读写不互斥，SSE 高频读 + 状态写不会互相阻塞
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self._conn.commit()

    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # ------------------------------------------------------------ 写
    def create(self, spec: JobSpec, user_id=None, tenant=None, idempotency_key=None,
               output_dir=None):
        """新建作业；若幂等键命中已有作业，返回 ``(row, True)``。"""
        job_id = new_job_id()
        with self._lock:
            if idempotency_key:
                row = self._find_by_idem(tenant, user_id, idempotency_key)
                if row is not None:
                    return row, True
            self._conn.execute(
                "INSERT INTO jobs (job_id, status, spec_json, user_id, tenant, "
                "labels_json, idempotency_key, output_dir, created_ts) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (job_id, "queued", spec.model_dump_json(), user_id, tenant,
                 json.dumps(spec.labels or {}, ensure_ascii=False), idempotency_key,
                 output_dir, time.time()),
            )
            self._conn.commit()
        return self.get(job_id), False

    def update(self, job_id, **fields):
        if not fields:
            return
        cols = ", ".join("{} = ?".format(k) for k in fields)
        args = list(fields.values()) + [job_id]
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET {} WHERE job_id = ?".format(cols), args)
            self._conn.commit()

    def mark_running(self, job_id, pid, log_path, metrics_path, started_ts=None):
        self.update(job_id, status="running", pid=pid, log_path=log_path,
                    metrics_path=metrics_path,
                    started_ts=started_ts if started_ts is not None else time.time())

    def mark_finished(self, job_id, status, exit_reason=None, exit_code=None,
                      error=None, finished_ts=None):
        self.update(
            job_id, status=status, exit_reason=exit_reason, exit_code=exit_code,
            error=error,
            finished_ts=finished_ts if finished_ts is not None else time.time(),
        )

    def request_cancel(self, job_id):
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE job_id = ? "
                "AND status IN ('queued','running')", (job_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def cancel_requested(self, job_id):
        row = self.get(job_id)
        return bool(row and row["cancel_requested"])

    # ------------------------------------------------------------ 读
    def get(self, job_id):
        with self._lock:
            cur = self._conn.execute(
                "SELECT {} FROM jobs WHERE job_id = ?".format(_COLUMNS),
                (job_id,))
            return cur.fetchone()

    def _find_by_idem(self, tenant, user_id, key):
        cur = self._conn.execute(
            "SELECT {} FROM jobs WHERE tenant IS ? AND user_id IS ? "
            "AND idempotency_key = ? ORDER BY created_ts DESC LIMIT 1".format(_COLUMNS),
            (tenant, user_id, key))
        return cur.fetchone()

    def list(self, status=None, user_id=None, tenant=None, limit=50, offset=0):
        where, args = [], []
        if status:
            where.append("status = ?")
            args.append(status)
        if user_id:
            where.append("user_id = ?")
            args.append(user_id)
        if tenant:
            where.append("tenant = ?")
            args.append(tenant)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM jobs " + clause, args).fetchone()[0]
            cur = self._conn.execute(
                "SELECT {} FROM jobs {} ORDER BY created_ts DESC LIMIT ? OFFSET ?".format(
                    _COLUMNS, clause), args + [limit, offset])
            items = cur.fetchall()
        return total, items

    def active_rows(self):
        """重启恢复用：所有未到终态的作业。"""
        marks = ",".join("?" * len(TERMINAL))
        with self._lock:
            cur = self._conn.execute(
                "SELECT {} FROM jobs WHERE status NOT IN ({}) "
                "ORDER BY created_ts ASC".format(_COLUMNS, marks), list(TERMINAL))
            return cur.fetchall()

    def bump_metrics_seq(self, job_id, seq):
        """只前进不后退（多 poller 竞争时取最大值）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET metrics_seq = ? WHERE job_id = ? AND metrics_seq < ?",
                (seq, job_id, seq))
            self._conn.commit()

    def orphaned(self, job_id, error):
        """进程已不在且拿不到 end 事件 -> 按失败收敛（见 runner 的判定规则）。"""
        self.mark_finished(job_id, "failed", exit_reason="orphaned",
                           error=error)


#: ``JobInfo`` 认识的字段（显式过滤，避免把 pid/idempotency_key 之类
#: 直接喂给 Pydantic 而依赖其 extra='ignore' 默认行为——那太隐式了）
_INFO_FIELDS = (
    "job_id", "status", "spec", "exit_reason", "exit_code", "created_ts",
    "started_ts", "finished_ts", "duration_sec", "output_dir", "log_path",
    "metrics_path", "metrics_seq", "user_id", "tenant", "labels", "error",
    "queue_position",
)


def row_to_dict(row):
    """``sqlite3.Row`` -> ``JobInfo`` 可用的 dict（spec 还原为 ``JobSpec``）。"""
    if row is None:
        return None
    d = {k: row[k] for k in row.keys()}
    d["spec"] = JobSpec.model_validate_json(d.pop("spec_json"))
    try:
        d["labels"] = json.loads(d.pop("labels_json") or "{}")
    except (TypeError, ValueError):
        d["labels"] = {}
    d["duration_sec"] = (
        round(d["finished_ts"] - d["started_ts"], 2)
        if d.get("started_ts") and d.get("finished_ts") else None
    )
    d.setdefault("queue_position", None)
    return {k: d.get(k) for k in _INFO_FIELDS}
