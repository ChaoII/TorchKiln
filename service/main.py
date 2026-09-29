"""TorchKiln 训练服务（FastAPI）。

对外三类接口（契约见 service/README.md）：
  1. 模型自描述   GET  /api/v1/models · /models/{name} · /models/{name}/schema
  2. 训练作业     POST /api/v1/train/jobs  · GET /train/jobs · /train/jobs/{id}
                  POST /train/jobs/{id}/cancel
  3. 日志与指标   GET  /train/jobs/{id}/logs        · /logs/stream   (SSE)
                  GET  /train/jobs/{id}/metrics     · /metrics/stream(SSE)

设计约束（与 AIStation 的分工，见模块 docstring 与 README）：
  - **指标真值在文件**（``<output_dir>/metrics.jsonl``），本服务只 tail 转发，
    不做二次存储——避免双写口径不一致。
  - **单一排队点**：GPU 排队与并发准入只在服务端发生。
  - **幂等**：``Idempotency-Key`` 命中则返回同一作业，绝不重复排队烧卡。
  - 状态判定统一走 ``runner.classify_exit``，实时路径与重启恢复路径结论一致。
"""
from __future__ import absolute_import

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import registry
from .jobs import JobManager
from .runner import sse_log_frame, sse_metric_frame
from .schemas import (STATUSES, CancelResult, JobCreated, JobInfo, JobList,
                      JobSpec)
from .settings import Settings
from .store import JobStore
from .streams import Hub, LogRing, sse_comment


def create_app(settings=None):
    settings = settings or Settings()
    app = FastAPI(
        title="TorchKiln Training Service",
        version="0.1.0",
        description=__doc__,
    )
    app.state.settings = settings
    app.state.hub = Hub()
    app.state.log_ring = LogRing(settings.log_ring)
    app.state.store = JobStore(settings.db_path)
    app.state.jobs = JobManager(
        app.state.store, settings, hub=app.state.hub, log_ring=app.state.log_ring)

    # ------------------------------------------------------------ 鉴权
    async def require_auth(authorization: Optional[str] = Header(default=None)):
        """服务间认证：内网隔离 + Bearer token。

        ``TKILN_SERVICE_TOKEN`` 未设置时不校验（仅适合本地/单机开发）。
        """
        if not settings.service_token:
            return
        want = "Bearer {}".format(settings.service_token)
        if (authorization or "").strip() != want:
            raise HTTPException(status_code=401, detail="invalid service token")

    # ------------------------------------------------------------ 生命周期
    @app.on_event("startup")
    async def _startup():
        settings.ensure_dirs()
        await app.state.jobs.start()

    @app.on_event("shutdown")
    async def _shutdown():
        await app.state.jobs.stop()
        app.state.store.close()

    # ------------------------------------------------------------ 基础
    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "framework": "torchkiln",
                "max_concurrent": settings.max_concurrent}

    @app.get("/api/v1/info", dependencies=[Depends(require_auth)])
    async def info():
        return registry.framework_info(settings.repo_root)

    # ------------------------------------------------------------ 1. 模型自描述
    @app.get("/api/v1/models", dependencies=[Depends(require_auth)])
    async def api_models(
        task: Optional[str] = Query(None),
        family: Optional[str] = Query(None, alias="model_family"),
        name: Optional[str] = Query(None),
        subdir: Optional[str] = Query(None),
    ):
        return {
            "items": registry.list_models(
                settings.repo_root, task=task, family=family, name=name,
                subdir=subdir),
        }

    @app.get("/api/v1/models/{model_name}", dependencies=[Depends(require_auth)])
    async def api_model(model_name: str):
        try:
            return registry.get_summary(settings.repo_root, model_name)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/v1/models/{model_name}/schema",
             dependencies=[Depends(require_auth)])
    async def api_model_schema(model_name: str, o: Optional[str] = Query(None)):
        """超参 schema。前端据此**动态生成表单**（类型/范围/控件/分组/中文字段名）。"""
        overrides = None
        if o:
            overrides = [s for s in o.replace(",", " ").split() if s]
        try:
            return registry.describe_model(
                settings.repo_root, model_name, overrides=overrides)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------ 2. 训练作业
    @app.post("/api/v1/train/jobs", status_code=202,
              response_model=JobCreated, dependencies=[Depends(require_auth)])
    async def submit_job(
        spec: JobSpec,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
        x_tenant: Optional[str] = Header(default=None, alias="X-Tenant"),
    ):
        if spec.framework and spec.framework != "torchkiln":
            raise HTTPException(status_code=400,
                                detail="unsupported framework: {}".format(spec.framework))
        try:
            job, hit = await app.state.jobs.submit(
                spec, user_id=x_user_id, tenant=x_tenant,
                idempotency_key=idempotency_key)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JobCreated(job_id=job["job_id"], status=job["status"],
                          idempotent_hit=hit, created_ts=job["created_ts"])

    @app.get("/api/v1/train/jobs", response_model=JobList,
             dependencies=[Depends(require_auth)])
    async def list_jobs(
        status: Optional[str] = Query(None),
        user_id: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        if status and status not in STATUSES:
            raise HTTPException(status_code=400,
                                detail="bad status: {}".format(status))
        data = app.state.jobs.list(status=status, user_id=user_id,
                                   limit=limit, offset=offset)
        return JobList(**data)

    def _must_get(job_id):
        job = app.state.jobs.info(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found: {}".format(job_id))
        return job

    @app.get("/api/v1/train/jobs/{job_id}", response_model=JobInfo,
             dependencies=[Depends(require_auth)])
    async def get_job(job_id: str):
        return JobInfo(**_must_get(job_id))

    @app.post("/api/v1/train/jobs/{job_id}/cancel", response_model=CancelResult,
              dependencies=[Depends(require_auth)])
    async def cancel_job(job_id: str):
        _must_get(job_id)
        res = await app.state.jobs.cancel(job_id)
        return CancelResult(**res)

    # ------------------------------------------------------------ 3b. 产物
    @app.get("/api/v1/train/jobs/{job_id}/artifacts",
             dependencies=[Depends(require_auth)])
    async def list_artifacts(job_id: str):
        """列出作业产物（权重/配置/指标等），供调用方取回。"""
        job = _must_get(job_id)
        out_dir = job.get("output_dir")
        items = []
        if out_dir and os.path.isdir(out_dir):
            for name in sorted(os.listdir(out_dir)):
                p = os.path.join(out_dir, name)
                if os.path.isfile(p):
                    items.append({"name": name, "size": os.path.getsize(p)})
        return {"job_id": job_id, "output_dir": out_dir, "items": items}

    @app.get("/api/v1/train/jobs/{job_id}/artifacts/{filename}",
             dependencies=[Depends(require_auth)])
    async def download_artifact(job_id: str, filename: str):
        """下载单个产物文件（流式）。

        ⚠️ **必须做路径逃逸校验**：``filename`` 来自 URL，直接拼进 output_dir
        会被 ``../../etc/passwd`` 之类的输入穿越到任意路径。这里只允许**文件名**
        （不含分隔符），且解析后必须仍落在作业目录内。
        """
        job = _must_get(job_id)
        out_dir = job.get("output_dir")
        if not out_dir:
            raise HTTPException(status_code=404, detail="job has no output_dir")
        # 只接受纯文件名，拒绝任何目录分隔符与 ..（防路径穿越）
        if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
            raise HTTPException(status_code=400, detail="invalid filename")
        base = os.path.abspath(out_dir)
        path = os.path.abspath(os.path.join(base, filename))
        # 双保险：解析后必须仍在作业目录内
        if os.path.commonpath([base, path]) != base:
            raise HTTPException(status_code=400, detail="path traversal rejected")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="artifact not found: " + filename)

        def _iter():
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 256)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            _iter(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="{}"'.format(filename)},
        )

    # ------------------------------------------------------------ 3. 日志 / 指标
    @app.get("/api/v1/train/jobs/{job_id}/logs", dependencies=[Depends(require_auth)])
    async def get_logs(job_id: str, tail: int = Query(200, ge=0, le=10000)):
        """补发历史日志（断线重连/首次拉取用）。"""
        _must_get(job_id)
        return {"job_id": job_id, "lines": app.state.jobs.log_tail(job_id, tail)}

    @app.get("/api/v1/train/jobs/{job_id}/metrics", dependencies=[Depends(require_auth)])
    async def get_metrics(
        job_id: str,
        offset: int = Query(-1, ge=-1, description="只返回 seq > offset 的事件"),
        limit: int = Query(5000, ge=1, le=50000),
    ):
        """按 seq 补发历史指标。``offset=-1`` 表示从头。"""
        job = _must_get(job_id)
        events = app.state.jobs.metrics_since_seq(job_id, offset, limit)
        return {
            "job_id": job_id,
            "status": job["status"],
            "metrics_seq": job.get("metrics_seq", -1),
            "items": events,
        }

    # ---- SSE ----
    def _sse_response(agen):
        """把 async generator 包成 SSE 响应。

        这几个头是 SSE 能穿透生产环境的关键：
          - ``no-transform`` / ``X-Accel-Buffering: no``：禁止压缩与代理缓冲，
            否则客户端会看到"整块才到"，实时性直接没了
        """
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        return StreamingResponse(
            agen, media_type="text/event-stream", headers=headers)

    #: 终态通知的包装标记：作业结束时服务端会在 "end" 频道发一条 dict
    _TERMINAL = "__terminal__"

    async def _follow(job_id, kinds, request):
        """订阅 ``kinds``（末尾自动带上 ``end`` 终态频道）。

        产出 ``(item, terminal)``；心跳超时时产出 ``(None, False)``。
        收到终态帧后由调用方收流——客户端 EventSource 看到流结束即知作业已终结，
        不必再轮询。
        """
        hub = app.state.hub
        sub = await hub.subscribe_multi(job_id, list(kinds) + ["end"])
        hb = settings.heartbeat
        try:
            while not await request.is_disconnected():
                try:
                    item = await sub.get(timeout=hb)
                except asyncio.TimeoutError:
                    yield (None, False)
                    continue
                if isinstance(item, dict) and item.get("__end__"):
                    yield (item, True)
                    return
                yield (item, False)
        finally:
            await hub.unsubscribe_multi(job_id, sub)

    @app.get("/api/v1/train/jobs/{job_id}/logs/stream",
             dependencies=[Depends(require_auth)])
    async def stream_logs(job_id: str, request: Request,
                          tail: int = Query(50, ge=0, le=2000)):
        """原始日志 SSE。⚠️ **指标不要从这里解析**，那是 metrics 流的事。"""
        _must_get(job_id)

        async def gen():
            # 用 jobs.log_tail 而不是直接读环形缓冲：作业结束后环会被释放，
            # 但落盘的 service.log 还在——否则"跑完再看日志"就什么都拿不到了。
            for line in app.state.jobs.log_tail(job_id, tail):
                yield sse_log_frame(line)
            async for item, terminal in _follow(job_id, ["log"], request):
                if terminal:
                    yield sse_event(item, event="end")
                    return
                if item is None:
                    yield sse_comment("keepalive")
                else:
                    yield sse_log_frame(item)

        return _sse_response(gen())

    @app.get("/api/v1/train/jobs/{job_id}/metrics/stream",
             dependencies=[Depends(require_auth)])
    async def stream_metrics(
        job_id: str, request: Request,
        offset: Optional[int] = Query(
            None, description="从该 seq 之后开始；默认用作业已记录的游标"),
        last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    ):
        """指标 SSE。

        断点续传：客户端重连时带 ``Last-Event-ID``（或 ``?offset=``），
        服务端**先补发缺口**再切实时，刷新/断网都不丢不重。
        """
        job = _must_get(job_id)
        cursor = offset
        if cursor is None and last_event_id:
            try:
                cursor = int(last_event_id)
            except (TypeError, ValueError):
                cursor = None
        if cursor is None:
            cursor = job.get("metrics_seq", -1)

        async def gen():
            # 1) 先补历史缺口（seq > cursor）
            for ev in app.state.jobs.metrics_since_seq(job_id, cursor, limit=50000):
                yield sse_metric_frame(ev)
            # 2) 再切实时；补发与实时可能重叠，用 seq 去重
            async for ev, terminal in _follow(job_id, ["metric"], request):
                if terminal:
                    yield sse_event(ev, event="end")
                    return
                if ev is None:
                    yield sse_comment("keepalive")
                    continue
                seq = ev.get("seq")
                if isinstance(seq, int) and seq <= cursor:
                    continue
                yield sse_metric_frame(ev)

        return _sse_response(gen())

    return app


app = create_app()
