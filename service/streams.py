"""SSE 广播中心。

为什么用 SSE 而不是 WebSocket：
  指标与日志都是**单向**流，SSE 走普通 HTTP，浏览器原生 ``EventSource`` 自带
  自动重连，穿 Nginx/网关不用额外配置；而 WebSocket 需要 upgrade 握手、
  心跳、负载均衡粘性会话那套东西，对"吐点数字"而言过重。

每个作业两条流：
  ``log``    原始训练日志（排查用；**指标不要从这里解析**）
  ``metric`` ``metrics.jsonl`` 里的结构化事件（机器读）

断点续传约定：
  - 每条 metric 事件都带 SSE ``id: <seq>``；
  - 客户端重连时带 ``Last-Event-ID`` 头或 ``?offset=<seq>``，服务端**先补发缺口**
    （从 ``metrics.jsonl`` 按 seq 回放），再切到实时广播。
  - 没有这个机制，刷新页面就丢曲线、或重连后指标翻倍。
"""
from __future__ import absolute_import

import asyncio
import json
import time
from collections import deque

#: 单个订阅者的积压上限：慢客户端不能把服务端内存吃光
_QUEUE_MAX = 2048


class Subscription(object):
    """一个订阅者的有界队列。满了丢最旧的，并记 dropped 供排障。"""

    def __init__(self, maxsize=_QUEUE_MAX):
        self.q = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, item):
        try:
            self.q.put_nowait(item)
            return True
        except asyncio.QueueFull:
            # 丢最旧的一条给新数据腾位置（宁可丢历史也不要卡住生产者）
            try:
                self.q.get_nowait()
                self.dropped += 1
                self.q.put_nowait(item)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                self.dropped += 1
            return False

    async def get(self, timeout=None):
        if timeout is None:
            return await self.q.get()
        return await asyncio.wait_for(self.q.get(), timeout=timeout)


class _MultiSubscription(Subscription):
    """把多个频道桥接进一个队列的订阅。

    用途：指标流既要收 ``metric``，也要收作业终态的 ``end`` 通知。
    两条流的 payload 互不重叠，合并成一条流不会重复。
    """

    def __init__(self, sources):
        super(_MultiSubscription, self).__init__()
        self.sources = sources      # [(job_id, kind, Subscription), ...]


class Hub(object):
    """按 (job_id, kind) 组织的发布订阅。"""

    def __init__(self):
        self._subs = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id, kind):
        sub = Subscription()
        async with self._lock:
            self._subs.setdefault((job_id, kind), []).append(sub)
        return sub

    async def subscribe_multi(self, job_id, kinds):
        """同时订阅多个频道，共用一个队列。"""
        sources = []
        async with self._lock:
            for kind in kinds:
                sub = Subscription()
                self._subs.setdefault((job_id, kind), []).append(sub)
                sources.append((kind, sub))
        merged = _MultiSubscription(sources)

        async def bridge():
            while True:
                waits = [asyncio.ensure_future(s.q.get()) for _k, s in sources]
                done, pending = await asyncio.wait(
                    waits, return_when=asyncio.FIRST_COMPLETED)
                for fut in pending:
                    fut.cancel()
                for fut in done:
                    try:
                        merged.offer(fut.result())
                    except asyncio.CancelledError:
                        return

        merged._bridge = asyncio.ensure_future(bridge())
        return merged

    async def _drop(self, job_id, kind, sub):
        lst = self._subs.get((job_id, kind))
        if lst and sub in lst:
            lst.remove(sub)
            if not lst:
                self._subs.pop((job_id, kind), None)

    async def unsubscribe(self, job_id, kind, sub):
        async with self._lock:
            await self._drop(job_id, kind, sub)

    async def unsubscribe_multi(self, job_id, sub):
        async with self._lock:
            for kind, s in sub.sources:
                await self._drop(job_id, kind, s)
        bridge = getattr(sub, "_bridge", None)
        if bridge is not None:
            bridge.cancel()

    def subscriber_count(self, job_id, kind):
        return len(self._subs.get((job_id, kind), ()))

    def publish(self, job_id, kind, payload):
        """同步版（可从非 async 上下文调用）。返回送达人数。"""
        lst = self._subs.get((job_id, kind))
        if not lst:
            return 0
        n = 0
        for sub in list(lst):
            if sub.offer(payload):
                n += 1
        return n


# ------------------------------------------------------------------ 日志环形缓冲
class LogRing(object):
    """每个作业保留最近 N 行日志，供新订阅者/断线重连补发。"""

    def __init__(self, maxlen=2000):
        self._rings = {}
        self._maxlen = max(1, maxlen)

    def append(self, job_id, line):
        self._rings.setdefault(job_id, deque(maxlen=self._maxlen)).append(line)

    def tail(self, job_id, n=200):
        d = self._rings.get(job_id)
        if not d:
            return []
        if n <= 0 or n >= len(d):
            return list(d)
        return list(d)[-n:]

    def drop(self, job_id):
        self._rings.pop(job_id, None)


# ------------------------------------------------------------------ SSE 帧
def sse_event(data, event=None, event_id=None):
    """按 SSE 规范拼一帧（字段间必须换行分隔，空行表示帧结束）。"""
    out = []
    if event:
        out.append("event: {}".format(event))
    if event_id is not None:
        out.append("id: {}".format(event_id))
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    for line in payload.split("\n"):
        out.append("data: {}".format(line))
    return "\n".join(out) + "\n\n"


def sse_comment(text=""):
    """心跳/注释帧：`: ping`，代理会原样透传但浏览器不触发 onmessage。"""
    return ": {}\n\n".format(text or "ping")


def now():
    return time.time()
