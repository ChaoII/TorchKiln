"""事件模型与回调列表。

简单模式的轻量事件抽象：训练器通过 CallbackList 广播事件，
各 sink（指标、采样器、日志等）订阅。纯标准库实现。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["EVENT_TYPES", "Event", "EventSink", "CallbackList"]

#: 全部合法事件类型（顺序即语义先后）。
EVENT_TYPES: tuple[str, ...] = ("step", "eval", "best", "env", "end", "failed")


@dataclass
class Event:
    """一条事件：自增序号 seq、epoch 秒 ts、类型 type 与载荷 fields。"""

    seq: int
    ts: float
    type: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """扁平化为 dict；载荷中值为 None 的键丢弃。"""
        return {
            "seq": self.seq,
            "ts": self.ts,
            "type": self.type,
            **{k: v for k, v in self.fields.items() if v is not None},
        }


class EventSink(Protocol):
    """事件订阅方协议：带 emit(type, fields) 方法的对象，或同签名普通函数。"""

    def emit(self, type: str, fields: dict[str, Any]) -> None:  # noqa: A002 - 与 brief 契约一致
        ...


class CallbackList:
    """事件广播器：维护自增 seq，向所有 sink 广播并做错误隔离。

    任一 sink 抛异常时只记录 warning，绝不向上传播。
    """

    def __init__(self, sinks: list[EventSink] | None = None) -> None:
        self._sinks: list[Any] = list(sinks or [])
        self._seq = 0

    def add(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def emit(self, type: str, fields: dict[str, Any]) -> None:
        self._seq += 1
        event_type = type
        payload = {"seq": self._seq, "ts": time.time(), "type": type, **fields}
        for sink in self._sinks:
            try:
                if hasattr(sink, "emit"):
                    sink.emit(event_type, payload)
                else:
                    sink(event_type, payload)
            except Exception:
                logging.warning(
                    "sink %s 处理事件 %r 时抛出异常，已忽略",
                    repr(sink),
                    event_type,
                    exc_info=True,
                )
