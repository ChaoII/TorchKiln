"""MetricSink ``env`` 事件测试：环境指标（GPU/CPU 采样）写入 ``metrics.jsonl`` 并声明到 ``meta.json``。

契约要点：
- env 行与现有事件同形：``{"ts", "seq", "type": "env", **fields}``，seq 由 sink 计数器分配；
- ``meta.json`` 的 ``metric_events`` 列表必须声明 ``env``，消费方据此识别事件类型；
- 禁用时（``_NullSink``）``env()`` 是空操作，不抛异常。
"""

import json

from ptcore.metrics_sink import build_sink


def test_env_event_written_and_meta_updated(tmp_path):
    sink = build_sink(str(tmp_path), enabled=True, meta={"schema_version": 1})
    sink.env({"gpu": [{"id": 0, "util": 42.0}], "cpu": 0.3})
    sink.close()
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    row = json.loads(lines[-1])
    # 首条事件 seq=0（沿用现有 emit 的 0-based 计数器语义；brief 里写的 ``> 0``
    # 与"不得改变现有 emit 行为"的全局契约冲突，这里按契约断言 ``>= 0``）
    assert row["type"] == "env" and row["gpu"][0]["util"] == 42.0 and row["seq"] >= 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert "env" in meta["metric_events"]


def test_null_sink_env_noop(tmp_path):
    s = build_sink(str(tmp_path), enabled=False)
    s.env({"cpu": 0.3})   # 不抛
