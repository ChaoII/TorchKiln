"""BaseTrainer 事件回调与 env 周期采样测试（Task 4）。

测试策略：轻量 ``_FakeTrainer`` 只跑被测的事件接线
（``_init_events`` / ``_emit`` / ``_maybe_emit_env``），不构建
模型/数据集/优化器等重量依赖；``_emit`` 与 ``_maybe_emit_env``
是真实的 BaseTrainer 方法，不 mock 被测逻辑。
"""
import json

from ptcore.events import CallbackList
from ptcore.metrics_sink import _NullSink, build_sink
from ptcore.trainers.base import BaseTrainer


def _config(**global_extra):
    g = {"device": "cpu"}
    g.update(global_extra)
    return {"Global": g}


class _FakeTrainer(BaseTrainer):
    """轻量 trainer：只初始化被测的事件接线，其余依赖一律不构建。"""

    def __init__(self, config, event_sink=None, dump_config=True):
        self.config = config
        self.global_config = config["Global"]
        self.is_main = True
        # 真实 __init__ 必有 metrics_sink（dump_config=False 时为 _NullSink）；
        # _maybe_emit_env 的 JSONL 双路对它无 if 直调，这里用 _NullSink 兜底。
        self.metrics_sink = _NullSink()
        # 与真实 __init__ 相同：train_path = dump_config and is_main
        self._init_events(event_sink, train_path=dump_config and self.is_main)


def test_emit_delivers_event_with_seq_and_ts():
    events = []
    cb = CallbackList([lambda t, f: events.append((t, f))])
    trainer = _FakeTrainer(_config(), event_sink=cb)
    trainer._emit("step", {"loss": 0.1})
    assert events[-1][0] == "step"
    assert events[-1][1]["loss"] == 0.1
    # seq/ts 由 CallbackList 注入
    assert events[-1][1]["seq"] == 1 and events[-1][1]["ts"] > 0
    trainer._emit("end", {"exit_reason": "finished"})
    assert events[-1][0] == "end" and events[-1][1]["seq"] == 2


def test_emit_without_event_sink_is_noop():
    trainer = _FakeTrainer(_config(), event_sink=None)
    trainer._emit("step", {"loss": 0.1})  # 不得抛异常
    assert trainer._event_cb is not None


def test_env_emitted_after_interval(monkeypatch):
    events = []
    cb = CallbackList([lambda t, f: events.append((t, f))])
    trainer = _FakeTrainer(_config(), event_sink=cb)
    assert trainer._env_sampler is not None
    monkeypatch.setattr(trainer._env_sampler, "maybe_sample", lambda: {"cpu": 42.5})
    trainer._maybe_emit_env()
    assert events[-1][0] == "env"
    assert events[-1][1]["cpu"] == 42.5
    # maybe_sample 返回 None（未到间隔）时不产生 env 事件
    before = len(events)
    monkeypatch.setattr(trainer._env_sampler, "maybe_sample", lambda: None)
    trainer._maybe_emit_env()
    assert len(events) == before


def test_env_sampler_interval_from_config():
    trainer = _FakeTrainer(_config(env_sample_interval=0.25), event_sink=None)
    assert trainer._env_sampler._interval == 0.25
    default = _FakeTrainer(_config(), event_sink=None)
    assert default._env_sampler._interval == 5.0


def test_env_event_dual_path_writes_metrics_jsonl(tmp_path, monkeypatch):
    """I1 双路输出：_maybe_emit_env 同时写 metrics.jsonl（真实 build_sink）+ 回调通道。"""
    events = []
    cb = CallbackList([lambda t, f: events.append((t, f))])
    trainer = _FakeTrainer(_config(), event_sink=cb)
    # 真实 sink（不 mock 被测接线）：把 _FakeTrainer 的 _NullSink 换成 build_sink
    trainer.metrics_sink = build_sink(
        str(tmp_path), enabled=True, meta={"framework": "torchkiln"})
    sample = {"cpu": 12.5, "mem": {"used_mb": 100.0, "total_mb": 200.0}}
    monkeypatch.setattr(trainer._env_sampler, "maybe_sample", lambda: sample)
    trainer._maybe_emit_env()

    # 回调通道
    assert events[-1][0] == "env" and events[-1][1]["cpu"] == 12.5
    # JSONL 通道：metrics.jsonl 里真实出现 type=="env" 行且字段一致
    with open(trainer.metrics_sink.jsonl_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    env_lines = [line for line in lines if line.get("type") == "env"]
    assert len(env_lines) == 1
    assert env_lines[0]["cpu"] == 12.5
    assert env_lines[0]["mem"] == {"used_mb": 100.0, "total_mb": 200.0}
    assert {"ts", "seq", "type"} <= set(env_lines[0])


def test_dump_config_false_still_emits():
    """dump_config=False 时 metrics sink 是 _NullSink，但 event_sink 仍收到事件。"""
    events = []
    cb = CallbackList([lambda t, f: events.append((t, f))])
    trainer = _FakeTrainer(_config(), event_sink=cb, dump_config=False)
    assert trainer._env_sampler is None
    trainer._maybe_emit_env()  # 非训练路径：no-op，不得抛异常
    trainer._emit("step", {"loss": 0.7})
    assert events[-1][0] == "step" and events[-1][1]["loss"] == 0.7


def test_step_loss_is_number():
    """step 事件的 loss 必须可 JSON 序列化为数字（GPU 烟雾发现的回归）。"""
    import pytest

    from ptcore.trainers.base import _num
    import torch

    assert _num(torch.tensor(3.14)) == pytest.approx(3.14)  # float32 精度
    assert _num(2.5) == 2.5
    # 带梯度 tensor 走 detach 路径，不产生 autograd 告警
    t = torch.tensor(1.5, requires_grad=True)
    assert _num(t) == 1.5
    # 关键断言：序列化后是数字而非 "tensor(...)" 字符串
    import json as _json

    s = _json.dumps({"loss": _num(torch.tensor(2.5))}, default=str)
    assert '"loss": 2.5' in s and "tensor" not in s


def test_build_trainer_passes_event_sink(monkeypatch):
    """event_sink 经 build_trainer 的 kwargs 原样到达 trainer 构造。"""
    from ptcore import trainers

    captured = {}

    class _Fake:
        def __init__(self, config, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(trainers.TRAINER_REGISTRY, "detect", _Fake)
    sink = object()
    trainers.build_trainer({"Architecture": {"task": "detect"}}, event_sink=sink)
    assert captured["event_sink"] is sink
