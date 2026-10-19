"""ptcore.events 事件模型与回调列表测试。"""
from ptcore.events import CallbackList, Event


def test_event_to_dict_drops_none():
    e = Event(seq=1, ts=1.0, type="step", fields={"loss": 0.5, "x": None})
    assert e.to_dict() == {"seq": 1, "ts": 1.0, "type": "step", "loss": 0.5}


def test_callback_list_injects_seq_and_swallows_errors():
    out = {}
    def bad(type, fields): raise RuntimeError("network down")
    cb = CallbackList([bad, lambda t, f: out.update(f)])
    cb.emit("step", {"loss": 1.0})          # bad sink 抛错不影响第二个 sink
    assert out["seq"] == 1 and out["ts"] > 0 and out["type"] == "step"
    cb.emit("step", {"loss": 0.9})
    assert out["seq"] == 2
