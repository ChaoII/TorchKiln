"""torchkiln.simple_run 单作业进程入口测试（简单模式）。

覆盖：
  - JobSpec 解析与四种 kind 分派（train / eval / predict / export）；
  - 终态事件契约：正常路径 end（exit_reason=finished），无法构造 end 的启动
    即失败用 failed，且**不发双终态事件**；
  - HttpEventSink「永不抛 + 重试 3 次后丢弃」与 body 契约
    （``{"task_id": ..., "seq": ..., "ts": ..., "type": ..., **fields}``）。

不真实启动训练/推理：``build_trainer`` / ``Runtime.load`` / ``subprocess.run``
用假对象替换；事件通过 monkeypatch ``build_http_events`` 注入的记录型 sink
捕获（HTTP 传输本身在专门的 sink 测试里验证）。
"""
import json
import sys
import types
from pathlib import Path

import torchkiln.simple_run as sr

REPO = Path(__file__).resolve().parents[1]


class RecordingSink:
    """与 HttpEventSink 同接口（emit/close/terminal_emitted）的记录器。"""

    def __init__(self):
        self.events = []
        self.terminal_emitted = False

    def emit(self, type, fields):  # noqa: A002 - 与 EventSink 契约一致
        p = dict(fields or {})
        if "type" not in p:
            p["type"] = type
        if p["type"] in ("end", "failed"):
            self.terminal_emitted = True
        self.events.append(p)

    def close(self, timeout=15):
        pass

    def types(self):
        return [e.get("type") for e in self.events]


def run_one(argv, monkeypatch):
    """跑一次 main(argv)（带 --repo-root），返回 (退出码, 记录 sink)。"""
    rec = RecordingSink()
    monkeypatch.setattr(sr, "build_http_events",
                        lambda url, token, task_id=None: rec)
    rc = sr.main(list(argv) + ["--repo-root", str(REPO)])
    return rc, rec


def _write_spec(tmp_path, **overrides):
    spec = {"spec_version": "1.0", "framework": "torchkiln",
            "kind": "train",
            "config_path": "configs/yolo/yolo11-det.yml",
            "params": {}, "resources": {"gpu": 1}}
    spec.update(overrides)
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return str(sp)


# ---------------------------------------------------------------- train
def test_train_dispatch_injects_sink_and_returns_zero(tmp_path, monkeypatch):
    captured = {}

    class FakeTrainer:
        def __init__(self, config, **kwargs):
            self.config = config
            self.kwargs = kwargs

        def train(self):
            # 真 trainer 在 finally 里发 end（含 error 路径），这里模拟之
            self.kwargs["event_sink"].emit("end", {"exit_reason": "finished"})

    def fake_build(config, **kwargs):
        captured.update(kwargs)
        captured["config"] = config
        return FakeTrainer(config, **kwargs)

    monkeypatch.setattr(sr, "build_trainer", fake_build)
    sp = _write_spec(tmp_path, kind="train", model_name="yolo11n",
                     params={"Global.epoch_num": 1})
    rc, rec = run_one([sp, "--callback-url", "http://127.0.0.1:1/x",
                       "--token", "t"], monkeypatch)

    assert rc == 0
    assert captured.get("event_sink") is rec, "event_sink 必须注入 trainer"
    # spec.params 的点分键必须落到 config 上
    assert captured["config"]["Global"]["epoch_num"] == 1
    # 终态：只有一条 end（trainer 发的），simple_run 层不得再补
    assert rec.types().count("end") == 1
    assert rec.events[-1]["exit_reason"] == "finished"


def test_train_builder_failure_emits_single_error_end(tmp_path, monkeypatch):
    def boom(config, **kwargs):
        raise RuntimeError("config exploded")
    monkeypatch.setattr(sr, "build_trainer", boom)
    sp = _write_spec(tmp_path, kind="train")
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 1
    # trainer 没建起来，trainer 的 finally 不会执行，但 end 仍可构造：
    # 四 kind 一致发 end(error)，不发 failed，且只有这一条终态
    assert rec.types() == ["end"]
    assert rec.events[0]["exit_reason"] == "error"
    assert "config exploded" in rec.events[0].get("error", "")


# ---------------------------------------------------------------- eval
def test_eval_dispatch_emits_eval_and_end(tmp_path, monkeypatch):
    captured = {}

    class FakeTrainer:
        main_indicator = "hmean"
        _indicator_mode = "max"

        def __init__(self, config, **kwargs):
            self.config = config

        def evaluate(self):
            return {"hmean": 0.9, "precision": 0.85, "recall": 0.91, "fps": 12.0}

    def fake_build(config, **kwargs):
        captured.update(kwargs)
        captured["config"] = config
        return FakeTrainer(config, **kwargs)

    monkeypatch.setattr(sr, "build_trainer", fake_build)
    sp = _write_spec(tmp_path, kind="eval", weights_path="w.pth",
                     dataset={"data_dir": "/data", "val_list": "/data/val.txt"})
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 0
    # 对齐 tools/eval.py 语义
    assert captured["config"]["Global"]["epoch_num"] == 0
    assert captured["config"]["Global"]["use_ema"] is False
    assert captured["config"]["Global"]["pretrained_model"] == "w.pth"
    assert captured.get("dump_config") is False
    # eval 作业必须注入 Eval 数据集（且 Train 指向同一份，对齐 runner 语义）
    assert captured["config"]["Eval"]["dataset"]["label_file_list"] == ["/data/val.txt"]
    assert captured["config"]["Train"]["dataset"]["label_file_list"] == ["/data/val.txt"]
    # 事件：eval 一条 + end 一条（simple_run 层补 end，因为 evaluate() 不发终态）
    assert rec.types() == ["eval", "end"]
    ev = rec.events[0]
    assert ev["main_indicator"] == "hmean" and ev["main_value"] == 0.9
    assert ev["metrics"]["hmean"] == 0.9 and "fps" not in ev["metrics"]
    assert rec.events[-1]["exit_reason"] == "finished"


def test_eval_exception_emits_single_error_end(tmp_path, monkeypatch):
    class FakeTrainer:
        main_indicator = "hmean"

        def __init__(self, config, **kwargs):
            pass

        def evaluate(self):
            raise RuntimeError("eval blew up")

    monkeypatch.setattr(sr, "build_trainer",
                        lambda config, **kw: FakeTrainer(config, **kw))
    sp = _write_spec(tmp_path, kind="eval", weights_path="w.pth")
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 1
    # 补一条 end(error) 即可，绝不双终态
    assert rec.types() == ["end"]
    assert rec.events[0]["exit_reason"] == "error"
    assert "eval blew up" in rec.events[0].get("error", "")


def test_eval_no_dataset_emits_error_end(tmp_path, monkeypatch):
    class FakeTrainer:
        main_indicator = "hmean"

        def __init__(self, config, **kwargs):
            pass

        def evaluate(self):
            return None

    monkeypatch.setattr(sr, "build_trainer",
                        lambda config, **kw: FakeTrainer(config, **kw))
    sp = _write_spec(tmp_path, kind="eval", weights_path="w.pth")
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 1
    assert rec.types() == ["end"]
    assert rec.events[0]["exit_reason"] == "error"


# ---------------------------------------------------------------- predict
def test_predict_dispatch_batches_and_emits_progress(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    indir = tmp_path / "imgs"
    (indir / "sub").mkdir(parents=True)
    for p in ("a.jpg", "sub/b.png"):
        cv2.imwrite(str(indir / p), np.zeros((16, 16, 3), dtype=np.uint8))
    outdir = str(tmp_path / "out")

    loaded = {}

    class FakeRuntime:
        @staticmethod
        def load(config, weights, device="cpu", overrides=None):
            loaded.update(config=config, weights=weights,
                          device=device, overrides=overrides)
            return FakeRuntime()

        def infer(self, img0):
            return types.SimpleNamespace(notes=["detections: 1"])

        def visualize(self, img0, result):
            return img0.copy()

    monkeypatch.setattr(sr, "Runtime", FakeRuntime)
    sp = _write_spec(tmp_path, kind="predict", weights_path="w.pth",
                     input_dir=str(indir),
                     params={"Global.output": outdir})
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 0
    # resources.gpu=1 -> cuda:0；权重原样透传
    assert loaded["weights"] == "w.pth"
    assert loaded["device"] == "cuda:0"
    # 结果图保持输入相对路径，扩展名统一 .jpg
    assert (Path(outdir) / "a.jpg").is_file()
    assert (Path(outdir) / "sub" / "b.jpg").is_file()
    # predict 进度事件与 predict_yolo 同字段，逐张上报
    predicts = [e for e in rec.events if e["type"] == "predict"]
    assert predicts and predicts[-1] == {"type": "predict",
                                         "images_total": 2, "images_done": 2}
    assert rec.events[-1]["type"] == "end"
    assert rec.events[-1]["exit_reason"] == "finished"
    assert rec.events[-1]["images_done"] == 2


def test_predict_failure_emits_error_end(tmp_path, monkeypatch):
    class FakeRuntime:
        @staticmethod
        def load(config, weights, device="cpu", overrides=None):
            raise FileNotFoundError("Could not resolve weights: w.pth")

    monkeypatch.setattr(sr, "Runtime", FakeRuntime)
    sp = _write_spec(tmp_path, kind="predict", weights_path="w.pth",
                     input_dir=str(tmp_path / "imgs"))
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 1
    assert rec.types() == ["end"]
    assert rec.events[0]["exit_reason"] == "error"
    assert "weights" in rec.events[0].get("error", "")


# ---------------------------------------------------------------- export
def test_export_dispatch_subprocess_success(tmp_path, monkeypatch):
    calls = []

    class R:
        returncode = 0

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return R()

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    sp = _write_spec(tmp_path, kind="export", weights_path="w.pth",
                     params={"Global.output": "/tmp/exp"})
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 0
    cmd, kw = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == "tools/export.py"
    assert cmd[cmd.index("--weights") + 1] == "w.pth"
    assert cmd[cmd.index("--save-dir") + 1] == "/tmp/exp"
    assert kw.get("cwd") == str(REPO)
    assert rec.events[-1]["type"] == "end"
    assert rec.events[-1]["exit_reason"] == "finished"


def test_export_failure_emits_error_end(tmp_path, monkeypatch):
    class R:
        returncode = 1

    monkeypatch.setattr(sr.subprocess, "run", lambda cmd, **kw: R())
    sp = _write_spec(tmp_path, kind="export", weights_path="w.pth")
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 1
    assert rec.events[-1]["type"] == "end"
    assert rec.events[-1]["exit_reason"] == "error"
    assert "1" in rec.events[-1].get("error", "")


def test_export_missing_weights_emits_single_error_end(tmp_path, monkeypatch):
    # spec 已解析但 job 无法开始：end 可构造 -> end(error)，不发 failed
    calls = []
    monkeypatch.setattr(sr.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    sp = _write_spec(tmp_path, kind="export")   # 无 weights_path
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)

    assert rc == 1
    assert not calls, "缺 weights_path 不应真的起 export 子进程"
    assert rec.types() == ["end"]
    assert rec.events[0]["exit_reason"] == "error"
    assert "weights_path" in rec.events[0].get("error", "")


# ---------------------------------------------------------------- 启动即失败
def test_missing_spec_file_emits_failed(tmp_path, monkeypatch):
    rc, rec = run_one([str(tmp_path / "nope.json"), "--callback-url", "u",
                       "--token", "t"], monkeypatch)
    assert rc == 1
    assert rec.types() == ["failed"]


def test_unknown_kind_emits_failed(tmp_path, monkeypatch):
    sp = _write_spec(tmp_path, kind="bogus")
    rc, rec = run_one([sp, "--callback-url", "u", "--token", "t"], monkeypatch)
    assert rc == 1
    assert rec.types() == ["failed"]


# ---------------------------------------------------------------- HttpEventSink
def test_http_sink_retries_three_times_then_drops(monkeypatch):
    attempts = []

    def boom(req, timeout=None):
        attempts.append(req)
        raise OSError("connection refused")

    monkeypatch.setattr(sr.urllib.request, "urlopen", boom)
    monkeypatch.setattr(sr, "RETRY_DELAYS", (0.01, 0.01, 0.01))
    sink = sr.HttpEventSink("http://127.0.0.1:1/x", "tok", task_id=7)

    sink.emit("step", {"loss": 0.5, "x": None})   # 永不阻塞/上抛
    sink.close(timeout=10)

    assert len(attempts) == 4, "1 次首发 + 3 次重试，仍失败则丢弃"
    req = attempts[0]
    body = json.loads(req.data.decode("utf-8"))
    # body 契约：{"task_id": ..., **event.to_dict()}，None 字段丢弃
    assert body["task_id"] == 7
    assert body["type"] == "step" and body["loss"] == 0.5
    assert body["seq"] == 1 and body["ts"] > 0
    assert "x" not in body
    assert req.get_header("X-torchkiln-token") == "tok"
    assert sink.terminal_emitted is False


def test_http_sink_posts_once_on_success(monkeypatch):
    sent = []

    class Resp:
        def read(self):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def ok(req, timeout=None):
        sent.append(req)
        return Resp()

    monkeypatch.setattr(sr.urllib.request, "urlopen", ok)
    sink = sr.HttpEventSink("http://127.0.0.1:9/x", "tok")
    sink.emit("end", {"exit_reason": "finished"})
    sink.close(timeout=10)

    assert len(sent) == 1
    body = json.loads(sent[0].data.decode("utf-8"))
    assert body["type"] == "end" and body["exit_reason"] == "finished"
    assert body["seq"] == 1
    assert sink.terminal_emitted is True


def test_http_sink_accepts_callbacklist_payload(monkeypatch):
    """CallbackList 广播时 fields 里已带 seq/ts/type，不得重复注入。"""
    sent = []

    class Resp:
        def read(self):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def ok(req, timeout=None):
        sent.append(req)
        return Resp()

    monkeypatch.setattr(sr.urllib.request, "urlopen", ok)
    sink = sr.HttpEventSink("http://127.0.0.1:9/x", "tok")
    # 模拟 CallbackList.emit 的调用形态
    sink.emit("step", {"seq": 5, "ts": 1234.5, "type": "step", "loss": 0.1})
    sink.close(timeout=10)
    assert len(sent) == 1
    body = json.loads(sent[0].data.decode("utf-8"))
    assert body["seq"] == 5 and body["ts"] == 1234.5 and body["loss"] == 0.1
