"""评估作业（``kind="eval"``）的 argv 构造与端点契约测试。

为什么需要
----------
评估原先只有「起一次性容器跑 ``tkiln val`` CLI」这一条路，AIStation 侧靠
**正则解析控制台日志**拿指标（``cur metric, ...`` / ``main indicator (...)``）。
日志格式随版本变，小版本一改指标就**静默变成空**。

做成 HTTP 作业后有两处硬依赖：

1. ``tkiln val`` 必须写 ``metrics.jsonl`` 并含 ``end`` 事件，否则
   ``classify_exit`` 会把"进程正常退出但没 end"判成 ``no_end_event / failed``；
2. 指标必须能被 HTTP 消费方直接读到，不能靠解析日志。

这里把第1 条的前提（argv 形状）与第 2 条的端点契约钉住。
"""

import json

import pytest

from service import runner
from service.schemas import JOB_KINDS, JobSpec


# ---------------------------------------------------------------------------
# 契约常量
# ---------------------------------------------------------------------------


def test_job_kinds_contains_train_eval_predict():
    assert JOB_KINDS == ("train", "eval", "predict")


def test_spec_kind_defaults_to_train():
    """老客户端不带 kind 时行为不变。"""
    assert JobSpec().kind == "train"
    assert JobSpec().validate_kind() == "train"


@pytest.mark.parametrize("raw,expected", [
    ("eval", "eval"), ("EVAL", "eval"), (" eval ", "eval"), ("train", "train"),
])
def test_validate_kind_normalizes(raw, expected):
    assert JobSpec(kind=raw).validate_kind() == expected


def test_validate_kind_rejects_unknown_with_helpful_message():
    # "predict" 已是一等作业种类，所以拿一个真正未知的来测
    with pytest.raises(ValueError) as ei:
        JobSpec(kind="export").validate_kind()
    msg = str(ei.value)
    assert "export" in msg
    # 错误信息要能告诉调用方有哪些可选值
    assert "train" in msg and "eval" in msg and "predict" in msg


def test_validate_kind_accepts_predict():
    assert JobSpec(kind="predict").validate_kind() == "predict"


# ---------------------------------------------------------------------------
# build_eval_argv
# ---------------------------------------------------------------------------


def _spec(**kw):
    return JobSpec(**kw)


def test_eval_argv_basic_shape():
    argv = runner.build_eval_argv(
        _spec(kind="eval", config_path="x"), r"C:\repo\configs\yolo11-seg.yml",
        "/out", "python", r"C:\repo")
    assert argv[0:4] == ["python", "-m", "torchkiln", "val"]
    # 路径必须是**相对仓库根**的 POSIX 形式：容器内 cwd 是 repo_root
    assert "configs/yolo11-seg.yml" in argv
    assert "\\" not in argv[argv.index("-c") + 1]


def test_eval_argv_uses_dedicated_weights_flag():
    """权重走 ``--weights`` 而不是 ``-o Global.pretrained_model=``。"""
    argv = runner.build_eval_argv(
        _spec(kind="eval", weights_path="/model/best.pth"), "/repo/c.yml",
        "/out", "python", "/repo")
    assert "--weights" in argv
    assert argv[argv.index("--weights") + 1] == "/model/best.pth"
    assert not any("Global.pretrained_model" in a for a in argv)


def test_eval_argv_omits_weights_flag_when_absent():
    argv = runner.build_eval_argv(
        _spec(kind="eval"), "/repo/c.yml", "/out", "python", "/repo")
    assert "--weights" not in argv


def test_eval_argv_injects_both_datasets():
    """必须**同时**注入 ``Train.dataset.*``——见下面那条实跑回归。

    实跑发现：``build_trainer`` 在 ``BaseTrainer.__init__`` 里无条件构造两个
    数据集，构造期就会 ``open(label_file)``。只给 ``Eval.dataset.*`` 时训练集
    退回配置默认值，作业在 trainer 构造阶段就死：
    ``FileNotFoundError: datasets/seg_demo/train.txt`` → ``setup_failed``。
    这条断言就是为了不让那个 bug 再回来。
    """
    from service.schemas import DatasetRef

    argv = runner.build_eval_argv(
        _spec(kind="eval", dataset=DatasetRef(
            data_dir="/data", train_list="/data/train.txt", val_list="/data/val.txt")),
        "/repo/c.yml", "/out", "python", "/repo")

    opts = " ".join(argv)
    assert "Eval.dataset.data_dir=/data" in opts
    assert "Eval.dataset.label_file_list" in opts
    assert "Train.dataset.data_dir=/data" in opts, (
        "缺 Train.dataset.data_dir —— trainer 构造期会读配置默认值并 FileNotFoundError")
    assert "Train.dataset.label_file_list" in opts


def test_eval_argv_points_train_at_the_val_list():
    """训练集指向**评估那份**清单，不是 train_list。

    ``tools/eval.py`` 把 ``epoch_num`` 设成 0，训练集构造完不会被迭代，
    所以指向哪份不影响结果。这样契约更小：调用方只需导出一份清单。
    """
    from service.schemas import DatasetRef

    argv = runner.build_eval_argv(
        _spec(kind="eval", dataset=DatasetRef(
            data_dir="/data", train_list="/data/train.txt", val_list="/data/val.txt")),
        "/repo/c.yml", "/out", "python", "/repo")
    opts = " ".join(argv)
    assert "Train.dataset.label_file_list=[\"/data/val.txt\"]" in opts
    assert "/data/train.txt" not in opts


def test_eval_argv_without_val_list_injects_neither():
    """没给 val_list 时两边都不注入——不凭空造清单。"""
    from service.schemas import DatasetRef

    argv = runner.build_eval_argv(
        _spec(kind="eval", dataset=DatasetRef(data_dir="/d", train_list="/d/train.txt")),
        "/repo/c.yml", "/out", "python", "/repo")
    opts = " ".join(argv)
    assert "Eval.dataset.label_file_list" not in opts
    assert "Train.dataset.label_file_list" not in opts


# ---------------------------------------------------------------------------
# classify_exit 的文案按 kind 分（实跑时踩到：评估作业报「训练未开始即失败」）
# ---------------------------------------------------------------------------


def test_classify_exit_wording_covers_all_three_kinds():
    """三种 kind 的失败文案必须各自说对——这是用户看到的第一现场。

    实跑抓到过两次退化：
    1. 文案写死「训练」，于是**评估**作业失败时报「训练未开始即失败」；
    2. 改成 train / 其它 两分之后，**预测**作业又报「评估进程被强杀」。
    现在是三种各自对应，未知 kind 用中性文案。
    """
    msgs = {}
    for kind in ("train", "eval", "predict"):
        for has_file in (False, True):
            _s, _r, msg = runner.classify_exit(
                None, 1, False, has_metrics_file=has_file, kind=kind)
            msgs[(kind, has_file)] = msg

    assert "训练未开始即失败" in msgs[("train", False)]
    assert "评估未开始即失败" in msgs[("eval", False)]
    assert "预测未开始即失败" in msgs[("predict", False)]

    assert "训练进程被强杀" in msgs[("train", True)]
    assert "评估进程被强杀" in msgs[("eval", True)]
    assert "预测进程被强杀" in msgs[("predict", True)]

    # 交叉检查：每条文案里都不该出现别的 kind 的字眼
    words = {"train": "训练", "eval": "评估", "predict": "预测"}
    for (kind, _hf), msg in msgs.items():
        for other, w in words.items():
            if other != kind:
                assert w not in msg, f"{kind} 的文案里混进了 {other} 的字眼: {msg}"


def test_classify_exit_unknown_kind_falls_back_neutral():
    """未知 kind 用中性文案，而不是错说成训练。"""
    _s, _r, msg = runner.classify_exit(None, 1, False, False, kind="weird")
    assert "作业未开始即失败" in msg
    assert "训练" not in msg


def test_classify_exit_kind_defaults_to_train():
    """旧调用点不传 kind 时行为不变。"""
    _s, _r, msg = runner.classify_exit(None, 1, False, has_metrics_file=False)
    assert "训练未开始即失败" in msg


def test_classify_exit_status_independent_of_kind():
    """判定逻辑与 kind 无关——否则「什么算失败」会有两套解释。"""
    for kind in ("train", "eval"):
        assert runner.classify_exit(None, 1, False, False, kind=kind)[0] == "failed"
        assert runner.classify_exit(None, None, False, False, kind=kind)[1] == "no_end_event"
        assert runner.classify_exit(None, 0, True, False, kind=kind)[0] == "cancelled"
        assert runner.classify_exit(
            {"exit_reason": "finished"}, 0, False, True, kind=kind)[0] == "succeeded"


def test_eval_argv_forces_output_dir_and_metrics_sink():
    """评估的产物是指标：不强制落到 output_dir、sink 不开，就没人读得到。"""
    argv = runner.build_eval_argv(
        _spec(kind="eval", params={"Global.save_model_dir": "/tmp/hack",
                                   "Global.metrics_sink": False}),
        "/repo/c.yml", "/out", "python", "/repo")
    opts = " ".join(argv)
    assert "Global.save_model_dir=/out" in opts
    assert "Global.metrics_sink=true" in opts
    # 平台托管的键必须**在参数之后**覆盖掉外部的同名项
    assert opts.index("Global.save_model_dir=/out") > opts.index("/tmp/hack")


def test_eval_argv_passes_params_and_seed():
    argv = runner.build_eval_argv(
        _spec(kind="eval", params={"Global.epoch_num": 1, "Eval.batch_size": 8}, seed=7),
        "/repo/c.yml", "/out", "python", "/repo")
    opts = " ".join(argv)
    assert "Global.epoch_num=1" in opts
    assert "Eval.batch_size=8" in opts
    assert "Global.seed=7" in opts


def test_eval_argv_no_opts_when_nothing_to_inject():
    """没有可注入项时不要留一个光秃秃的 ``-o``——argparse 会把它当空列表参数。"""
    argv = runner.build_eval_argv(_spec(kind="eval"), "/repo/c.yml", "/out", "python", "/repo")
    # 平台托管的键总会注入，所以 -o 必然存在；但不应出现孤立的 "-o" 后面直接跟 -o
    assert argv.count("-o") == 1
    assert argv[argv.index("-o") + 1].startswith("Global.save_model_dir=")


# ---------------------------------------------------------------------------
# build_argv 分派
# ---------------------------------------------------------------------------


def test_build_argv_dispatches_to_train():
    argv = runner.build_argv(_spec(kind="train"), "/repo/c.yml", "/out", "python", "/repo")
    assert argv[:4] == ["python", "-m", "torchkiln", "train"]
    assert "--weights" not in argv


def test_build_argv_dispatches_to_eval():
    argv = runner.build_argv(_spec(kind="eval", weights_path="/m/b.pth"),
                             "/repo/c.yml", "/out", "python", "/repo")
    assert argv[:4] == ["python", "-m", "torchkiln", "val"]
    assert "--weights" in argv


def test_build_argv_defaults_to_train_when_kind_absent():
    spec = JobSpec()
    spec.kind = ""            # 老数据/老客户端可能是空串
    argv = runner.build_argv(spec, "/repo/c.yml", "/out", "python", "/repo")
    assert argv[:4] == ["python", "-m", "torchkiln", "train"]


def test_build_argv_rejects_unknown_kind():
    with pytest.raises(ValueError):
        runner.build_argv(_spec(kind="nope"), "/repo/c.yml", "/out", "python", "/repo")


def test_builders_table_covers_every_kind():
    """JOB_KINDS 与 argv 构造器表必须同步——加一种漏改表就会在运行时才炸。"""
    assert set(JOB_KINDS) == set(runner._ARGV_BUILDERS)


# ---------------------------------------------------------------------------
# 端点契约
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    from service.main import create_app

    return create_app()


def _paths(app):
    out = set()
    for r in app.routes:
        if hasattr(r, "methods"):
            for m in r.methods:
                out.add((m, r.path))
    return out


def test_eval_submit_endpoint_exists(app):
    assert ("POST", "/api/v1/eval/jobs") in _paths(app)


def test_eval_lifecycle_endpoints_exist(app):
    """评估作业也要能查/取消/取产物/看日志与指标，否则只能提交不能管。"""
    paths = _paths(app)
    for method, path in [
        ("GET", "/api/v1/eval/jobs"),
        ("GET", "/api/v1/eval/jobs/{job_id}"),
        ("POST", "/api/v1/eval/jobs/{job_id}/cancel"),
        ("GET", "/api/v1/eval/jobs/{job_id}/artifacts"),
        ("GET", "/api/v1/eval/jobs/{job_id}/metrics"),
        ("GET", "/api/v1/eval/jobs/{job_id}/metrics/stream"),
        ("GET", "/api/v1/eval/jobs/{job_id}/logs"),
        ("GET", "/api/v1/eval/jobs/{job_id}/logs/stream"),
    ]:
        assert (method, path) in paths, f"缺少 {method} {path}"


def test_train_endpoints_still_exist(app):
    """加 eval 不能把原有训练通路弄坏。"""
    paths = _paths(app)
    for method, path in [
        ("POST", "/api/v1/train/jobs"),
        ("GET", "/api/v1/train/jobs"),
        ("GET", "/api/v1/train/jobs/{job_id}"),
        ("POST", "/api/v1/train/jobs/{job_id}/cancel"),
        ("GET", "/api/v1/train/jobs/{job_id}/metrics/stream"),
    ]:
        assert (method, path) in paths, f"缺少 {method} {path}"


def test_no_duplicate_eval_list_route(app):
    """/api/v1/eval/jobs 只能有一条 GET——重复注册会在 OpenAPI 里出两条。"""
    gets = [p for m, p in _paths(app) if m == "GET" and p == "/api/v1/eval/jobs"]
    assert len(gets) == 1


# ---------------------------------------------------------------------------
# tools/eval.py 的指标契约（终态判定的硬依赖）
# ---------------------------------------------------------------------------


def _read_eval_py():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[1].joinpath(
        "tools", "eval.py").read_text(encoding="utf-8")


def test_eval_writes_metrics_contract():
    """必须写 eval + end 两条事件，否则 classify_exit 判 no_end_event/failed。"""
    src = _read_eval_py()
    assert "sink.eval(" in src, "评估没写 eval 事件"
    assert "sink.end(" in src, "评估没写 end 事件——作业会永远卡在 running"


def test_eval_writes_end_before_raise():
    """评估抛异常时也必须收尾写 end，否则外部服务等不到终态。"""
    src = _read_eval_py()
    assert "except Exception" in src
    # except 分支里必须有 sink.end
    idx = src.index("except Exception")
    seg = src[idx:idx + 400]
    assert "sink.end(" in seg


def test_eval_no_dataset_is_an_error_not_silence():
    """没有评估数据集时必须非零退出，不能「成功但没指标」。"""
    src = _read_eval_py()
    assert "sys.exit(" in src


def test_eval_keeps_human_readable_marker_line():
    """保留 EVAL_METRIC_JSON 标记行：给人看，也让仍在解析日志的消费方不断线。"""
    src = _read_eval_py()
    assert "EVAL_METRIC_JSON" in src
    assert "cur metric" in src and "main indicator" in src


def test_eval_argv_and_cli_marker_use_same_terminology():
    """AIStation 侧解析的是 EVAL_METRIC_JSON；这里用常量导出，避免拼写漂移。"""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1].joinpath("tools", "eval.py")
    spec = importlib.util.spec_from_file_location("_tk_eval_probe", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 —— 该模块会 import torch，缺依赖时跳过
        pytest.skip("无法导入 tools/eval.py（可能缺 torch 依赖）")
    assert mod.METRIC_MARKER.strip() == "EVAL_METRIC_JSON"


def test_marker_line_payload_is_valid_json_object():
    """标记行内容必须是合法 JSON 对象——消费方直接 json.loads，不做兜底。"""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1].joinpath("tools", "eval.py")
    spec = importlib.util.spec_from_file_location("_tk_eval_probe2", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        pytest.skip("无法导入 tools/eval.py（可能缺 torch 依赖）")

    payload = {"box_mAP50": 0.87, "fps": 12.3,
               "main_indicator": "hmean", "main_value": 0.83}
    line = mod.METRIC_MARKER + json.dumps(payload, ensure_ascii=False, allow_nan=False)
    body = line.split(mod.METRIC_MARKER, 1)[1].strip()
    assert json.loads(body) == payload