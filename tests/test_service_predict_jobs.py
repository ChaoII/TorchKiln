"""预测作业（``kind="predict"``）的 argv 构造、契约与端点测试。

与评估的差别正是需要被守住的地方
----------------------------------
预测的产物是**结果图**而不是指标，所以：

- ``--output`` 指向结果目录，**不是**训练/评估的 ``Global.save_model_dir``；
- 指标事件报的是「处理了多少张图 / 耗时」，**没有** mAP 之类；
- 但 ``end`` 事件必须写——服务侧把「进程退出且没有 end」判成
  ``no_end_event / failed``，哪怕退出码是 0。
"""

import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from service import runner  # noqa: E402
from service.schemas import JOB_KINDS, JobSpec  # noqa: E402

PREDICT_YOLO = pathlib.Path(__file__).resolve().parents[1] / "tools" / "infer" / "predict_yolo.py"


def _spec(**kw):
    kw.setdefault("kind", "predict")
    kw.setdefault("weights_path", "/workspace/model/best.pth")
    kw.setdefault("input_dir", "/workspace/data")
    return JobSpec(**kw)


def _argv(**kw):
    return runner.build_predict_argv(
        _spec(**kw), "/repo/configs/yolo11-seg.yml",
        "/workspace/jobs/job_1", "python", "/repo")


# ---------------------------------------------------------------------------
# 契约常量
# ---------------------------------------------------------------------------


def test_predict_is_a_first_class_kind():
    assert "predict" in JOB_KINDS


def test_builders_table_covers_predict():
    assert "predict" in runner._ARGV_BUILDERS
    assert set(JOB_KINDS) == set(runner._ARGV_BUILDERS)


def test_build_argv_dispatches_to_predict():
    argv = runner.build_argv(_spec(), "/repo/c.yml", "/out", "python", "/repo")
    assert argv[:4] == ["python", "-m", "torchkiln", "predict"]


# ---------------------------------------------------------------------------
# argv 形状
# ---------------------------------------------------------------------------


def test_predict_argv_basic_shape():
    argv = _argv()
    assert argv[:4] == ["python", "-m", "torchkiln", "predict"]
    assert argv[argv.index("--weights") + 1] == "/workspace/model/best.pth"
    assert argv[argv.index("--input") + 1] == "/workspace/data"


def test_predict_argv_passes_metrics_dir():
    """必须显式传 ``--metrics-dir``。

    ``--output`` 的语义随输入而变（单图=文件、批量=目录），从它反推
    metrics.jsonl 该写在哪是不可靠的；而外部调度正是靠那个文件里的 ``end``
    事件判终态，写错地方等于每次预测都被判失败。
    """
    argv = _argv()
    assert argv[argv.index("--metrics-dir") + 1] == "/workspace/jobs/job_1"


def test_predict_argv_output_is_subdir_of_job_dir():
    """结果图落在作业目录的子目录里，不与 metrics.jsonl 混在一起。"""
    argv = _argv()
    out = argv[argv.index("--output") + 1]
    assert out == "/workspace/jobs/job_1/predict_results"


def test_predict_argv_paths_use_posix_separators():
    """⚠️ 关键回归：不能用 ``os.path.join`` 拼 ``--output``。

    本机（平台侧）在 Windows、容器在 Linux 时，
    ``os.path.join("/workspace/jobs/job_1", "predict_results")`` 得到
    ``/workspace/jobs/job_1\\predict_results``——在宿主上看着完全正常，
    传进容器就找不到文件，而且没有任何本地信号。
    """
    argv = runner.build_predict_argv(
        _spec(), r"C:\repo\configs\yolo11-seg.yml",
        r"/workspace/jobs/job_1", "python", r"C:\repo")
    for flag in ("--output", "--metrics-dir", "--input", "--weights", "-c"):
        assert flag in argv, f"缺少 {flag}"
        value = argv[argv.index(flag) + 1]
        assert "\\" not in value, f"{flag} 的值带反斜杠: {value!r}"


def test_posix_join_strips_separators():
    assert runner._posix_join("/a/b", "c") == "/a/b/c"
    assert runner._posix_join("/a/b/", "/c/") == "/a/b/c"
    assert runner._posix_join(r"\a\b", "c") == r"\a\b/c"  # base 里的反斜杠是调用方的责任


def test_predict_argv_does_not_inject_save_model_dir():
    """预测不写权重，构造器**不该主动注入** ``Global.save_model_dir``。

    训练/评估是由服务端强制注入该键（并放在最后覆盖同名项）；预测的产物是
    ``--output`` 指定的图，与权重落盘无关，注入一个不生效的键只会让日志里
    出现一个看起来像在起作用、实际被忽略的参数。
    """
    argv = _argv()
    assert "Global.save_model_dir" not in " ".join(argv)
    assert "Global.metrics_sink" not in " ".join(argv)


def test_predict_argv_passes_user_params_through():
    """用户给的 ``params`` 原样透传——构造器只负责注入平台托管的键。"""
    argv = _argv(params={"Global.conf": 0.3})
    assert "Global.conf=0.3" in " ".join(argv)


def test_predict_argv_passes_params_as_opt():
    argv = _argv(params={"Global.conf": 0.25, "Global.iou": 0.45})
    joined = " ".join(argv)
    assert "Global.conf=0.25" in joined
    assert "Global.iou=0.45" in joined
    assert argv.count("-o") == 1


def test_predict_argv_omits_opt_when_no_params():
    argv = _argv()
    assert "-o" not in argv


def test_predict_argv_omits_weights_flag_when_absent():
    argv = runner.build_predict_argv(
        JobSpec(kind="predict", input_dir="/workspace/data"),
        "/repo/c.yml", "/out", "python", "/repo")
    assert "--weights" not in argv


def test_predict_argv_omits_input_flag_when_absent():
    argv = runner.build_predict_argv(
        JobSpec(kind="predict", weights_path="/w/b.pth"),
        "/repo/c.yml", "/out", "python", "/repo")
    assert "--input" not in argv


def test_predict_argv_serializes_list_params_as_json():
    argv = _argv(params={"Eval.something": [1, 2]})
    assert "Eval.something=[1, 2]" in " ".join(argv)


def test_predict_argv_serializes_bool_params():
    argv = _argv(params={"Global.flag": True, "Global.off": False})
    joined = " ".join(argv)
    assert "Global.flag=true" in joined
    assert "Global.off=false" in joined


# ---------------------------------------------------------------------------
# predict_yolo.py 的契约写入
# ---------------------------------------------------------------------------


def _src():
    return PREDICT_YOLO.read_text(encoding="utf-8")


def test_predict_yolo_exposes_metrics_dir_flag():
    """CLI 必须接受 ``--metrics-dir``，否则作业侧传了会被 argparse 拒绝。"""
    src = _src()
    assert '--metrics-dir' in src


def test_predict_yolo_writes_end_event():
    """必须有 ``sink.end(...)``——不写就等于每次预测都被判 ``no_end_event``。"""
    src = _src()
    assert "sink.end(" in src


def test_predict_yolo_writes_predict_event():
    src = _src()
    assert "sink.predict(" in src


def test_predict_yolo_writes_end_on_exception():
    """异常路径也要收尾写 end，否则外部服务永远等不到终态。"""
    src = _src()
    assert "except Exception as exc" in src
    idx = src.index("except Exception as exc")
    assert "sink.end(" in src[idx:idx + 400]


def test_predict_yolo_imports_sink_helpers():
    """import 必须存在，否则名字未定义。"""
    src = _src()
    assert "build_sink" in src and "sink_enabled" in src
    assert "from ptcore.metrics_sink import" in src


def test_predict_yolo_sink_dir_prefers_explicit_flag():
    """``--metrics-dir`` 优先于任何推导。"""
    src = _src()
    assert "if args.metrics_dir:" in src


def test_predict_yolo_sink_respects_kill_switch():
    """``TKILN_METRICS=0`` 要能关掉契约（与训练/评估一致）。"""
    src = _src()
    assert "sink_enabled(True)" in src


def test_predict_yolo_still_lists_images_recursively():
    """目录批量枚举不能被这次的改动破坏。"""
    src = _src()
    assert "os.walk(input_path)" in src
    assert "_list_images" in src


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


def test_predict_submit_and_list_endpoints_exist(app):
    paths = _paths(app)
    assert ("POST", "/api/v1/predict/jobs") in paths
    assert ("GET", "/api/v1/predict/jobs") in paths


def test_predict_lifecycle_endpoints_exist(app):
    paths = _paths(app)
    for method, path in [
        ("GET", "/api/v1/predict/jobs/{job_id}"),
        ("POST", "/api/v1/predict/jobs/{job_id}/cancel"),
        ("GET", "/api/v1/predict/jobs/{job_id}/artifacts"),
        ("GET", "/api/v1/predict/jobs/{job_id}/metrics"),
        ("GET", "/api/v1/predict/jobs/{job_id}/metrics/stream"),
        ("GET", "/api/v1/predict/jobs/{job_id}/logs"),
        ("GET", "/api/v1/predict/jobs/{job_id}/logs/stream"),
    ]:
        assert (method, path) in paths, f"缺少 {method} {path}"


def test_train_and_eval_endpoints_unaffected(app):
    paths = _paths(app)
    for method, path in [
        ("POST", "/api/v1/train/jobs"),
        ("POST", "/api/v1/eval/jobs"),
        ("GET", "/api/v1/eval/jobs"),
    ]:
        assert (method, path) in paths, f"缺少 {method} {path}"


def test_no_duplicate_predict_list_route(app):
    gets = [p for m, p in _paths(app) if m == "GET" and p == "/api/v1/predict/jobs"]
    assert len(gets) == 1, f"/api/v1/predict/jobs 重复注册: {gets}"


def test_alias_registration_is_shared_between_eval_and_predict(app):
    """eval 与 predict 的别名必须来自**同一个**表。

    分成两张表的话，加一种新端点只改了一张 —— 症状是「A 通路能看日志、
    B 通路不能」，非常难查。
    """
    import ast
    import pathlib

    main_py = pathlib.Path(r"D:\TorchKiln\service\main.py")
    src = main_py.read_text(encoding="utf-8")
    assert "_ALIAS_SUFFIXES = (" in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and "suffix" in ast.unparse(node.target):
            body = ast.unparse(node.body)
            # ast.unparse 会把字符串规范化成单引号，所以这里不能用双引号去匹配
            assert "for prefix in ('eval', 'predict')" in body, (
                "别名注册循环应同时注册 eval 与 predict 两个前缀")
            return
    raise AssertionError("找不到别名注册循环")


def test_metrics_contract_shape_documented():
    """``predict`` 事件报的是张数与耗时，不该有精度指标。

    这条是**语义**约束：预测没有 ground truth，任何 mAP 都是编出来的。
    """
    assert "images_total" in _src()
    assert "images_done" in _src()
    assert "elapsed_sec" in _src()


def test_metrics_jsonl_is_written_next_to_results_dir():
    """契约文件与结果目录同属作业目录，宿主可直接读到。"""
    argv = _argv()
    out = argv[argv.index("--output") + 1]
    mdir = argv[argv.index("--metrics-dir") + 1]
    assert os.path.dirname(out) == mdir
