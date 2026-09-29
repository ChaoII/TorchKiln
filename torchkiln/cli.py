"""Unified task/mode CLI.

    tkiln <mode> [args...]              # task 由配置决定(推荐)
    tkiln <task> <mode> [args...]       # 显式写 task(等价 + 一致性校验)
    python -m torchkiln <mode> ...     # 免安装等价写法

    tkiln train   -c configs/yolo/yolov8-det.yml -o Global.epoch_num=100
    tkiln val     -c configs/yolo/yolov8-obb.yml --weights output/x/best_accuracy.pth
    tkiln check   -c configs/attr/vehicle_attribute.yml
    tkiln export  -c configs/yolo/yolov8-pose.yml --weights ... --onnx
    tkiln predict -c configs/yolo/yolov8-pose.yml --weights ... --input imgs
    tkiln data list | tkiln data get <name>          # 数据集(ModelScope)下载/检查
  tkiln detect train -c configs/yolo/yolov8-det.yml        # 显式 task

``mode ∈ {train, val, export, predict, check}``。除 ``<mode>``(与可选 ``<task>``)之外,
其余参数**原样透传**给 ``tools/{train,eval,export}.py`` / ``tools/infer/predict_{yolo,det,rec}.py``。
"""
from __future__ import absolute_import

import json
import os
import sys

from ptcore.config import flatten_opts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CLI 命名空间别名(显式写法与 predict 路由用)-> 规范任务名
# 变点/漂移检测（纯统计工具，无训练/无权重 -> 不注册为 task）
TOOL_SCRIPTS = {"changepoint": "tools/changepoint.py",
                "cp": "tools/changepoint.py"}

#: 不需要 task/mode 的「平台元信息」命令（外部平台靠它们自动生成模型下拉与参数表单）
META_COMMANDS = ("list", "schema")

TASK_ALIASES = {
    "detect": "detect",
    "det": "detect",
    "segment": "segment",
    "seg": "segment",
    "obb": "obb",
    "pose": "pose",
    "classify": "classify",
    "cls": "classify",
    "semantic": "semantic",
    "sem": "semantic",
    "depth": "depth",
    "lane_seg": "lane_seg",
    "lane-seg": "lane_seg",
    "lane_row": "lane_row",
    "lane-row": "lane_row",
    "lane": "lane_seg",
    "pc_seg": "pc_seg",
    "pc-seg": "pc_seg",
    "pcseg": "pc_seg",
    "det3d": "det3d",
    "det-3d": "det3d",
    "3d": "det3d",
    "lane_bev": "lane_bev",
    "lane-bev": "lane_bev",
    "lanebev": "lane_bev",
    "plate_det": "plate_det",
    "plate-det": "plate_det",
    "plate_rec": "plate_rec",
    "plate-rec": "plate_rec",
    "kokoro_tts": "kokoro_tts",
    "panns_cls": "panns_cls",
    "panns-cls": "panns_cls",
    "audio_cls": "panns_cls",
    "kokoro-tts": "kokoro_tts",
    "tts": "kokoro_tts",
    "attribute": "attribute",
    "attr": "attribute",
    "pose_action": "pose_action",
    "pose-action": "pose_action",
    "action": "pose_action",
    "video_cls": "video_cls",
    "video-cls": "video_cls",
    "video": "video_cls",
    "ts_forecast": "ts_forecast",
    "ts-forecast": "ts_forecast",
    "ts": "ts_forecast",
    "ts_anomaly": "ts_anomaly",
    "ts_classify": "ts_classify",
    "ts_embed": "ts_embed",
    "ts_rul": "ts_rul",
    "ts-rul": "ts_rul",
    "rul": "ts_rul",
    "ts-embed": "ts_embed",
    "ts_repr": "ts_embed",
    "repr": "ts_embed",
    "ts-classify": "ts_classify",
    "ts_cls": "ts_classify",
    "ts-cls": "ts_classify",
    "ts-anomaly": "ts_anomaly",
    "anomaly": "ts_anomaly",
    "ad": "ts_anomaly",
    "ocr": "ocr",
    "ocr_det": "ocr_det",
    "ocr-det": "ocr_det",
    "ocr_rec": "ocr_rec",
    "ocr-rec": "ocr_rec",
    "ocr_e2e": "ocr_e2e",
    "ocr-e2e": "ocr_e2e",
    "e2e": "ocr_e2e",
}

MODES = ("train", "val", "export", "predict", "check")

# 配置里的 Architecture.task -> CLI 命名空间 / 模型族
CONFIG_TASK_ALIAS = {"det": "ocr_det", "rec": "ocr_rec", "e2e": "ocr_e2e"}
FAMILY_OF = {
    "detect": "yolo",
    "segment": "yolo",
    "obb": "yolo",
    "pose": "yolo",
    "classify": "yolo",
    "semantic": "yolo",
    "depth": "yolo",
    "lane_seg": "yolo",
    "lane_row": "yolo",
    "pc_seg": "pc",
    "det3d": "pc",
    "lane_bev": "lane_bev",
    "plate_det": "yolo",
    "plate_rec": "yolo",
    "kokoro_tts": "kokoro_tts",
    "attribute": "yolo",
    "pose_action": "yolo",
    "video_cls": "yolo",
    "ts_forecast": "ts",
    "ts_anomaly": "ts",
    "ts_classify": "ts",
    "ts_embed": "ts",
    "ts_rul": "ts",
    "ocr": "ocr_det",
    "ocr_det": "ocr_det",
    "ocr_rec": "ocr_rec",
}

SCRIPT = {
    "train": "tools/train.py",
    "val": "tools/eval.py",
    "export": "tools/export.py",
}

PREDICT_SCRIPT = {
    "yolo": "tools/infer/predict_yolo.py",
    "pc": "tools/infer/predict_pc.py",
    "ocr_det": "tools/infer/predict_det.py",
    "ocr_rec": "tools/infer/predict_rec.py",
}

USAGE = """用法: tkiln [task] <mode> [args...]        (task 可省略,由配置 Architecture.task 决定)

  task(可选): {tasks}
  mode      : {modes}

推荐(省略 task):
  tkiln train   -c configs/yolo/yolov8-det.yml -o Global.epoch_num=100
  tkiln val     -c configs/yolo/yolov8-obb.yml --weights output/x/best_accuracy.pth
  tkiln check   -c configs/attr/vehicle_attribute.yml
  tkiln export  -c configs/yolo/yolov8-pose.yml --weights ... --onnx
  tkiln predict -c configs/yolo/yolov8-pose.yml --weights ... --input imgs

显式写 task(等价,并做一致性校验):
  tkiln detect train -c configs/yolo/yolov8-det.yml

平台元信息(供外部平台自动生成模型列表/参数表单,不需要 torch 与 GPU):
  tkiln list [--json] [--task detect] [--family yolo] [子目录过滤]
  tkiln schema <模型名或配置路径> [--json] [-o Key.Sub=value ...]

其余参数与 tools/*.py 完全一致(-c 配置、-o 覆盖、--weights ...)。
""".format(tasks=" | ".join(sorted(set(TASK_ALIASES))), modes=" | ".join(MODES))


def _load_script(rel_path, name):
    import importlib.util

    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config_task(argv):
    """取 ``-c`` 配置并读它的 ``Architecture.task``(返回 (task, cfg_raw))。"""
    cfg = None
    for i, a in enumerate(argv):
        if a in ("-c", "--config", "--cfg") and i + 1 < len(argv):
            cfg = argv[i + 1]
            break
    if not cfg:
        return None, None
    path = cfg if os.path.isabs(cfg) else os.path.join(ROOT, cfg)
    if not os.path.isfile(path):
        return None, cfg
    # 任务推断:优先 Configuration 里的 task/name,再退回配置所在目录(det/rec/cls...)
    dirname = os.path.basename(os.path.dirname(os.path.abspath(path))).lower()
    try:
        from torchkiln.ocr.utils.config import load_config

        arch = load_config(path).get("Architecture") or {}
        raw = str(arch.get("task") or arch.get("name") or "").lower()
        for key in ("ts_rul", "ts_embed", "ts_classify", "ts_anomaly", "ts_forecast", "kokoro_tts", "panns_cls", "plate_rec", "plate_det", "attribute", "pose_action", "video_cls",
                    "lane_bev", "lane_seg", "lane_row", "pc_seg", "det3d", "detect", "segment", "obb", "pose",
                    "classify", "semantic", "depth", "e2e", "det", "rec", "cls"):
            if key in raw:
                return key, cfg
        return (dirname if dirname in ("det", "rec", "cls", "pc") else None), cfg
    except Exception:
        return (dirname if dirname in ("det", "rec", "cls", "pc") else None), cfg


def _canonical(cfg_task):
    return CONFIG_TASK_ALIAS.get(cfg_task, cfg_task)


# ------------------------------------------------------------------ 元信息命令
def _opt_value(argv, *names):
    """取 ``--flag value`` 形式的值（也接受 ``--flag=value``）。"""
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def _cmd_list(argv):
    """``tkiln list``：扫描 configs/ 输出模型清单（表格或 JSON）。

    外部平台用它生成"模型下拉框"——加模型只要丢一个 YAML，无需改平台代码。
    """
    import argparse

    from ptcore.config_schema import list_models

    ap = argparse.ArgumentParser(
        prog="tkiln list", description="列出全部可用模型配置")
    ap.add_argument("--json", action="store_true", help="输出 JSON（给程序消费）")
    ap.add_argument("--task", help="按 Architecture.task 过滤")
    ap.add_argument("--family", "--model-family", dest="family",
                    help="按 Architecture.model_family 过滤")
    ap.add_argument("--name", help="按 Global.model_name 子串过滤")
    ap.add_argument("--no-header", action="store_true", help="表格不打印表头")
    ap.add_argument("subdir", nargs="?", help="只列该子目录，如 yolo / ocr / pc")
    args = ap.parse_args(argv)

    rows = list_models()
    if args.subdir:
        rows = [r for r in rows if args.subdir in (r.get("config_path") or "")]
    if args.task:
        rows = [r for r in rows if (r.get("task") or "") == args.task]
    if args.family:
        rows = [r for r in rows if (r.get("model_family") or "") == args.family]
    if args.name:
        low = args.name.lower()
        rows = [r for r in rows
                if low in (r.get("model_name") or "").lower()]

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    if not rows:
        print("没有匹配的模型配置。")
        return 0
    bad = [r for r in rows if r.get("error")]
    good = [r for r in rows if not r.get("error")]
    header = ("{:<26} {:<14} {:<10} {:<16} {:>5} {:>6}  {}".format(
        "MODEL", "TASK", "FAMILY", "MAIN_INDICATOR", "EPOCH", "PT", "CONFIG"))
    if not args.no_header:
        print(header)
        print("-" * len(header))
    for r in good:
        print("{:<26} {:<14} {:<10} {:<16} {:>5} {:>6}  {}".format(
            str(r.get("model_name") or "-")[:26],
            str(r.get("task") or "-")[:14],
            str(r.get("model_family") or "-")[:10],
            str(r.get("main_indicator") or "-")[:16],
            str(r.get("epoch_num") if r.get("epoch_num") is not None else "-"),
            "Y" if r.get("has_pretrained") else "-",
            r.get("config_path"),
        ))
    for r in bad:
        print("! {}  <解析失败: {}>".format(r.get("config_path"), r.get("error")))
    print("\n共 {} 个配置（可用 {} 个）。".format(len(rows), len(good)))
    return 0


def _cmd_schema(argv):
    """``tkiln schema <模型名|配置路径>``：输出超参 JSON Schema（表格或 JSON）。

    外部平台用它生成"参数表单"——全部可覆盖键都在 ``params`` 里，类型/默认值/
    取值范围/控件类型/分组都带好，不再需要平台侧维护手写映射表。
    """
    import argparse

    from ptcore.config_schema import MODEL_GROUPS, describe_config, resolve_config

    ap = argparse.ArgumentParser(
        prog="tkiln schema", description="输出某配置的超参 schema")
    ap.add_argument("target", nargs="?", help="模型名或配置路径")
    ap.add_argument("--json", action="store_true", help="输出 JSON（给程序消费）")
    ap.add_argument("-o", "--opt", nargs="*", action="append", default=None,
                    help="先应用覆盖再看 schema（预览最终参数）")
    args = ap.parse_args(argv)

    if not args.target:
        print("用法: tkiln schema <模型名或配置路径> [--json] [-o Key.Sub=value ...]")
        print("提示: tkiln list  可先看有哪些模型。")
        return 2
    try:
        path, how = resolve_config(args.target)
    except (FileNotFoundError, ValueError) as exc:
        print("解析失败: {}".format(exc))
        return 2
    overrides = flatten_opts(args.opt)
    try:
        info = describe_config(path, overrides=overrides or None)
    except Exception as exc:  # noqa: BLE001
        print("读取配置失败: {}: {}".format(type(exc).__name__, exc))
        return 2

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    print("model      : {}".format(info.get("model_name")))
    print("config     : {}  (匹配方式: {})".format(info.get("config_path"), how))
    print("task/family: {} / {}".format(info.get("task"), info.get("model_family")))
    print("algorithm  : {}   scale: {}".format(
        info.get("algorithm"), info.get("scale")))
    print("main       : {} ({})".format(
        info.get("main_indicator"), info.get("main_indicator_mode")))
    if overrides:
        print("overrides  : {}".format(", ".join(overrides)))
    print("data       : train={}  val={}".format(
        info["data"].get("train_name"), info["data"].get("eval_name")))
    print("-" * 72)
    for group in info.get("groups") or MODEL_GROUPS:
        items = [(k, v) for k, v in info["params"].items() if v.get("group") == group]
        if not items:
            continue
        print("[{}]".format(group))
        for key, v in items:
            extra = []
            if v.get("min") is not None and v.get("max") is not None:
                extra.append("range={}..{}".format(v["min"], v["max"]))
            extra.append("widget={}".format(v.get("widget")))
            print("  {:<48} {:<7} default={:<20} {}".format(
                key, v.get("type"),
                json.dumps(v.get("default"), ensure_ascii=False), "  ".join(extra)))
    print("-" * 72)
    print("可覆盖参数 {} 个；平台注入: {}".format(
        len(info["params"]), ", ".join(info.get("managed_keys") or []) or "无"))
    return 0


def _check(argv):
    """``check`` 模式:构建模型/损失/指标/数据并打印摘要(不训练)。"""
    from ptcore.factory import build_trainer
    from torchkiln.ocr.utils.config import load_config

    _, cfg_raw = _config_task(argv)
    if cfg_raw is None:
        print("check 模式需要 -c <config>")
        return 2
    path = cfg_raw if os.path.isabs(cfg_raw) else os.path.join(ROOT, cfg_raw)
    config = load_config(path)
    config.setdefault("Global", {})["pretrained_model"] = None
    trainer = build_trainer(config, dump_config=False)
    arch = config.get("Architecture") or {}
    print("=" * 72)
    print("config      : {}".format(cfg_raw))
    print("model_family: {}".format(arch.get("model_family")))
    print("task        : {}".format(arch.get("task")))
    print("model       : {}".format(type(trainer.model).__name__))
    print("loss        : {}".format(type(trainer.loss).__name__))
    print("metric      : {}".format(type(trainer.metric).__name__))
    print("postprocess : {}".format(type(trainer.post_process).__name__))
    for line in trainer.task.summary_lines(
        config, config.get("Global", {}), trainer.post_process
    ) or []:
        print("summary     : {}".format(line))
    print("=" * 72)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # 纯统计工具（无 task）：`tkiln changepoint ...`
    if argv and argv[0].lower() in TOOL_SCRIPTS:
        import runpy
        script = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), TOOL_SCRIPTS[argv[0].lower()])
        sys.argv = [script] + list(argv[1:])
        runpy.run_path(script, run_name="__main__")
        return 0
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv[0].lower() in ("data", "dataset", "datasets"):
        from torchkiln.datasets import data_main

        return data_main(argv[1:])

    # 平台元信息命令：不需要 task/mode，也不 import torch（可在无 GPU 环境跑）
    if argv[0].lower() in META_COMMANDS:
        cmd = argv[0].lower()
        if cmd == "list":
            return _cmd_list(argv[1:])
        return _cmd_schema(argv[1:])

    explicit = (
        len(argv) >= 2
        and argv[0].lower() in TASK_ALIASES
        and argv[1].lower() in MODES
    )
    if explicit:
        task, mode, rest = TASK_ALIASES[argv[0].lower()], argv[1].lower(), argv[2:]
    else:
        mode = argv[0].lower()
        if mode not in MODES:
            print(
                "未知 mode: {!r}(显式写法请用 <task> <mode>)\n".format(argv[0]) + USAGE
            )
            return 2
        rest = argv[1:]
        cfg_task, _ = _config_task(rest)
        if not cfg_task:
            print("省略 <task> 时必须用 -c <config> 指定配置(任务由 Architecture.task 决定)")
            return 2
        task = _canonical(cfg_task)
        if task not in TASK_ALIASES:
            print("配置里的 task={!r} 未在 CLI 命名空间中注册".format(cfg_task))
            return 2
        print("[task] {:<10} (来自配置 Architecture.task={!r})".format(task, cfg_task))

    if explicit:
        cfg_task, _ = _config_task(rest)
        if cfg_task and cfg_task != task and not (
            task == "ocr" and cfg_task in CONFIG_TASK_ALIAS
        ):
            print(
                "提示:命令行 task={!r} 与配置里的 Architecture.task={!r} 不一致,"
                "以配置为准(命令行只做命名空间/校验)。".format(task, cfg_task)
            )

    if mode == "check":
        return _check(rest)

    if mode == "predict":
        script = PREDICT_SCRIPT.get(FAMILY_OF.get(task, "yolo"), PREDICT_SCRIPT["yolo"])
        mod = _load_script(script, "predict_" + os.path.basename(script).replace(".py", ""))
        sys.argv = [script] + rest
        mod.main()
        return 0

    mod = _load_script(SCRIPT[mode], "tkiln_" + mode)
    sys.argv = [SCRIPT[mode]] + rest
    mod.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
