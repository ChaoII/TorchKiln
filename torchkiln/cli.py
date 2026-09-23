"""Unified task/mode CLI.

    tkiln <mode> [args...]              # task 由配置决定(推荐)
    tkiln <task> <mode> [args...]       # 显式写 task(等价 + 一致性校验)
    python -m torchkiln <mode> ...     # 免安装等价写法

    tkiln train   -c configs/yolo/yolov8_graph.yml -o Global.epoch_num=100
    tkiln val     -c configs/yolo/yolov8-obb_graph.yml --weights output/x/best_accuracy.pth
    tkiln check   -c configs/attr/vehicle_attribute.yml
    tkiln export  -c configs/yolo/yolov8-pose_graph.yml --weights ... --onnx
    tkiln predict -c configs/yolo/yolov8-pose_graph.yml --weights ... --input imgs
    tkiln data list | tkiln data get <name>          # 数据集(ModelScope)下载/检查
  tkiln detect train -c configs/yolo/yolov8_graph.yml        # 显式 task

``mode ∈ {train, val, export, predict, check}``。除 ``<mode>``(与可选 ``<task>``)之外,
其余参数**原样透传**给 ``tools/{train,eval,export}.py`` / ``tools/infer/predict_{yolo,det,rec}.py``。
"""
from __future__ import absolute_import

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CLI 命名空间别名(显式写法与 predict 路由用)-> 规范任务名
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
    "plate_det": "plate_det",
    "plate-det": "plate_det",
    "plate_rec": "plate_rec",
    "plate-rec": "plate_rec",
    "attribute": "attribute",
    "attr": "attribute",
    "pose_action": "pose_action",
    "pose-action": "pose_action",
    "action": "pose_action",
    "video_cls": "video_cls",
    "video-cls": "video_cls",
    "video": "video_cls",
    "ocr": "ocr",
    "ocr_det": "ocr_det",
    "ocr-det": "ocr_det",
    "ocr_rec": "ocr_rec",
    "ocr-rec": "ocr_rec",
}

MODES = ("train", "val", "export", "predict", "check")

# 配置里的 Architecture.task -> CLI 命名空间 / 模型族
CONFIG_TASK_ALIAS = {"det": "ocr_det", "rec": "ocr_rec"}
FAMILY_OF = {
    "detect": "yolo",
    "segment": "yolo",
    "obb": "yolo",
    "pose": "yolo",
    "classify": "yolo",
    "semantic": "yolo",
    "depth": "yolo",
    "plate_det": "yolo",
    "plate_rec": "yolo",
    "attribute": "yolo",
    "pose_action": "yolo",
    "video_cls": "yolo",
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
    "ocr_det": "tools/infer/predict_det.py",
    "ocr_rec": "tools/infer/predict_rec.py",
}

USAGE = """用法: tkiln [task] <mode> [args...]        (task 可省略,由配置 Architecture.task 决定)

  task(可选): {tasks}
  mode      : {modes}

推荐(省略 task):
  tkiln train   -c configs/yolo/yolov8_graph.yml -o Global.epoch_num=100
  tkiln val     -c configs/yolo/yolov8-obb_graph.yml --weights output/x/best_accuracy.pth
  tkiln check   -c configs/attr/vehicle_attribute.yml
  tkiln export  -c configs/yolo/yolov8-pose_graph.yml --weights ... --onnx
  tkiln predict -c configs/yolo/yolov8-pose_graph.yml --weights ... --input imgs

显式写 task(等价,并做一致性校验):
  tkiln detect train -c configs/yolo/yolov8_graph.yml

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
        for key in ("plate_rec", "plate_det", "attribute", "pose_action", "video_cls",
                    "detect", "segment", "obb", "pose", "classify", "semantic", "depth",
                    "det", "rec", "cls"):
            if key in raw:
                return key, cfg
        return (dirname if dirname in ("det", "rec", "cls") else None), cfg
    except Exception:
        return (dirname if dirname in ("det", "rec", "cls") else None), cfg


def _canonical(cfg_task):
    return CONFIG_TASK_ALIAS.get(cfg_task, cfg_task)


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
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv[0].lower() in ("data", "dataset", "datasets"):
        from torchkiln.datasets import data_main

        return data_main(argv[1:])

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
