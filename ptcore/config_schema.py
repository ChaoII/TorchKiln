"""配置自描述：把 ``configs/**.yml`` 变成机器可读的**模型清单**与**超参 schema**。

目的：让外部平台（AIStation 等）**不硬编码任何超参映射表**。
以前每接一个框架就要维护一张 ``_ULTRALYTICS_HP`` 手写表（加参数 → 改前后端 → 重新发布）；
现在"加模型 = 丢一个 YAML"，平台侧 ``list`` 一下下拉框就有，``schema`` 一下表单就有。

对外两个能力：

  :func:`list_models`   扫描 configs/，每个 YAML 一行摘要（模型名/任务/主指标/…）
  :func:`describe_config`  单个配置的完整超参 schema（类型/默认值/取值范围/控件类型/分组）

设计取舍：
  - **所有叶子都是可覆盖的**（``-o Key.Sub=value`` 本来就支持任意路径），所以 schema
    不做"白名单"，而是全量给出 + ``managed`` 标记由平台注入的键；
  - 类型从 YAML 值的 Python 类型直接推断（``bool/int/float/str/list/None``）；
  - 中文 label 走「内置常见键表 + 配置内 ``_ui`` 段覆盖」——不用一次性给 123 个配置
    标完 3000 个键，团队可以按需在自己的 YAML 里补；
  - ``min/max`` 与控件类型（switch/slider/number）走「叶子名」维度的常见键表。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from .config import load_config, parse_args_to_config
from .metrics_sink import clean_value

__all__ = [
    "REPO_ROOT",
    "list_models",
    "describe_config",
    "resolve_config",
    "MODEL_GROUPS",
]

#: 仓库根目录（ptcore/config_schema.py -> 上两级）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 配置扫描根
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")

#: 前端表单分组顺序（不在表里的 section 按字母序追加在后面）
MODEL_GROUPS = [
    "Global",
    "Optimizer",
    "Architecture",
    "Loss",
    "Metric",
    "Train",
    "Eval",
    "PostProcess",
]

#: 由**平台**注入、不该出现在用户表单里的键（输出目录、续训权重等）
MANAGED_KEYS = {
    "Global.save_model_dir",
    "Global.output",
    "Global.checkpoints",
}

#: 叶子名 -> 中文 label（内置常见项；配置内 `_ui` 段可覆盖）
LEAF_LABELS = {
    "model_name": "模型名称",
    "model_type": "模型类型",
    "epoch_num": "训练轮数",
    "max_epochs": "最大轮数",
    "batch_size": "批大小",
    "batch_size_per_card": "单卡批大小",
    "learning_rate": "学习率",
    "lr": "学习率",
    "momentum": "动量",
    "weight_decay": "权重衰减",
    "seed": "随机种子",
    "num_workers": "数据加载线程",
    "use_ema": "启用 EMA",
    "ema_decay": "EMA 衰减率",
    "amp": "混合精度(AMP)",
    "distributed": "分布式训练",
    "device": "运行设备",
    "pretrained_model": "预训练权重",
    "print_batch_step": "日志打印步长",
    "save_epoch_step": "存权重间隔(轮)",
    "eval_epoch_step": "评估间隔(轮)",
    "eval_batch_step": "按步评估间隔",
    "patience": "早停耐心轮数",
    "show_eval_progress": "评估进度条",
    "main_indicator": "主指标",
    "main_indicator_mode": "主指标方向",
    "num_classes": "类别数",
    "scale": "模型规格",
    "warmup_epoch": "预热轮数",
    "warmup_ratio": "预热比例",
    "freeze": "冻结骨干层数",
    "in_channels": "输入通道",
    "dropout": "Dropout",
    "ignore_index": "忽略标签",
    "use_color_jitter": "颜色抖动增强",
    "cudnn_deterministic": "cuDNN 确定性",
}

#: 叶子名 -> (min, max, widget)。widget: switch/slider/number
LEAF_RANGES = {
    "epoch_num": (1, 2000, "slider"),
    "max_epochs": (1, 2000, "slider"),
    "batch_size": (1, 512, "number"),
    "batch_size_per_card": (1, 256, "number"),
    "learning_rate": (0.0, 1.0, "number"),
    "lr": (0.0, 1.0, "number"),
    "momentum": (0.0, 0.999, "number"),
    "weight_decay": (0.0, 1.0, "number"),
    "num_workers": (0, 32, "number"),
    "seed": (0, 2147483647, "number"),
    "patience": (0, 500, "number"),
    "print_batch_step": (1, 5000, "number"),
    "save_epoch_step": (1, 100, "number"),
    "eval_epoch_step": (1, 100, "number"),
    "ema_decay": (0.0, 1.0, "number"),
    "warmup_epoch": (0, 50, "number"),
    "warmup_ratio": (0.0, 1.0, "number"),
    "num_classes": (1, 10000, "number"),
    "dropout": (0.0, 1.0, "number"),
}

#: 值为 True/False 的叶子 -> 强制 switch
_SWITCH_LEAVES = {
    "use_ema",
    "amp",
    "distributed",
    "show_eval_progress",
    "use_color_jitter",
    "cudnn_deterministic",
    "use_pretrain",
    "drop_block",
    "multi_scale",
    "save_optimizer",
}

#: 值像路径/权重/数据的叶子 -> 强制 text（平台侧渲染成路径选择器）
_TEXT_LEAVES = {
    "pretrained_model",
    "data_dir",
    "ann_path",
    "dict_path",
    "yaml_file",
    "dataset_type",
    "weight_decay_name",
    "logit_adjust",
    "resume_path",
}


def _rel(path):
    try:
        return os.path.relpath(path, REPO_ROOT).replace("\\", "/")
    except ValueError:
        return path


def _flatten(node, prefix=""):
    """递归展开配置为 ``(点分键, 叶子值)``；空 dict 当叶子保留结构。"""
    if isinstance(node, dict):
        for k, v in node.items():
            key = "{}.{}".format(prefix, k) if prefix else str(k)
            if isinstance(v, dict) and v:
                for item in _flatten(v, key):
                    yield item
            else:
                yield key, v
    else:
        yield prefix, node


def _type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "str"
    return "unknown"


def _label_of(dotted):
    leaf = dotted.rsplit(".", 1)[-1]
    return LEAF_LABELS.get(leaf, dotted)


def describe_param(dotted, value, ui=None):
    """推断单个超参的类型/控件/范围/分组。"""
    leaf = dotted.rsplit(".", 1)[-1]
    info = {
        "label": LEAF_LABELS.get(leaf, dotted),
        "type": _type_name(value),
        "default": clean_value(value),
        "group": dotted.split(".")[0],
        "key": dotted,
    }
    if isinstance(value, bool) or leaf in _SWITCH_LEAVES:
        info["widget"] = "switch"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        info["widget"] = "number"
    elif isinstance(value, list):
        info["widget"] = "list"
    elif value is None:
        # 多数是路径/权重：None = "未设置"，允许填字符串
        info["widget"] = "path" if leaf in _TEXT_LEAVES else "text"
        info["nullable"] = True
    else:
        info["widget"] = "path" if leaf in _TEXT_LEAVES else "text"

    rng = LEAF_RANGES.get(leaf)
    if rng is not None:
        lo, hi, _default_widget = rng
        if info["type"] in ("int", "float"):
            info["min"], info["max"] = lo, hi
        if _default_widget == "slider" and info["widget"] == "number":
            info["widget"] = "slider"

    # 配置内 `_ui` 段可覆盖任意字段（label/min/max/options/widget…）
    if isinstance(ui, dict):
        info.update(clean_value(ui))
    return info


def _ui_overrides(config):
    ui = config.get("_ui")
    return ui if isinstance(ui, dict) else {}


def describe_config(config_path, overrides=None, include_managed=False):
    """把一个配置 YAML 描述成完整 schema。

    Args:
        config_path: 配置路径（相对仓库根或绝对路径）。
        overrides: 可选的 ``-o k=v`` 列表，会先合并再描述。
        include_managed: 是否把由平台注入的键也列进 ``params``（默认否，单列 ``managed``）。

    Returns:
        ``{model_name, task, family, ..., params: {点分键: 描述}, managed: [...]}``
    """
    path = config_path
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, config_path)
    config = parse_args_to_config(path, overrides) if overrides else load_config(path)

    gcfg = config.get("Global") or {}
    arch = config.get("Architecture") or {}
    metric = config.get("Metric") or {}
    train = (config.get("Train") or {}).get("dataset") or {}
    eval_ds = ((config.get("Eval") or {}).get("dataset")) or {}
    ui = _ui_overrides(config)

    params = {}
    managed = []
    for dotted, value in _flatten(config):
        top = dotted.split(".")[0]
        if top in ("_ui", "_overrides"):
            continue
        if dotted in MANAGED_KEYS or top in MANAGED_KEYS:
            managed.append(dotted)
            if not include_managed:
                continue
        params[dotted] = describe_param(dotted, value, ui.get(dotted))

    groups = [g for g in MODEL_GROUPS if any(
        p["group"] == g for p in params.values()
    )]
    groups += sorted({
        p["group"] for p in params.values() if p["group"] not in MODEL_GROUPS
    })

    mode = str(gcfg.get("main_indicator_mode", "max") or "max").lower()
    return {
        "schema_version": 1,
        "model_name": gcfg.get("model_name"),
        "config_path": _rel(path),
        "task": arch.get("task"),
        "model_family": arch.get("model_family"),
        "algorithm": arch.get("algorithm"),
        "model_type": arch.get("model_type"),
        "scale": arch.get("scale"),
        "main_indicator": metric.get("main_indicator", "hmean"),
        "main_indicator_mode": mode,
        "groups": groups,
        "overridable_keys": sorted(params.keys()),
        "managed_keys": sorted(set(managed)),
        "data": {
            "train_name": train.get("name"),
            "train_data_dir": train.get("data_dir"),
            "train_label_file_list": clean_value(train.get("label_file_list")),
            "eval_name": eval_ds.get("name"),
            "eval_data_dir": eval_ds.get("data_dir"),
            "eval_label_file_list": clean_value(eval_ds.get("label_file_list")),
        },
        "defaults": {
            "epoch_num": gcfg.get("epoch_num"),
            "batch_size_per_card": (
                (config.get("Train") or {}).get("loader", {}) or {}
            ).get("batch_size_per_card"),
            "learning_rate": _first_lr(config),
            "pretrained_model": gcfg.get("pretrained_model"),
            "seed": gcfg.get("seed"),
            "use_ema": gcfg.get("use_ema"),
            "amp": gcfg.get("amp"),
        },
        "params": params,
    }


def _first_lr(config):
    """学习率可能挂在 ``Optimizer.lr.learning_rate`` 或 ``Global.lr``。"""
    lr = ((config.get("Optimizer") or {}).get("lr")) or {}
    if isinstance(lr, dict):
        return lr.get("learning_rate")
    if lr is not None:
        return lr
    return (config.get("Global") or {}).get("lr")


def _summary(path, config):
    arch = config.get("Architecture") or {}
    gcfg = config.get("Global") or {}
    metric = config.get("Metric") or {}
    mode = str(gcfg.get("main_indicator_mode", "max") or "max").lower()
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    return {
        "model_name": gcfg.get("model_name"),
        "config_path": _rel(path),
        "task": arch.get("task"),
        "model_family": arch.get("model_family"),
        "algorithm": arch.get("algorithm"),
        "model_type": arch.get("model_type"),
        "scale": arch.get("scale"),
        "main_indicator": metric.get("main_indicator", "hmean"),
        "main_indicator_mode": mode,
        "epoch_num": gcfg.get("epoch_num"),
        "pretrained_model": gcfg.get("pretrained_model"),
        "has_pretrained": bool(gcfg.get("pretrained_model")),
        "config_bytes": size,
    }


def iter_config_files(config_root=None):
    """遍历 ``configs/`` 下所有 YAML（跳过 ``_``/``.`` 开头的目录，如 ``_parity/``）。"""
    base = config_root or CONFIG_ROOT
    if not os.path.isdir(base):
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(("_", "."))
        )
        for fn in sorted(filenames):
            if fn.endswith((".yml", ".yaml")):
                yield os.path.join(dirpath, fn)


def list_models(config_root=None, include_broken=True):
    """扫描全部配置，每个一行摘要，按 (family, task, model_name) 排序。"""
    rows = []
    for path in iter_config_files(config_root):
        try:
            config = load_config(path)
        except Exception as exc:  # noqa: BLE001
            if include_broken:
                rows.append({
                    "model_name": None,
                    "config_path": _rel(path),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                })
            continue
        if not isinstance(config, dict):
            if include_broken:
                rows.append({
                    "model_name": None,
                    "config_path": _rel(path),
                    "error": "not a mapping",
                })
            continue
        rows.append(_summary(path, config))
    rows.sort(key=lambda r: (
        r.get("model_family") or "",
        r.get("task") or "",
        r.get("model_name") or "",
    ))
    return rows


def resolve_config(name_or_path, config_root=None):
    """按「路径 / 模型名 / 相对路径片段」解析出唯一配置文件。

    优先级：直接是文件 -> ``<root>/<name>`` -> ``configs/**/<name>.yml`` ->
    ``Global.model_name == name`` -> 路径片段模糊匹配（取唯一命中，多命中报错）。

    Returns:
        ``(绝对路径, 命中方式)``；找不到抛 ``FileNotFoundError``。
    """
    base = config_root or CONFIG_ROOT
    cand = name_or_path
    if os.path.isabs(cand) and os.path.isfile(cand):
        return cand, "abs_path"
    for p in (cand, os.path.join(REPO_ROOT, cand), os.path.join(base, cand)):
        if os.path.isfile(p):
            return os.path.abspath(p), "path"

    files = list(iter_config_files(base))
    stem = os.path.splitext(os.path.basename(cand))[0]
    # 1) 文件名精确匹配
    for p in files:
        if os.path.splitext(os.path.basename(p))[0] == stem:
            return p, "filename"
    # 2) Global.model_name 精确匹配
    by_model = []
    for p in files:
        try:
            cfg = load_config(p)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(cfg, dict) and (cfg.get("Global") or {}).get("model_name") == stem:
            by_model.append(p)
    if len(by_model) == 1:
        return by_model[0], "model_name"
    if len(by_model) > 1:
        raise ValueError(
            "model_name {!r} matches multiple configs: {}".format(
                stem, ", ".join(_rel(p) for p in by_model)
            )
        )
    # 3) 路径片段匹配
    frag = [s for s in cand.replace("\\", "/").split("/") if s not in ("", ".")]
    hits = [p for p in files if all(s.lower() in _rel(p).lower() for s in frag)]
    if len(hits) == 1:
        return hits[0], "fragment"
    if len(hits) > 1:
        raise ValueError(
            "{!r} is ambiguous, candidates: {}".format(
                cand, ", ".join(_rel(p) for p in hits[:10])
            )
        )
    raise FileNotFoundError(
        "config not found: {!r} (searched {} YAMLs under {})".format(
            cand, len(files), base
        )
    )
