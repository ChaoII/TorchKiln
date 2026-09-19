from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import difflib
import logging

import yaml

__all__ = ["load_config", "merge_config", "parse_args_to_config", "flatten_opts"]


def flatten_opts(opt):
    """Accept both ``-o a=1 b=2`` and repeated ``-o a=1 -o b=2``.

    argparse collects these respectively as ``[['a=1', 'b=2']]`` and
    ``[['a=1'], ['b=2']]`` when ``nargs='*'`` + ``action='append'`` is used.
    """
    if not opt:
        return []
    flat = []
    for item in opt:
        if isinstance(item, str):
            flat.append(item)
        else:
            flat.extend(item)
    return flat


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def _parse_value(value):
    if not isinstance(value, str):
        return value
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    if value.startswith("[") or value.startswith("{"):
        try:
            return yaml.safe_load(value)
        except Exception:
            pass
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value


def _get_nested(config, keys):
    cur = config
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _coerce_like(existing, value):
    """Let list fields be given without brackets, e.g. label_file_list=a.txt."""
    if isinstance(existing, list):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip() != ""]
            if len(parts) > 1:
                return parts
            return [value]
        return [value]
    return value


def _set_nested(config, keys, value):
    cur = config
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def merge_config(config, opts):
    """Apply PaddleOCR-style ``-o Key.Sub=value`` overrides.

    ``opts`` is a list of strings. A bare string without ``=`` is treated as a
    dotted key with value True. List fields may be written without brackets
    (``...label_file_list=a.txt,b.txt``).
    """
    config = copy.deepcopy(config)
    bad_sections = []
    unknown_keys = []
    applied = []
    for opt in opts:
        if "=" in opt:
            key, value = opt.split("=", 1)
        else:
            key, value = opt, "True"
        keys = key.split(".")
        if keys[0] not in config:
            bad_sections.append(key)
            continue
        if _get_nested(config, keys) is None and keys[-1] not in (
            config.get(keys[0]) or {}
        ):
            unknown_keys.append(key)
        existing = _get_nested(config, keys)
        _set_nested(config, keys, _coerce_like(existing, _parse_value(value)))
        applied.append(key)

    if bad_sections:
        sections = list(config.keys())
        lines = []
        for key in bad_sections:
            section = key.split(".")[0]
            guess = difflib.get_close_matches(section, sections, n=1)
            item = "  - '{}'".format(key)
            if guess:
                item += "  (did you mean '{}..'?)".format(guess[0])
            lines.append(item)
        raise ValueError(
            "Invalid -o override: unknown config section(s):\n{}\n"
            "Available sections: {}\n"
            "Hint: cmd.exe/Bash need no quotes; PowerShell users should wrap the "
            "whole key=value in double quotes.".format(
                "\n".join(lines), ", ".join(sections)
            )
        )

    if unknown_keys:
        logging.getLogger("pytorchx/ocr").warning(
            "Overriding key(s) not present in the yml (added anyway): %s",
            ", ".join(unknown_keys),
        )
    if applied:
        prev = config.get("_overrides") or []
        config["_overrides"] = list(prev) + [k for k in applied if k not in prev]
    return config


def parse_args_to_config(cfg_path, opts=None):
    config = load_config(cfg_path)
    if opts:
        config = merge_config(config, opts)
    return config
