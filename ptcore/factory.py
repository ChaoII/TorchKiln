"""Trainer factory: pick the trainer for a config.

Kept for backward compatibility; the registry lives in :mod:`ptcore.trainers`.
The CLIs only read the config and ask for a trainer, e.g.::

    config["Architecture"]["model_family"] == "ocr"   -> OcrTrainer
    config["Architecture"]["task"]         == "detect" -> DetectTrainer
"""
from __future__ import absolute_import

from ptcore.trainers import build_trainer, get_trainer  # noqa: F401

__all__ = ["model_family", "build_trainer", "get_trainer"]


def model_family(config):
    return (config.get("Architecture") or {}).get("model_family", "ocr")
