"""Backward-compatible shim.

The training loop now lives in :mod:`ptcore.trainers.base` and the per-task
trainers live in :mod:`ptcore.trainers`. Import from there in new code.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer, get_logger  # noqa: F401

__all__ = ["BaseTrainer", "get_logger"]
