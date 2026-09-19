"""detect trainer (`Architecture.task: detect`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `pytorchx/tasks/detect.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["DetectTrainer"]


class DetectTrainer(BaseTrainer):
    """Object detection."""

    def _default_task(self, config):
        from pytorchx.tasks import get_task

        return get_task("detect")
