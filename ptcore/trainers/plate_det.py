"""plate_det trainer (`Architecture.task: plate_det`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `pytorchx/tasks/plate_det.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["PlateDetTrainer"]


class PlateDetTrainer(BaseTrainer):
    """License-plate detection."""

    def _default_task(self, config):
        from pytorchx.tasks import get_task

        return get_task("plate_det")
