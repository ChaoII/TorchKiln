"""plate_rec trainer (`Architecture.task: plate_rec`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `pytorchx/tasks/plate_rec.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["PlateRecTrainer"]


class PlateRecTrainer(BaseTrainer):
    """License-plate recognition."""

    def _default_task(self, config):
        from pytorchx.tasks import get_task

        return get_task("plate_rec")
