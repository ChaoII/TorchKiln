"""classify trainer (`Architecture.task: classify`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `torchkiln/tasks/classify.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["ClassifyTrainer"]


class ClassifyTrainer(BaseTrainer):
    """Image classification."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("classify")
