"""depth trainer (`Architecture.task: depth`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `torchkiln/tasks/depth.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["DepthTrainer"]


class DepthTrainer(BaseTrainer):
    """Depth regression."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("depth")
