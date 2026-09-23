"""obb trainer (`Architecture.task: obb`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `torchkiln/tasks/obb.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["ObbTrainer"]


class ObbTrainer(BaseTrainer):
    """Oriented bounding boxes."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("obb")
