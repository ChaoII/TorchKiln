"""pose trainer (`Architecture.task: pose`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `torchkiln/tasks/pose.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["PoseTrainer"]


class PoseTrainer(BaseTrainer):
    """Keypoint / pose estimation."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("pose")
