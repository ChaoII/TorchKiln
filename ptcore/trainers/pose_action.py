"""pose_action trainer (`Architecture.task: pose_action`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `pytorchx/tasks/pose_action.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["PoseActionTrainer"]


class PoseActionTrainer(BaseTrainer):
    """Skeleton-based action recognition."""

    def _default_task(self, config):
        from pytorchx.tasks import get_task

        return get_task("pose_action")
