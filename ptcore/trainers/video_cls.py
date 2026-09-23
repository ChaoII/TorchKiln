"""video_cls trainer (`Architecture.task: video_cls`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `torchkiln/tasks/video_cls.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["VideoClsTrainer"]


class VideoClsTrainer(BaseTrainer):
    """Video classification."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("video_cls")
