"""segment trainer (`Architecture.task: segment`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `torchkiln/tasks/segment.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["SegmentTrainer"]


class SegmentTrainer(BaseTrainer):
    """Instance segmentation."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("segment")
