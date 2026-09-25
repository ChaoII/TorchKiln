"""lane_seg trainer (`Architecture.task: lane_seg`)."""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["LaneSegTrainer"]


class LaneSegTrainer(BaseTrainer):
    """Lane segmentation (mask)."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("lane_seg")
