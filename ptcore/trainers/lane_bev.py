"""lane_bev trainer (`Architecture.task: lane_bev`)."""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["LaneBEVTrainer"]


class LaneBEVTrainer(BaseTrainer):
    """BEV-LaneDet (camera -> BEV lane)."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("lane_bev")
