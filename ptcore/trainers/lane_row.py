"""lane_row trainer (`Architecture.task: lane_row`)."""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["LaneRowTrainer"]


class LaneRowTrainer(BaseTrainer):
    """Row-based lane detection (UFLD-style)."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("lane_row")
