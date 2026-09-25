"""pc_seg trainer (`Architecture.task: pc_seg`)."""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["PcSegTrainer"]


class PcSegTrainer(BaseTrainer):
    """Point-cloud pillar segmentation."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("pc_seg")
