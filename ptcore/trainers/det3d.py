"""det3d trainer (`Architecture.task: det3d`)."""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["Det3DTrainer"]


class Det3DTrainer(BaseTrainer):
    """LiDAR 3D detection (CenterPoint-style)."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("det3d")
