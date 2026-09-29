"""ts_classify trainer（``Architecture.task: ts_classify``）。"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["TsClassifyTrainer"]


class TsClassifyTrainer(BaseTrainer):
    """Time-series classification (CNN / InceptionTime)."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("ts_classify")
