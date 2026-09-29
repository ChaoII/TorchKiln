"""ts_rul trainer（``Architecture.task: ts_rul``）—— 剩余寿命预测。"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["TsRulTrainer"]


class TsRulTrainer(BaseTrainer):
    """Remaining Useful Life prediction (reuses ts_forecast model family)."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("ts_rul")
