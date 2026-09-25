"""ts_forecast trainer (`Architecture.task: ts_forecast`)."""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["TsForecastTrainer"]


class TsForecastTrainer(BaseTrainer):
    """Time-series forecasting."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("ts_forecast")
