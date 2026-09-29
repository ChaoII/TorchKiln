"""ts_anomaly trainer（``Architecture.task: ts_anomaly``）。

与其它 trainer 的差别：**两阶段异常检测模型（USAD / AnomalyTransformer）**
需要「每个 batch 两次 backward + step」，由 ``TSAnomalyLoss.train_step`` 钩子实现，
``BaseTrainer`` 检测到该钩子后会跳过默认的单次 backward/step 路径。
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["TsAnomalyTrainer"]


class TsAnomalyTrainer(BaseTrainer):
    """Time-series anomaly detection (AE / VAE / USAD / MTAD-GAT / AnomalyTransformer)."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("ts_anomaly")
