"""ts_embed trainer（``Architecture.task: ts_embed``）—— 自监督表示学习。"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["TsEmbedTrainer"]


class TsEmbedTrainer(BaseTrainer):
    """Time-series representation learning (TS2Vec / CoST)."""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("ts_embed")
