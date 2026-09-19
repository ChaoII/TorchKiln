"""semantic trainer (`Architecture.task: semantic`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `pytorchx/tasks/semantic.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["SemanticTrainer"]


class SemanticTrainer(BaseTrainer):
    """Semantic segmentation."""

    def _default_task(self, config):
        from pytorchx.tasks import get_task

        return get_task("semantic")
