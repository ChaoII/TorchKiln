"""attribute trainer (`Architecture.task: attribute`).

Shared loop lives in :class:ptcore.trainers.base.BaseTrainer; task specific
model/loss/metric/dataset construction lives in `pytorchx/tasks/attribute.py`.
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["AttributeTrainer"]


class AttributeTrainer(BaseTrainer):
    """Multi-label attribute recognition."""

    def _default_task(self, config):
        from pytorchx.tasks import get_task

        return get_task("attribute")
