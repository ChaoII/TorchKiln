"""YOLO trainer: ``ptcore``'s generic trainer pre-wired with YOLO tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptcore.trainer import BaseTrainer

__all__ = ["Trainer", "build_task"]


def build_task(config):
    """Pick the task adapter for ``Architecture.task`` (see ``pytorchx.tasks``)."""
    from pytorchx.tasks import get_task

    arch = config.get("Architecture") or {}
    return get_task(arch.get("task", "classify"))


class Trainer(BaseTrainer):
    def _default_task(self, config):
        return build_task(config)
