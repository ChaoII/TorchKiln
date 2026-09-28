"""PANNs 音频分类 trainer (``Architecture.task: panns_cls``)。"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["PannsClsTrainer"]


class PannsClsTrainer(BaseTrainer):
    """PANNs CNN14 音频分类（ESC-50 训练对齐用）。"""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("panns_cls")
