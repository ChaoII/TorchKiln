"""OCR trainer: ``ptcore``'s generic trainer pre-wired with the OCR task.

The implementation lives in :class:`ptcore.trainer.BaseTrainer`; this module
only binds the OCR task adapter (text detection / recognition).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptcore.trainer import BaseTrainer, get_logger
from torchkiln.ocr.task import (
    OcrTask,
    det_eval_collate,
    det_train_collate,
    rec_collate,
)

__all__ = [
    "Trainer",
    "OcrTask",
    "get_logger",
    "det_train_collate",
    "det_eval_collate",
    "rec_collate",
]


class Trainer(BaseTrainer):
    """Backward-compatible name for the OCR trainer."""

    def _default_task(self, config):
        return OcrTask(config)
