"""OCR trainer: text detection / recognition (``model_family: ocr``).

Wraps :class:`torchkiln.ocr.task.OcrTask`, which internally dispatches on the
configured algorithm (det / rec).
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["OcrTrainer"]


class OcrTrainer(BaseTrainer):
    """Text detection / recognition."""

    def _default_task(self, config):
        from torchkiln.ocr.task import OcrTask

        return OcrTask(config)
