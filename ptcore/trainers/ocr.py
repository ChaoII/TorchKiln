"""OCR trainer: text detection / recognition (``model_family: ocr``).

Wraps :class:`pytorchx.ocr.task.OcrTask`, which internally dispatches on the
configured algorithm (det / rec).
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["OcrTrainer"]


class OcrTrainer(BaseTrainer):
    """Text detection / recognition."""

    def _default_task(self, config):
        from pytorchx.ocr.task import OcrTask

        return OcrTask(config)
