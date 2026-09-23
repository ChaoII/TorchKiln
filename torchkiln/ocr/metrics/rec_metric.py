from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import numpy as np
import editdistance


class RecMetric(object):
    def __init__(
        self, main_indicator="acc", is_filter=False, ignore_space=True, **kwargs
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text.lower()

    @staticmethod
    def _is_text_pair_list(value):
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            return False
        item = value[0]
        return (
            isinstance(item, (list, tuple))
            and len(item) >= 1
            and isinstance(item[0], str)
        )

    def _split_preds_labels(self, preds, batch):
        if batch is None:
            return preds
        if (
            isinstance(preds, (list, tuple))
            and len(preds) == 2
            and self._is_text_pair_list(preds[0])
            and self._is_text_pair_list(preds[1])
        ):
            return preds
        return preds, batch

    def __call__(self, preds, batch=None, *args, **kwargs):
        preds, labels = self._split_preds_labels(preds, batch)
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            denom = max(len(pred), len(target))
            if denom == 0:
                norm_edit_dis += 0.0
            else:
                norm_edit_dis += editdistance.eval(pred, target) / denom
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            "acc": correct_num / (all_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        self.reset()
        return {"acc": acc, "norm_edit_dis": norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
