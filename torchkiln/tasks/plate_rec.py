"""Plate recognition task: CTC + colour loss / post-process / metric / adapter."""

from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from ptcore.task import TaskAdapter
from torchkiln.nn.plate import PLATE_CHARSET, PLATE_COLORS

__all__ = [
    "PlateRecLoss",
    "PlateRecPostProcess",
    "PlateRecMetric",
    "PlateRecTask",
    "build_plate_rec_loss",
    "build_plate_rec_postprocess",
    "build_plate_rec_metric",
]


class PlateRecLoss(nn.Module):
    """CTC (blank = 0) on the character head + cross entropy on the colour head."""

    def __init__(
        self,
        ctc_gain=1.0,
        color_gain=1.0,
        label_smoothing=0.0,
        blank=0,
        **kwargs
    ):
        super().__init__()
        self.ctc_gain = float(ctc_gain)
        self.color_gain = float(color_gain)
        self.blank = int(blank)
        self.ctc = nn.CTCLoss(
            blank=self.blank, reduction="mean", zero_infinity=True
        )
        self.color_ce = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing or 0.0)
        )

    def forward(self, preds, batch):
        log_probs, color_logits = preds
        targets, lengths, colors = batch[1], batch[2], batch[3]
        t = int(log_probs.shape[0])
        n = int(log_probs.shape[1])
        input_lengths = torch.full((n,), t, dtype=torch.long, device=log_probs.device)
        loss_ctc = self.ctc(
            log_probs, targets.to(log_probs.device), input_lengths,
            lengths.to(log_probs.device),
        )
        loss_color = log_probs.new_zeros(())
        if color_logits is not None and colors is not None and len(colors):
            loss_color = self.color_ce(color_logits, colors.to(color_logits.device))
        loss = self.ctc_gain * loss_ctc + self.color_gain * loss_color
        return {
            "loss": loss,
            "loss_ctc": loss_ctc.detach(),
            "loss_color": loss_color.detach(),
        }


class PlateRecPostProcess(object):
    """Upstream greedy CTC decode + colour argmax (``plate_rec.py``)."""

    def __init__(self, blank=0, charset=None, colors=None, **kwargs):
        self.blank = int(blank)
        self.charset = list(charset) if charset else list(PLATE_CHARSET)
        self.colors = list(colors) if colors else list(PLATE_COLORS)

    @staticmethod
    def _greedy(index, blank=0):
        """Collapse repeats and drop blanks, keeping per-char probabilities."""
        out, prob_idx, pre = [], [], -1
        for i, v in enumerate(index):
            if v != blank and v != pre:
                out.append(v)
                prob_idx.append(i)
            pre = v
        return out, prob_idx

    def __call__(self, preds):
        log_probs, color_logits = preds if isinstance(preds, (tuple, list)) else (preds, None)
        # ``log_probs`` is (T, N, C) with log-softmax already applied
        probs = log_probs.detach().exp()
        index = probs.argmax(-1).transpose(0, 1)  # (N, T)
        conf = probs.max(-1)[0].transpose(0, 1)  # (N, T)
        results = []
        for b in range(index.shape[0]):
            ids, pos = self._greedy(index[b].tolist(), self.blank)
            text = "".join(self.charset[i] for i in ids)
            score = float(np.mean([conf[b, p].item() for p in pos])) if pos else 0.0
            item = {"text": text, "chars": ids, "score": score}
            if color_logits is not None:
                cp = torch.softmax(color_logits.detach()[b], dim=-1)
                c_conf, c_idx = cp.max(-1)
                item["color"] = int(c_idx)
                item["color_name"] = self.colors[int(c_idx)]
                item["color_score"] = float(c_conf)
            results.append(item)
        return results


class PlateRecMetric(object):
    def __init__(self, main_indicator="acc", **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.n = 0
        self.correct = 0
        self.char_total = 0
        self.char_correct = 0
        self.color_n = 0
        self.color_correct = 0

    def __call__(self, post_result, batch):
        targets, lengths, colors = batch[1], batch[2], batch[3]
        offsets = torch.cumsum(lengths, 0)
        start = 0
        for i, res in enumerate(post_result):
            end = int(offsets[i])
            gt = targets[start:end].tolist()
            start = end
            pred = res.get("chars", [])
            self.n += 1
            if pred == gt:
                self.correct += 1
            self.char_total += max(len(gt), 1)
            self.char_correct += sum(1 for a, b in zip(pred, gt) if a == b)
            if "color" in res and colors is not None and i < len(colors):
                self.color_n += 1
                if int(res["color"]) == int(colors[i]):
                    self.color_correct += 1

    def get_metric(self):
        acc = self.correct / max(self.n, 1)
        return {
            "acc": acc,
            "char_acc": self.char_correct / max(self.char_total, 1),
            "color_acc": self.color_correct / max(self.color_n, 1) if self.color_n else 0.0,
            "num": self.n,
        }


def build_plate_rec_loss(loss_cfg):
    cfg = dict(loss_cfg or {})
    cfg.pop("name", None)
    cfg.pop("topk", None)
    cfg.pop("alpha", None)
    cfg.pop("beta", None)
    return PlateRecLoss(**cfg)


def build_plate_rec_postprocess(pp_cfg):
    cfg = dict(pp_cfg or {})
    cfg.pop("name", None)
    return PlateRecPostProcess(**cfg)


def build_plate_rec_metric(metric_cfg):
    cfg = dict(metric_cfg or {})
    cfg.pop("name", None)
    return PlateRecMetric(**cfg)


class PlateRecTask(TaskAdapter):
    """Licence-plate recognition + colour (``Architecture.task: plate_rec``)."""

    name = "plate_rec"

    def build_post_process(self, config):
        return build_plate_rec_postprocess(config.get("PostProcess"))

    def build_model(self, config, post_process):
        from torchkiln.models.plate import build_rec_model

        return build_rec_model(config["Architecture"])

    def build_loss(self, config, model):
        return build_plate_rec_loss(config.get("Loss"))

    def build_metric(self, config):
        return build_plate_rec_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from torchkiln.data.plate import PlateRecDataset

        train_ds = PlateRecDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = PlateRecDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.plate import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.plate import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        metric(post_process(model(images)), batch)

    def summary_lines(self, config, global_config, post_process):
        head = (config.get("Architecture") or {}).get("Head") or {}
        ds = (config.get("Train") or {}).get("dataset") or {}
        return [
            "task=plate_rec cfg={} num_classes={} color_num={} image_size={}".format(
                head.get("cfg"),
                head.get("num_classes"),
                head.get("color_num"),
                (ds.get("transform") or {}).get("image_size"),
            )
        ]
