"""OCR task adapter (text detection + recognition) for ``ptcore.trainer``."""
from __future__ import absolute_import

import copy

import numpy as np
import torch

from ptcore.task import TaskAdapter
from torchkiln.ocr.data.simple_dataset import SimpleDataSet
from torchkiln.ocr.losses import build_loss
from torchkiln.ocr.metrics import build_metric
from torchkiln.ocr.modeling.architectures.base_model import BaseModel
from torchkiln.ocr.postprocess import build_post_process

__all__ = ["OcrTask", "det_train_collate", "det_eval_collate", "rec_collate"]


def det_train_collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    num_keys = len(batch[0])
    return [
        torch.from_numpy(np.stack([s[k] for s in batch], axis=0)).float()
        for k in range(num_keys)
    ]


def det_eval_collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    image = torch.from_numpy(np.stack([s[0] for s in batch], axis=0)).float()
    shape = torch.from_numpy(np.stack([s[1] for s in batch], axis=0)).float()
    polys = [s[2] for s in batch]
    ignore_tags = [s[3] for s in batch]
    return [image, shape, polys, ignore_tags]


def rec_collate(batch):
    batch = [s for s in batch if s is not None and len(s) > 0]
    if len(batch) == 0:
        return []
    num_keys = len(batch[0])
    out = []
    for k in range(num_keys):
        arrs = [np.asarray(s[k]) for s in batch]
        out.append(torch.from_numpy(np.stack(arrs, axis=0)))
    return out


class OcrTask(TaskAdapter):
    """Text detection (``model_type: det``) / recognition (``rec``)."""

    def __init__(self, config=None):
        if config is not None:
            self.name = config.get("Architecture", {}).get("model_type", "det")

    # ------------------------------------------------------------------ build
    def build_post_process(self, config):
        return build_post_process(
            config.get("PostProcess"), global_config=config.get("Global")
        )

    def build_model(self, config, post_process):
        arch = copy.deepcopy(config["Architecture"])
        self._configure_head_outputs(arch, config, post_process)
        return BaseModel(arch)

    def build_loss(self, config, model):
        return build_loss(config["Loss"])

    def build_metric(self, config):
        return build_metric(config["Metric"])

    def build_datasets(self, config, logger):
        train_ds = SimpleDataSet(config, "Train", logger, None)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = SimpleDataSet(config, "Eval", logger, None)
        return train_ds, eval_ds

    # -------------------------------------------------------------- dataloader
    def train_collate(self, batch):
        if self.name == "det":
            return det_train_collate(batch)
        return rec_collate(batch)

    def eval_collate(self, batch):
        if self.name == "det":
            return det_eval_collate(batch)
        return rec_collate(batch)

    # ----------------------------------------------------------------- forward
    def forward_train(self, model, images, batch):
        if self.name == "rec":
            return model(images, data=batch[1:])
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        preds = model(images)
        if self.name == "det":
            shape_list = batch[1].cpu().numpy()
            post_result = post_process(preds, shape_list)
            metric(post_result, batch)
        else:
            preds_cpu = preds.detach().cpu()
            batch_cpu = [
                b.detach().cpu() if torch.is_tensor(b) else b for b in batch
            ]
            post_result = post_process(preds_cpu, batch_cpu[1])
            metric(post_result, batch_cpu)

    # ----------------------------------------------------------------- logging
    def summary_lines(self, config, global_config, post_process):
        lines = []
        char_num = len(getattr(post_process, "character", []) or [])
        if char_num:
            lines.append(
                "dict={} num_classes={} max_text_length={} use_space_char={}".format(
                    global_config.get("character_dict_path"),
                    char_num,
                    global_config.get("max_text_length"),
                    global_config.get("use_space_char"),
                )
            )
        post = config.get("PostProcess", {}) or {}
        if post.get("name") == "DBPostProcess":
            lines.append(
                "postprocess thresh={} box_thresh={} unclip={} max_candidates={}".format(
                    post.get("thresh"),
                    post.get("box_thresh"),
                    post.get("unclip_ratio"),
                    post.get("max_candidates"),
                )
            )
        return lines

    # ------------------------------------------------------------------ helper
    def _configure_head_outputs(self, arch, config, post_process):
        head = arch.get("Head")
        if not head:
            return
        if not hasattr(post_process, "character"):
            return
        char_num = len(getattr(post_process, "character"))
        loss_cfg = config.get("Loss", {})
        loss_list = loss_cfg.get("loss_config_list", [])
        if head.get("name") == "MultiHead":
            if config.get("PostProcess", {}).get("name") == "SARLabelDecode":
                char_num = char_num - 2
            if config.get("PostProcess", {}).get("name") == "NRTRLabelDecode":
                char_num = char_num - 3
            ocl = {"CTCLabelDecode": char_num}
            # Provide auxiliary head sizes so MultiHead can always build its
            # SAR/NRTR modules (weights are kept for conversion compatibility).
            ocl["SARLabelDecode"] = char_num + 2
            ocl["NRTRLabelDecode"] = char_num + 4
            if len(loss_list) > 1:
                second = list(loss_list[1].keys())[0]
                if second == "SARLoss":
                    if loss_list[1]["SARLoss"] is None:
                        loss_list[1]["SARLoss"] = {"ignore_index": char_num + 1}
                    else:
                        loss_list[1]["SARLoss"]["ignore_index"] = char_num + 1
            head["out_channels_list"] = ocl
        else:
            head["out_channels"] = char_num
