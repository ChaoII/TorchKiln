"""Lane segmentation task (mask; same data layout as semantic)."""
from __future__ import absolute_import

from ptcore.task import TaskAdapter


class LaneSegTask(TaskAdapter):
    """Lane detection via semantic segmentation (``Architecture.task: lane_seg``)."""

    name = "lane_seg"

    def build_post_process(self, config):
        from torchkiln.lane import build_lane_seg_postprocess

        return build_lane_seg_postprocess(config.get("PostProcess"))

    def build_model(self, config, post_process):
        from torchkiln.models import build_arch_model

        arch = config["Architecture"]
        if arch.get("yaml_file") or arch.get("yaml_text"):
            return build_arch_model(arch)
        # hand-written path: reuse semantic builder
        return build_arch_model(arch, "semantic")

    def build_loss(self, config, model):
        from torchkiln.lane import build_lane_seg_loss

        return build_lane_seg_loss(config.get("Loss"))

    def build_metric(self, config):
        from torchkiln.lane import build_lane_seg_metric

        arch = config.get("Architecture", {}) or {}
        num_classes = (arch.get("Head") or {}).get("num_classes")
        return build_lane_seg_metric(config.get("Metric"), num_classes=num_classes)

    def build_datasets(self, config, logger):
        from torchkiln.data.sem import SemDataset

        train_ds = SemDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = SemDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.sem import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.sem import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        preds = model(images)
        result = post_process(preds, size=batch[1].shape[-2:])
        metric(result, batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        ds = config.get("Train", {}).get("dataset") or {}
        return [
            "task=lane_seg algorithm={} num_classes={} imgsz={}".format(
                arch.get("algorithm"),
                head.get("num_classes"),
                (ds.get("transform") or {}).get("image_size"),
            )
        ]
