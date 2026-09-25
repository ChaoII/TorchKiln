"""BEV-LaneDet task (``Architecture.task: lane_bev``)."""
from __future__ import absolute_import

from ptcore.task import TaskAdapter


class LaneBEVTask(TaskAdapter):
    """Camera image -> BEV lane detection (BEV-LaneDet)."""

    name = "lane_bev"

    def build_post_process(self, config):
        from torchkiln.lane_bev import build_lane_bev_postprocess

        return build_lane_bev_postprocess(config.get("PostProcess"))

    def build_model(self, config, post_process):
        from torchkiln.models.lane_bev import build_lane_bev_model

        return build_lane_bev_model(config["Architecture"])

    def build_loss(self, config, model):
        from torchkiln.lane_bev import build_lane_bev_loss

        return build_lane_bev_loss(config.get("Loss"))

    def build_metric(self, config):
        from torchkiln.lane_bev import build_lane_bev_metric

        return build_lane_bev_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from torchkiln.data.lane_bev import LaneBEVDataset

        train_ds = LaneBEVDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = LaneBEVDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.lane_bev import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.lane_bev import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        preds = model(images)
        result = post_process(preds)
        metric(result, batch)

    def sample_count(self, batch):
        try:
            return int(batch[0].shape[0])
        except Exception:  # noqa: BLE001
            return 0

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        ds = (config.get("Train", {}) or {}).get("dataset") or {}
        return [
            "task=lane_bev bev_shape={} output_2d_shape={} imgsz={}".format(
                head.get("bev_shape"), head.get("output_2d_shape"), ds.get("input_shape"))
        ]
