"""Time-series forecasting task (``Architecture.task: ts_forecast``)."""
from __future__ import absolute_import

from ptcore.task import TaskAdapter


class TsForecastTask(TaskAdapter):
    """Multivariate time-series forecasting (PaddleTS-compatible models)."""

    name = "ts_forecast"

    def build_post_process(self, config):
        return None

    def build_model(self, config, post_process):
        from torchkiln.models.ts import build_ts_model

        return build_ts_model(config["Architecture"])

    def build_loss(self, config, model):
        from torchkiln.ts import build_ts_loss

        return build_ts_loss(config.get("Loss"))

    def build_metric(self, config):
        from torchkiln.ts import build_ts_metric

        return build_ts_metric(config.get("Metric"))

    def build_datasets(self, config, logger):
        from torchkiln.data.ts import TSDataset

        train_ds = TSDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = TSDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.ts import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.ts import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model({"past_target": images})

    def eval_step(self, model, batch, post_process, metric, device):
        past = batch[0].to(device, non_blocking=True)
        preds = model({"past_target": past})
        metric(preds, batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        ds = (config.get("Train", {}) or {}).get("dataset") or {}
        return [
            "task=ts_forecast model={} in_chunk_len={} out_chunk_len={} target_dim={}".format(
                head.get("model"), ds.get("in_chunk_len"), ds.get("out_chunk_len"),
                head.get("target_dim"))
        ]
