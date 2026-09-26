"""Time-series forecasting task (``Architecture.task: ts_forecast``)."""
from __future__ import absolute_import

from ptcore.task import TaskAdapter


class TsForecastTask(TaskAdapter):
    """Multivariate time-series forecasting (PaddleTS-compatible models)."""

    name = "ts_forecast"

    @staticmethod
    def _head_model(config):
        head = (config.get("Architecture") or {}).get("Head") or {}
        return str(head.get("model", "")).lower()

    def build_post_process(self, config):
        return None

    def build_model(self, config, post_process):
        from torchkiln.models.ts import build_ts_model

        return build_ts_model(config["Architecture"])

    def build_loss(self, config, model):
        from torchkiln.ts import build_ts_loss

        loss_cfg = dict(config.get("Loss") or {})
        if "type" not in loss_cfg and "name" not in loss_cfg:
            m = self._head_model(config)
            if m == "tft":
                loss_cfg["type"] = "quantile"
            elif m == "deepar":
                loss_cfg["type"] = "nll"
        return build_ts_loss(loss_cfg)

    def build_metric(self, config):
        from torchkiln.ts import build_ts_metric

        metric_cfg = dict(config.get("Metric") or {})
        if "pred_mode" not in metric_cfg:
            m = self._head_model(config)
            if m == "tft":
                metric_cfg["pred_mode"] = "quantile"
            elif m == "deepar":
                metric_cfg["pred_mode"] = "params"
        return build_ts_metric(metric_cfg)

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

    @staticmethod
    def _build_input(images, batch, with_future=True):
        inp = {"past_target": images}
        if with_future and len(batch) > 1:
            inp["future_target"] = batch[1]
        if len(batch) > 2:
            inp["known_cov_numeric"] = batch[2]
            inp["observed_cov_numeric"] = batch[3]
            inp["static_cov_numeric"] = batch[4]
        return inp

    def forward_train(self, model, images, batch):
        return model(self._build_input(images, batch, with_future=True))

    def eval_step(self, model, batch, post_process, metric, device):
        past = batch[0].to(device, non_blocking=True)
        batch = [b.to(device, non_blocking=True) if hasattr(b, "to") else b for b in batch]
        inp = self._build_input(past, batch, with_future=False)
        preds = model(inp)
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
