"""Row-based lane detection task (UFLD-style)."""
from __future__ import absolute_import

from ptcore.task import TaskAdapter


class LaneRowTask(TaskAdapter):
    """Lane detection via per-row x classification (``Architecture.task: lane_row``)."""

    name = "lane_row"

    def build_post_process(self, config):
        from torchkiln.lane import build_lane_row_postprocess

        return build_lane_row_postprocess(config.get("PostProcess"))

    def build_model(self, config, post_process):
        from torchkiln.models import build_arch_model

        return build_arch_model(config["Architecture"])

    def build_loss(self, config, model):
        from torchkiln.lane import build_lane_row_loss

        cfg = dict(config.get("Loss") or {})
        head = (config.get("Architecture") or {}).get("Head") or {}
        if "num_bins" not in cfg and head.get("num_bins"):
            cfg["num_bins"] = head["num_bins"]
        return build_lane_row_loss(cfg)

    def build_metric(self, config):
        from torchkiln.lane import build_lane_row_metric

        ds = ((config.get("Train") or {}).get("dataset") or {})
        kwargs = {}
        if ds.get("transform", {}) and ds.get("transform", {}).get("image_size"):
            kwargs["image_width"] = int(ds["transform"]["image_size"])
        return build_lane_row_metric(config.get("Metric"), **kwargs)

    def build_datasets(self, config, logger):
        from torchkiln.data.lane_row import LaneRowDataset

        head = (config.get("Architecture") or {}).get("Head") or {}
        # mirror Head hyper-params into dataset cfg if present
        for mode in ("Train", "Eval"):
            if config.get(mode) is None:
                continue
            d = config[mode].setdefault("dataset", {})
            for k in ("num_lanes", "num_rows"):
                if k in head and k not in d:
                    d[k] = head[k]
        train_ds = LaneRowDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = LaneRowDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.lane_row import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.lane_row import eval_collate

        return eval_collate(batch)

    def forward_train(self, model, images, batch):
        return model(images)

    def eval_step(self, model, batch, post_process, metric, device):
        images = batch[0].to(device, non_blocking=True)
        preds = model(images)
        result = post_process(preds)
        metric(result, batch)

    def summary_lines(self, config, global_config, post_process):
        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        return [
            "task=lane_row algorithm={} lanes={} rows={} bins={} imgsz={}".format(
                arch.get("algorithm"),
                head.get("num_lanes"),
                head.get("num_rows"),
                head.get("num_bins"),
                ((config.get("Train", {}).get("dataset") or {}).get("transform") or {}).get(
                    "image_size"
                ),
            )
        ]
