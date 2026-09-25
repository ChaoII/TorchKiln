"""Point-cloud pillar segmentation task (``Architecture.task: pc_seg``)."""
from __future__ import absolute_import

from ptcore.task import TaskAdapter


class PcSegTask(TaskAdapter):
    """LiDAR pillar/BEV segmentation."""

    name = "pc_seg"

    def _algo(self, config):
        return str((config.get("Architecture") or {}).get("algorithm", "pointpillars")).lower()

    def build_post_process(self, config):
        if self._algo(config) == "squeezesegv3":
            from torchkiln.nn.sac_rangenet import build_squeezesegv3_postprocess

            return build_squeezesegv3_postprocess(config.get("PostProcess"))
        from torchkiln.sem import build_sem_postprocess

        return build_sem_postprocess(config.get("PostProcess"))

    def build_model(self, config, post_process):
        from torchkiln.models.pc import build_pc_seg_model

        return build_pc_seg_model(config["Architecture"])

    def build_loss(self, config, model):
        arch = config.get("Architecture", {}) or {}
        num_classes = (arch.get("Head") or {}).get("num_classes")
        if self._algo(config) == "squeezesegv3":
            from torchkiln.nn.sac_rangenet import build_squeezesegv3_loss

            return build_squeezesegv3_loss(config.get("Loss"), num_classes=num_classes)
        from torchkiln.sem import build_sem_loss

        return build_sem_loss(config.get("Loss"))

    def build_metric(self, config):
        from torchkiln.sem import build_sem_metric

        arch = config.get("Architecture", {}) or {}
        num_classes = (arch.get("Head") or {}).get("num_classes")
        return build_sem_metric(config.get("Metric"), num_classes=num_classes)

    def build_datasets(self, config, logger):
        from torchkiln.data.pc import PointCloudDataset

        train_ds = PointCloudDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = PointCloudDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.pc import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.pc import eval_collate

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
        ds = (config.get("Train", {}) or {}).get("dataset") or {}
        return [
            "task=pc_seg num_classes={} pc_range={} pillar={} batch={}".format(
                head.get("num_classes"),
                ds.get("pc_range"),
                ds.get("pillar_size"),
                ((config.get("Train", {}) or {}).get("loader") or {}).get(
                    "batch_size_per_card"
                ),
            )
        ]
