"""3D detection task (``Architecture.task: det3d``).

Two algorithms:
* ``pointpillars`` (default) — lightweight dense-CNN ``PillarDetNet`` on a
  pre-pillarized BEV.
* ``centerpoint`` — SOTA CenterPoint-Pillars (``torchkiln.nn.centerpoint``)
  consuming the raw cloud; numerically aligned with Paddle3D.
"""
from __future__ import absolute_import

from ptcore.task import TaskAdapter


class Det3DTask(TaskAdapter):
    """LiDAR 3D detection (pillar BEV or CenterPoint)."""

    name = "det3d"

    def _algo(self, config):
        return str((config.get("Architecture") or {}).get("algorithm", "pointpillars")).lower()

    def _pc_opts(self, config):
        ds = (config.get("Train", {}) or {}).get("dataset") or {}
        return ds.get("pc_range"), ds.get("pillar_size")

    def _ds_cfg(self, config):
        return (config.get("Train", {}) or {}).get("dataset") or {}

    def build_post_process(self, config):
        if self._algo(config) == "centerpoint":
            from torchkiln.nn.centerpoint_task import build_centerpoint_postprocess

            return build_centerpoint_postprocess(
                config.get("PostProcess"), config.get("Architecture"), self._ds_cfg(config)
            )
        from torchkiln.det3d import build_det3d_postprocess

        pc_range, pillar_size = self._pc_opts(config)
        return build_det3d_postprocess(
            config.get("PostProcess"), pc_range=pc_range, pillar_size=pillar_size
        )

    def build_model(self, config, post_process):
        from torchkiln.models.det3d import build_det3d_model

        return build_det3d_model(config["Architecture"])

    def build_loss(self, config, model):
        if self._algo(config) == "centerpoint":
            from torchkiln.nn.centerpoint_task import build_centerpoint_loss

            return build_centerpoint_loss(
                config.get("Loss"), config.get("Architecture"), self._ds_cfg(config)
            )
        from torchkiln.det3d import build_det3d_loss

        arch = config.get("Architecture", {}) or {}
        head = arch.get("Head") or {}
        pc_range, pillar_size = self._pc_opts(config)
        return build_det3d_loss(
            config.get("Loss"),
            num_classes=head.get("num_classes"),
            pc_range=pc_range,
            pillar_size=pillar_size,
        )

    def build_metric(self, config):
        ds = (config.get("Eval", {}) or {}).get("dataset") or self._ds_cfg(config)
        if self._algo(config) == "centerpoint":
            from torchkiln.nn.centerpoint_task import build_centerpoint_metric

            return build_centerpoint_metric(
                config.get("Metric"), config.get("Architecture"), ds
            )
        from torchkiln.det3d import build_det3d_metric

        head = (config.get("Architecture", {}) or {}).get("Head") or {}
        return build_det3d_metric(
            config.get("Metric"),
            num_classes=head.get("num_classes"),
            names=ds.get("names"),
        )

    def build_datasets(self, config, logger):
        from torchkiln.data.det3d import Det3DDataset

        self._raw = self._algo(config) == "centerpoint"
        train_ds = Det3DDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = Det3DDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.det3d import raw_collate, train_collate

        if self._raw:
            return raw_collate(batch)
        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.det3d import eval_collate, raw_collate

        if self._raw:
            return raw_collate(batch)
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
        ds = self._ds_cfg(config)
        algo = self._algo(config)
        if algo == "centerpoint":
            return [
                "task=det3d algo=centerpoint tasks={} pc_range={} voxel={} batch={}".format(
                    head.get("tasks"),
                    head.get("point_cloud_range"),
                    head.get("voxel_size"),
                    ((config.get("Train", {}) or {}).get("loader") or {}).get(
                        "batch_size_per_card"
                    ),
                )
            ]
        return [
            "task=det3d num_classes={} pc_range={} pillar={} batch={}".format(
                head.get("num_classes"),
                ds.get("pc_range"),
                ds.get("pillar_size"),
                ((config.get("Train", {}) or {}).get("loader") or {}).get(
                    "batch_size_per_card"
                ),
            )
        ]
