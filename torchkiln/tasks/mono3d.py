"""单目 3D 检测任务（``Architecture.task: mono3d``）。

**模型 / loss / metric / 后处理全部复用 LiDAR 版 ``det3d``**，只换数据入口：
``Mono3DDataset`` 读 RGB 图 + ``labels/*.txt``（7-dof，LiDAR 系），collate 复用
``det3d`` 的实现（本来就是「按 ``(B,C,H,W)`` 堆叠 + ``(B,max_gt,8)`` 目标」，
对图像天然成立）。

为什么不新写一套网络：``PillarDetNet`` 就是若干 ``nn.Conv2d``，输入契约只有
``(B, C, H, W)``；RGB 图是 ``(B, 3, H, W)``，秩与通道数都对得上，改
``Architecture.in_channels: 3`` 即可。

⚠️ 坐标系：本任务**不读相机内参、不做任何坐标变换**。标签必须是 **LiDAR/ego 系**
（x 前 / y 左 / z 上）、``z`` 为框**中心**、``l`` 沿 ``yaw``——与 ``det3d`` 完全一致。
相机系 → LiDAR 系的转换在平台侧导出器里完成。
"""
from __future__ import absolute_import

from ptcore.task import TaskAdapter


class Mono3DTask(TaskAdapter):
    """Monocular 3D detection（复用 det3d 的网络与损失）."""

    name = "mono3d"

    def _pc_opts(self, config):
        ds = (config.get("Train", {}) or {}).get("dataset") or {}
        return ds.get("pc_range"), ds.get("pillar_size")

    def _ds_cfg(self, config):
        return (config.get("Train", {}) or {}).get("dataset") or {}

    def build_post_process(self, config):
        from torchkiln.det3d import build_det3d_postprocess

        pc_range, pillar_size = self._pc_opts(config)
        return build_det3d_postprocess(
            config.get("PostProcess"), pc_range=pc_range, pillar_size=pillar_size
        )

    def build_model(self, config, post_process):
        from torchkiln.models.det3d import build_det3d_model

        return build_det3d_model(config["Architecture"])

    def build_loss(self, config, model):
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
        from torchkiln.det3d import build_det3d_metric

        head = (config.get("Architecture", {}) or {}).get("Head") or {}
        return build_det3d_metric(
            config.get("Metric"),
            num_classes=head.get("num_classes"),
            names=self._ds_cfg(config).get("names"),
        )

    def build_datasets(self, config, logger):
        from torchkiln.data.mono3d import Mono3DDataset

        train_ds = Mono3DDataset(config, "Train", logger)
        eval_ds = None
        if config.get("Eval") is not None:
            eval_ds = Mono3DDataset(config, "Eval", logger)
        return train_ds, eval_ds

    def train_collate(self, batch):
        from torchkiln.data.det3d import train_collate

        return train_collate(batch)

    def eval_collate(self, batch):
        from torchkiln.data.det3d import eval_collate

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
        head = (config.get("Architecture", {}) or {}).get("Head") or {}
        return [
            "mono3d: classes={}, pc_range={}, pillar_size={}".format(
                head.get("num_classes"),
                self._pc_opts(config)[0],
                self._pc_opts(config)[1],
            ),
        ]
