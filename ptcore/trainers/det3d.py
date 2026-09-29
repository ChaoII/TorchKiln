"""det3d trainer (`Architecture.task: det3d`)."""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["Det3DTrainer"]


class Det3DTrainer(BaseTrainer):
    """3D detection trainer（LiDAR ``det3d`` 与单目 ``mono3d`` 共用）。

    ⚠️ ``_default_task`` **不能硬编码返回 det3d**。``ptcore.factory.build_trainer``
    虽然按 ``Architecture.task`` 选了 trainer 类，但构造时并**不把这个 task 传进去**，
    于是 ``BaseTrainer.__init__`` 走 ``self._default_task(config)``。以往「一个 task
    对应一个 trainer 类」，两者恰好一致所以没人发现；mono3d 复用本类后，
    这里若返回 det3d 就会**静默用 LiDAR 的 Det3DDataset 去读图像**——
    读不出点云 → 标签全空 → ``loss_box`` 恒为 0，训练照跑不误，数据全错。

    所以必须以配置里声明的 task 为准。
    """

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        name = str((config.get("Architecture") or {}).get("task") or "det3d").lower()
        return get_task(name)
