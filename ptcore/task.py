"""Task adapters.

Everything in the training loop that depends on the *task* (how to build the
model / loss / metric / datasets, how to collate, how to run a training or an
evaluation step, what to print in the summary) lives behind this interface.

``BaseTrainer`` only talks to a :class:`TaskAdapter`, so new task families
(text detection/recognition, YOLO detect/classify/obb/segment/semantic/depth,
...) can be added without touching the training loop.
"""
from __future__ import absolute_import


class TaskAdapter(object):
    """Base class for task plugins.

    Subclasses must implement the hooks they need; the trainer calls them in
    this order during ``__init__``::

        build_post_process -> build_model -> build_loss -> build_metric
        -> build_datasets -> train_collate / eval_collate
    """

    #: short task name, used in logs (e.g. ``"det"``, ``"rec"``, ``"yolo_cls"``)
    name = "task"

    # ------------------------------------------------------------------ build
    def build_post_process(self, config):
        """Return the post-processing object (may be ``None``)."""
        raise NotImplementedError

    def build_model(self, config, post_process):
        """Return the ``nn.Module`` for this task."""
        raise NotImplementedError

    def build_loss(self, config, model):
        """Return the loss module (must be an ``nn.Module``)."""
        raise NotImplementedError

    def build_metric(self, config):
        """Return the metric object (``reset()`` / ``get_metric()``)."""
        raise NotImplementedError

    def build_datasets(self, config, logger):
        """Return ``(train_dataset, eval_dataset_or_None)``."""
        raise NotImplementedError

    # -------------------------------------------------------------- dataloader
    def train_collate(self, batch):
        raise NotImplementedError

    def eval_collate(self, batch):
        raise NotImplementedError

    # -------------------------------------------------------------- forward
    def forward_train(self, model, images, batch):
        """Training forward. ``images``/``batch`` are already on the device."""
        raise NotImplementedError

    def eval_step(self, model, batch, post_process, metric, device):
        """Run one evaluation batch: inference + post-process + metric update."""
        raise NotImplementedError

    def sample_count(self, batch):
        try:
            return int(batch[0].shape[0])
        except Exception:  # noqa: BLE001
            return 0

    # -------------------------------------------------------------- logging
    def summary_lines(self, config, global_config, post_process):
        """Extra ``Config: ...`` lines printed at startup."""
        return []
