"""Compatibility shim: task adapters now live in :mod:`pytorchx.tasks`."""
from pytorchx.tasks import *  # noqa: F401,F403
from pytorchx.tasks import AVAILABLE, TASK_REGISTRY, get_task  # noqa: F401
