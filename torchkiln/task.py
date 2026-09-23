"""Compatibility shim: task adapters now live in :mod:`torchkiln.tasks`."""
from torchkiln.tasks import *  # noqa: F401,F403
from torchkiln.tasks import AVAILABLE, TASK_REGISTRY, get_task  # noqa: F401
