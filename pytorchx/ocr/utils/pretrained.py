"""Backward-compatible shim: this module moved to :mod:`ptcore.pretrained`."""
from __future__ import absolute_import

from ptcore.pretrained import *  # noqa: F401,F403
from ptcore.pretrained import (  # noqa: F401
    available_names,
    download_pretrained,
    model_url,
    pretrained_dir,
    resolve_pretrained,
)
