"""Backward-compatible shim: this module moved to :mod:`ptcore.config`."""
from __future__ import absolute_import

from ptcore.config import *  # noqa: F401,F403
from ptcore.config import (  # noqa: F401
    flatten_opts,
    load_config,
    merge_config,
    parse_args_to_config,
)
