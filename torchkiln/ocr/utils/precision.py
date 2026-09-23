"""Backward-compatible shim: this module moved to :mod:`ptcore.precision`."""
from __future__ import absolute_import

from ptcore.precision import *  # noqa: F401,F403
from ptcore.precision import enable_paddle_like_precision  # noqa: F401
