"""pytorchx.nn: module zoo + YAML graph parser."""
from __future__ import absolute_import

from pytorchx.nn import plate as _plate  # noqa: F401  (registers plate modules)
from pytorchx.nn.modules import REGISTRY, get_module, register  # noqa: F401

__all__ = ["REGISTRY", "get_module", "register"]
