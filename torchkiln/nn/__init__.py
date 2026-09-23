"""torchkiln.nn: module zoo + YAML graph parser."""
from __future__ import absolute_import

from torchkiln.nn import plate as _plate  # noqa: F401  (registers plate modules)
from torchkiln.nn.modules import REGISTRY, get_module, register  # noqa: F401

__all__ = ["REGISTRY", "get_module", "register"]
