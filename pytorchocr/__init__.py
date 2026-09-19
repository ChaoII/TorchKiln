"""Compatibility shim: the OCR package moved to :mod:pytorchx.ocr."""
import importlib
import importlib.abc
import importlib.util
import sys

_PREFIX = "pytorchocr"
_TARGET = "pytorchx.ocr"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, fullname, realname):
        self.fullname = fullname
        self.realname = realname

    def create_module(self, spec):
        module = importlib.import_module(self.realname)
        sys.modules[self.fullname] = module
        return module

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _PREFIX and not fullname.startswith(_PREFIX + "."):
            return None
        realname = _TARGET + fullname[len(_PREFIX):]
        return importlib.util.spec_from_loader(fullname, _AliasLoader(fullname, realname))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

from pytorchx.ocr import *  # noqa: F401,F403
