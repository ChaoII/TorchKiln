"""YAML graph parser + generic graph model.

Reproduces the upstream *layer-list* model definition format::

    backbone:
      - [-1, 1, Conv, [64, 3, 2]]
      - [-1, 3, C2f,  [128, True]]
    head:
      - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
      - [[-1, 6], 1, Concat, [1]]
      - [[15, 18, 21], 1, Detect, [nc]]

Each entry is ``[from, repeats, module, args]``; ``from`` may be a single index
or a list of indices (``-1`` = previous layer). ``scales`` (depth, width,
max_channels) are applied the usual way.
"""
from __future__ import absolute_import

import math
import os

import torch
import torch.nn as nn
import yaml

from pytorchx.nn import plate as _plate  # noqa: F401
from pytorchx.nn.modules import (
    Classify,
    Detect10,
    Detect26,
    Depth,
    SemanticSegment,
    Classify,
    Detect,
    OBB,
    Pose,
    Segment,
    get_module,
    make_divisible,
    REGISTRY,
)
from pytorchx.nn.modules import (  # noqa: F401  (import side effects: registry)
    C1,
    C2,
    C2f,
    C2PSA,
    C3,
    C3k,
    C3k2,
    Concat,
    Conv,
    ConvTranspose,
    DWConv,
    PSA,
    SPP,
    SPPF,
    Bottleneck,
)

__all__ = ["parse_model", "GraphModel", "build_from_arch"]

SCALED = {
    "Conv",
    "DWConv",
    "ConvTranspose",
    "Bottleneck",
    "C1",
    "C2",
    "C2f",
    "C2fCIB",
    "C3",
    "C3x",
    "C3TR",
    "C3Ghost",
    "C3k",
    "C3k2",
    "PSA",
    "C2PSA",
    "C2fAttn",
    "SPP",
    "SPPF",
    "ADown",
    "AConv",
    "GhostConv",
    "ELAN1",
    "SCDown",
    "RepNCSPELAN4",
    "SPPELAN",
    "A2C2f",
    "RepNCSP",
    "StemBlock",
    "ShuffleV2Block",
    "BottleneckV5",
    "C3V5",
}

HEAD_CLASSES = {
    "Detect": get_module("Detect"),
    "DetectDFL": get_module("DetectDFL"),
    "Detect26": get_module("Detect26"),
    "Detect10": Detect10,
    "v10Detect": get_module("v10Detect"),
    "OBB": get_module("OBB"),
    "OBBDFL": get_module("OBBDFL"),
    "OBB26": get_module("OBB26"),
    "Segment": get_module("Segment"),
    "SegmentDFL": get_module("SegmentDFL"),
    "Segment26": get_module("Segment26"),
    "Pose": get_module("Pose"),
    "PoseDFL": get_module("PoseDFL"),
    "Pose26": get_module("Pose26"),
    "SemanticSegment": SemanticSegment,
    "Depth": Depth,
    "PlateDetect": get_module("PlateDetect"),
}

DEFAULT_SCALES = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}

# Modules whose 3rd positional arg is the repeat count ``n``. Like upstream, we
# insert the layer count at that position and then build a single block (the
# repetition lives inside the module, not as an ``nn.Sequential`` wrapper).
REPEAT_MODULES = {
    "Bottleneck",
    "C1",
    "C2",
    "C2f",
    "C2fCIB",
    "C3",
    "C3TR",
    "C3Ghost",
    "C3k",
    "C3k2",
    "PSA",
    "C2PSA",
    "A2C2f",
    "RepNCSP",
    "C3V5",
}


def _eval_arg(value, ctx):
    if isinstance(value, str):
        try:
            return eval(value, {"__builtins__": {}}, ctx)  # noqa: S307
        except Exception:
            return value
    if isinstance(value, list):
        return [_eval_arg(v, ctx) for v in value]
    if isinstance(value, tuple):
        return tuple(_eval_arg(v, ctx) for v in value)
    return value


def _module_name(m):
    if isinstance(m, str):
        return m
    return getattr(m, "__name__", str(m))


def parse_model(d, ch=3, verbose=False):
    """Build ``nn.Sequential`` of layers from a parsed YAML dict."""
    nc = int(d.get("nc", 80))
    scale = d.get("scale", "n")
    yaml_scales = d.get("scales") or {}
    if d.get("width_multiple") is not None:
        # YOLOv5-era config: depth_multiple / width_multiple (no max_channels cap)
        scale_params = (
            float(d.get("depth_multiple", 1.0)),
            float(d.get("width_multiple", 1.0)),
            10 ** 9,
        )
    elif isinstance(yaml_scales, (list, tuple)):
        scale_params = tuple(yaml_scales)
    else:
        scale_params = yaml_scales.get(scale)
    depth, width, max_channels = (
        d.get("scale_params")
        or scale_params
        or DEFAULT_SCALES.get(scale, DEFAULT_SCALES["n"])
    )
    ctx = {"nc": nc, "math": math, "None": None, "True": True, "False": False}

    layers, save = [], []
    layer_ch = []  # layer_ch[i] == output channels of layer i

    def _chan(idx, i):
        # index -1 means "previous layer"; other negatives are relative offsets
        if idx < 0:
            j = i + idx
            return ch if j < 0 else layer_ch[j]
        return layer_ch[idx]

    for i, (f, n, m, args) in enumerate(d.get("backbone", []) + d.get("head", [])):
        module_name = _module_name(m)
        module_cls = get_module(module_name) if isinstance(m, str) else m
        args = [_eval_arg(a, ctx) for a in (args or [])]

        f_list = f if isinstance(f, (list, tuple)) else [f]
        if module_name == "CBFuse":
            c1 = _chan(f_list[-1], i)
        else:
            c1 = sum(_chan(x, i) for x in f_list)

        if module_name == "Detect" and (d.get("kpt_label") or d.get("anchors")):
            # yolov5-face style checkpoint: `Detect` carries landmark channels
            module_name = "PlateDetect"

        if module_name in HEAD_CLASSES:
            head_cls = HEAD_CLASSES[module_name]
            ch_list = [_chan(x, i) for x in f_list]
            kwargs = {}
            extra = list(args)
            if extra and isinstance(extra[0], int) and extra[0] == nc:
                extra = extra[1:]
            reg_max = d.get("reg_max", 1)
            cls_name = getattr(head_cls, "__name__", "")
            if "Depth" in cls_name or "Semantic" in cls_name:
                pass  # single output, no extra args
            elif "Segment" in cls_name:
                if extra:
                    kwargs["nm"] = extra[0]
                if len(extra) > 1:
                    kwargs["npr"] = make_divisible(int(extra[1]) * width, 8)
                if len(extra) > 2:
                    reg_max = extra[2]
            elif "OBB" in cls_name:
                if extra:
                    kwargs["ne"] = extra[0]
                if len(extra) > 1:
                    reg_max = extra[1]
                kwargs["layout"] = d.get("obb_layout", "upstream")
                kwargs["legacy"] = d.get("legacy", True)
            elif "Pose" in cls_name:
                if d.get("kpt_shape"):
                    kwargs["kpt_shape"] = tuple(d["kpt_shape"])
                elif extra:
                    kwargs["kpt_shape"] = extra[0]
                if len(extra) > 1:
                    reg_max = extra[1]
            else:  # Detect family
                if extra:
                    reg_max = extra[0]
            if module_name == "PlateDetect":
                kwargs.pop("reg_max", None)
                if d.get("anchors"):
                    kwargs["anchors"] = d["anchors"]
                elif extra:
                    kwargs["anchors"] = extra[0]
                kwargs["kpt_label"] = int(d.get("kpt_label", 4))
                layer = head_cls(nc, ch=ch_list, **kwargs)
            else:
                kwargs["reg_max"] = int(reg_max)
                layer = head_cls(nc, ch=ch_list, **kwargs)
            this_c2 = getattr(layer, "no", None) or (4 + nc)

        elif module_name == "Classify":
            this_c2 = int(args[0]) if args else nc
            layer = Classify(c1, this_c2)
        elif module_name == "CBLinear":
            outs = [make_divisible(int(o) * width, 8) for o in args[0]]
            layer = module_cls(c1, outs)
            this_c2 = int(sum(outs))
        elif module_name == "CBFuse":
            layer = module_cls(*args)
            this_c2 = c1
        elif module_name in SCALED:
            out_ch = make_divisible(min(int(args[0]), max_channels) * width, 8)
            args[0] = out_ch
            if module_name in REPEAT_MODULES:
                # insert the repeat count where the module expects it
                args = list(args)
                args.insert(1, max(round(n * depth), 1) if n > 1 else 1)
                n = 1
            layer = module_cls(c1, *args)
            this_c2 = out_ch
        elif module_name in ("nn.ConvTranspose2d", "nn.Conv2d"):
            layer = module_cls(c1, *args)
            this_c2 = int(args[0]) if args else c1
        else:
            layer = module_cls(*args) if args else module_cls()
            this_c2 = c1

        n = max(round(n * depth), 1) if isinstance(n, int) and not isinstance(n, bool) and n > 1 else n
        if n and n > 1 and not isinstance(layer, Classify):
            layer = nn.Sequential(*(_clone(layer) for _ in range(n)))

        layer.i = i
        layer.f = f
        layers.append(layer)
        save.extend((x if x >= 0 else i + x) for x in f_list if x != -1 and i + x >= 0)
        layer_ch.append(this_c2)
        if verbose:
            print("  {:>3} {:<14} {:<28} -> {}".format(i, str(f), module_name, this_c2))

    return nn.Sequential(*layers), sorted(set(save)), layer_ch[-1]


def _clone(layer):
    import copy

    return copy.deepcopy(layer)


def task_head(model):
    """Return the real task head of a graph model (`model.model[-1]`).

    Hand-written models expose `model.head`; YAML graph models wrap the layers
    in `model.model` (an `nn.Sequential`) whose last element is the head, and
    their `.head` attribute must not be used for head introspection.
    """
    if model is None:
        return None
    inner = getattr(model, "model", None)
    if isinstance(inner, (nn.ModuleList, nn.Sequential)) and len(inner):
        last = inner[-1]
        if any(hasattr(last, a) for a in ("nl", "no", "nc", "end2end")):
            return last
    head = getattr(model, "head", None)
    if isinstance(head, nn.Module):
        return head
    return model


class GraphModel(nn.Module):
    """Executes a parsed layer graph (with saved intermediate outputs)."""

    def __init__(self, model, save, yaml_spec, nc=80, ch=3):
        super().__init__()
        self.model = model
        self.save = list(save)
        self.yaml = yaml_spec
        self.nc = int(nc)
        self.ch = int(ch)
        self.stride = torch.zeros(3)
        # heads we recognise (for stride computation / fuse)
        self._heads = [m for m in self.model if _module_name(type(m).__name__) in HEAD_CLASSES]
        self._compute_stride()

    # ------------------------------------------------------------------ utils
    @property
    def head(self):
        """So ``model.head.num_classes`` works like our hand-written models."""
        return self

    @property
    def num_classes(self):
        return self.nc

    @property
    def nm(self):
        for m in self._heads:
            if hasattr(m, "nm"):
                return int(m.nm)
        return 32

    @property
    def reg_max(self):
        for m in self._heads:
            return int(getattr(m, "reg_max", 1))
        return 1

    @property
    def reg_layout(self):
        for m in self._heads:
            return getattr(m, "reg_layout", "ltrb_angle")
        return "ltrb_angle"

    @property
    def ne(self):
        for m in self._heads:
            return int(getattr(m, "ne", 1))
        return 1

    def _forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f] if m.f >= 0 else y[m.i + m.f]
                else:
                    x = [x if j == -1 else (y[j] if j >= 0 else y[m.i + j]) for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def _compute_stride(self):
        was_training = self.training
        try:
            s = 256
            self.eval()
            with torch.no_grad():
                out = self._forward_once(torch.zeros(1, self.ch, s, s))
            feats = out[0] if isinstance(out, tuple) else out
            if not isinstance(feats, (list, tuple)) and isinstance(out, (tuple, list)):
                # heads may return ``(decoded, per_level_maps)``
                for item in out:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item)
                        and torch.is_tensor(item[0])
                    ):
                        feats = item
                        break
            if isinstance(feats, (list, tuple)) and len(feats):
                self.stride = torch.tensor(
                    [s / float(f.shape[-2]) for f in feats], dtype=torch.float32
                )
                for m in self._heads:
                    if hasattr(m, "stride"):
                        m.stride = self.stride
                    if hasattr(m, "normalize_anchors"):
                        m.normalize_anchors()
                for m in self.model:
                    if hasattr(m, "stride") and not m.stride.any():
                        m.stride = self.stride
        except Exception:
            self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.train(was_training)

    def fuse(self):
        """Merge Conv+BN in every ``Conv`` (inference speed / cleaner graphs)."""
        from pytorchx.nn.modules import Conv as _Conv, _bn_conv_fuse

        count = 0
        for m in self.modules():
            if isinstance(m, _Conv) and hasattr(m, "bn"):
                try:
                    fused = _bn_conv_fuse(m.conv, m.bn)
                    m.conv = fused
                    m.bn = nn.Identity()
                    m.forward = m.forward_fuse
                    count += 1
                except Exception:
                    pass
        return count

    def forward(self, x):
        out = self._forward_once(x)
        if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[1]):
            # segment head -> ``{"feats": [...], "protos": ...}`` (what our task
            # adapters / losses expect).  ``(one2many, one2one)`` from an
            # end-to-end detection head is left untouched.
            feats, protos = out
            return {"feats": feats, "protos": protos}
        return out


def _load_yaml_spec(arch):
    if arch.get("yaml_text"):
        return yaml.safe_load(arch["yaml_text"])
    path = arch.get("yaml_file")
    if not path:
        raise ValueError("Architecture needs 'yaml_file' or 'yaml_text'")
    if not os.path.isabs(path):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(root, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_from_arch(arch):
    """Build a graph model from ``Architecture`` (``yaml_file``/``yaml_text``)."""
    spec = _load_yaml_spec(arch)
    head = arch.get("Head") or {}
    nc = head.get("num_classes") or arch.get("nc") or spec.get("nc", 80)
    spec["nc"] = int(nc)
    if head.get("kpt_shape"):
        spec["kpt_shape"] = list(head["kpt_shape"])
    if head.get("reg_max"):
        spec["reg_max"] = int(head["reg_max"])
    if head.get("reg_layout"):
        spec["obb_layout"] = str(head["reg_layout"])
    if head.get("legacy") is not None:
        spec["legacy"] = bool(head["legacy"])
    spec["scale"] = arch.get("scale", spec.get("scale", "n"))
    ch = int(arch.get("in_channels", arch.get("ch", 3)))
    model, save, last = parse_model(spec, ch=ch)
    _set_bn_ultralytics(model)
    return GraphModel(model, save, spec, nc=spec["nc"], ch=ch)


def _set_bn_ultralytics(model):
    """Align BN momentum/eps with ultralytics ``initialize_weights`` (0.03 / 1e-3)."""
    import torch.nn as _nn

    for m in model.modules():
        if isinstance(m, _nn.BatchNorm2d):
            m.momentum = 0.03
            m.eps = 1e-3
