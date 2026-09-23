"""Convert upstream plate checkpoints into plain PyTorch ``state_dict`` files.

The upstream project (``we0091234/Chinese_license_plate_detection_recognition``)
ships two kinds of checkpoints:

* ``weights/plate_detect.pt``  - a pickled YOLOv5 model object (needs their
  ``models.*`` classes to unpickle),
* ``weights/plate_rec_color.pth`` - ``{'cfg': [...], 'state_dict': {...}}``.

Both are mapped onto our own classes (``torchkiln.nn.plate``) and written as
``{'state_dict': ..., 'cfg': ...}`` so ``Global.pretrained_model`` can load them
with the usual name+shape matching.

Usage::

    python tools/convert/convert_plate_weights.py \
        --detect _downloads/Chinese_license_plate_detection_recognition/weights/plate_detect.pt \
        --rec    _downloads/Chinese_license_plate_detection_recognition/weights/plate_rec_color.pth \
        --out    _downloads/plate
"""

import argparse
import os
import pickle
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _class_map():
    import torch.nn as nn

    from torchkiln.nn import plate as P
    from torchkiln.nn.modules import C3, Concat, Conv

    class _Model(nn.Module):
        """Placeholder for upstream ``models.yolo.Model``."""

    return {
        "Conv": Conv,
        "Concat": Concat,
        "C3": P.C3V5,
        "StemBlock": P.StemBlock,
        "ShuffleV2Block": P.ShuffleV2Block,
        "Bottleneck": P.BottleneckV5,
        "Detect": P.PlateDetect,
        "Model": _Model,
    }


def load_upstream_state(path):
    """``torch.load`` an upstream checkpoint whose classes are not importable."""
    from torchkiln.nn import plate as _plate  # noqa: F401  (registers modules)

    class _Unpickler(pickle.Unpickler):
        MAP = _class_map()

        def find_class(self, module, name):
            if name in self.MAP:
                return self.MAP[name]
            if module.split(".")[0] in ("models", "utils"):
                raise pickle.UnpicklingError(
                    "unknown upstream class {}.{}".format(module, name)
                )
            return super().find_class(module, name)

    shim = type(sys)("plate_pickle_shim")
    shim.Unpickler = _Unpickler
    shim.load = pickle.load
    obj = torch.load(path, map_location="cpu", pickle_module=shim, weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"].state_dict(), obj
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    return obj.state_dict(), {}


def convert(path, out_path):
    sd, ck = load_upstream_state(path)
    payload = {"state_dict": sd}
    if ck.get("cfg") is not None:
        payload["cfg"] = ck["cfg"]
    if ck.get("epoch") is not None:
        payload["epoch"] = ck["epoch"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)
    n = sum(1 for v in sd.values() if hasattr(v, "numel"))
    total = sum(int(v.numel()) for v in sd.values() if hasattr(v, "numel"))
    print("{} -> {} ({} tensors, {:,} values)".format(path, out_path, n, total))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", default=None, help="upstream plate_detect.pt")
    ap.add_argument("--rec", default=None, help="upstream plate_rec_color.pth")
    ap.add_argument("--out", default=os.path.join(ROOT, "_downloads", "plate"))
    args = ap.parse_args()
    if args.detect:
        convert(args.detect, os.path.join(args.out, "plate_detect.pth"))
    if args.rec:
        convert(args.rec, os.path.join(args.out, "plate_rec_color.pth"))
    if not (args.detect or args.rec):
        print("nothing to do (pass --detect and/or --rec)")


if __name__ == "__main__":
    main()
