"""Shared helpers for task adapters.

* ``num_classes_of`` is re-exported from :mod:`pytorchx.tasks._cls` (single source).
* ``_class_weights_from_config`` builds ultralytics-style inverse-frequency class
  weights for detection losses.
"""
from __future__ import absolute_import

from pytorchx.tasks._cls import num_classes_of  # noqa: F401

__all__ = ["num_classes_of", "_class_weights_from_config"]


def _class_weights_from_config(config):
    """逆频率类别权重(ultralytics ``labels_to_class_weights``):总实例 / 每类实例,零类为 1。"""
    import os as _os

    ds = ((config.get("Train") or {}).get("dataset")) or {}
    data_dir = ds.get("data_dir", "") or ""
    counts = {}
    for lf in ds.get("label_file_list") or []:
        if not _os.path.isfile(lf):
            continue
        for line in open(lf, encoding="utf-8"):
            p = line.split()
            if len(p) < 2:
                continue
            img = p[0].replace(chr(92), "/")
            stem = _os.path.splitext(img)[0]
            cands = []
            parts = stem.split("/")
            if "images" in parts:
                q = list(parts)
                q[q.index("images")] = "labels"
                cands.append("/".join(q) + ".txt")
            cands.append(stem + ".txt")
            found = False
            for c in cands:
                fp = _os.path.join(data_dir, c)
                if _os.path.isfile(fp):
                    for ln in open(fp, encoding="utf-8"):
                        tk = ln.split()
                        if tk:
                            cls = int(float(tk[0]))
                            counts[cls] = counts.get(cls, 0) + 1
                    found = True
                    break
            if not found:
                cls = int(float(p[1]))
                counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return None
    total = float(sum(counts.values()))
    nc = max(counts) + 1
    return [total / counts[c] if counts.get(c, 0) else 1.0 for c in range(nc)]
