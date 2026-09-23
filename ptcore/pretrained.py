"""Resolve pretrained weights by name.

Resolution rules (in order):

1. ``None`` / empty            -> no pretrained weights (train from scratch)
2. an existing file path       -> use it directly
3. ``http(s)://...``           -> download into the local cache
4. a bare name                 -> look up the local cache, then the weights
   shipped inside this repo (``_downloads/official``), and finally download
   from ModelScope.

The local cache lives in a hidden folder under the user home:

    ~/.torchkiln/pretrained/<name>.pth

Override the location with the environment variables ``PYTORCHOCR_HOME`` or
``PYTORCHOCR_PRETRAINED_DIR``.

Accepted names::

    PP-OCRv6_tiny_det
    PP-OCRv6_tiny_det.pth
    yolo11n
    yolo11n.pth
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import time
import urllib.request

__all__ = [
    "resolve_pretrained",
    "pretrained_dir",
    "download_pretrained",
    "available_names",
    "model_url",
]

MODELSCOPE_NAMESPACE = "ChaoII0987"
MODELSCOPE_MODEL = "TorchKiln"
MODELSCOPE_REVISION = "master"
PRETRAINED_SUBDIR = "pretrained"

WEB_BASE = "https://www.modelscope.cn/models/{ns}/{model}".format(
    ns=MODELSCOPE_NAMESPACE, model=MODELSCOPE_MODEL
)
MODEL_URL_TEMPLATE = (
    WEB_BASE + "/resolve/" + MODELSCOPE_REVISION + "/" + PRETRAINED_SUBDIR + "/{filename}"
)

ENV_HOME = "PYTORCHOCR_HOME"
ENV_PRETRAINED_DIR = "PYTORCHOCR_PRETRAINED_DIR"
ENV_AUTO_DOWNLOAD = "PYTORCHOCR_AUTO_DOWNLOAD"
ENV_ALLOW_LOCAL_REPO = "PYTORCHOCR_ALLOW_LOCAL_REPO"

_TIMEOUT = 60
_RETRY_LIMIT = 3
_WAIT_TIMEOUT = 3600


def model_url(filename):
    """Build the ModelScope download URL for a weight file name."""
    if not filename.endswith(".pth"):
        filename += ".pth"
    return MODEL_URL_TEMPLATE.format(filename=filename)


def pretrained_dir():
    """Directory that holds the cached pretrained weights."""
    d = os.environ.get(ENV_PRETRAINED_DIR)
    if not d:
        home = os.environ.get(ENV_HOME)
        if not home:
            home = os.path.join(os.path.expanduser("~"), ".torchkiln")
        d = os.path.join(home, PRETRAINED_SUBDIR)
    return d


def _hide_dir(path):
    """Mark the cache folder as hidden on Windows (best effort)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        FILE_ATTRIBUTE_HIDDEN = 0x02
        ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
    except Exception:  # noqa: BLE001
        pass


def _hide_default_cache():
    """Hide ``~/.torchkiln`` (only when the default location is used)."""
    if os.environ.get(ENV_HOME) or os.environ.get(ENV_PRETRAINED_DIR):
        return
    home = os.path.expanduser("~")
    _hide_dir(os.path.join(home, ".torchkiln"))


def _local_search_dirs():
    dirs = [pretrained_dir()]
    # If the user explicitly points the cache somewhere, do not fall back to
    # the weights shipped inside this repo.
    if os.environ.get(ENV_HOME) or os.environ.get(ENV_PRETRAINED_DIR):
        return dirs
    if os.environ.get(ENV_ALLOW_LOCAL_REPO, "1") == "0":
        return dirs
    # Convenience for the dev checkout: weights next to the repo.
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(
        os.path.join(here, "..", "..", "_downloads", "official")
    )
    if os.path.isdir(repo):
        dirs.append(repo)
    return dirs


def _candidates(name):
    """Filenames to try for a bare model name."""
    n = os.path.basename(str(name))
    if n.endswith(".pth") or n.endswith(".pdparams"):
        return [n]
    return [n + ".pth", n]


def _wait_for_file(dst, logger=None):
    """In distributed runs only rank 0 downloads; the others wait for it."""
    waited = 0
    while not os.path.isfile(dst) and waited < _WAIT_TIMEOUT:
        if waited == 0 and logger is not None:
            logger.info("Waiting for rank 0 to finish downloading %s ...", os.path.basename(dst))
        time.sleep(1)
        waited += 1
    return dst if os.path.isfile(dst) else None


def _download_file(url, dst, logger=None):
    rank = os.environ.get("RANK")
    if rank is not None and rank != "0":
        got = _wait_for_file(dst, logger)
        if got:
            return got
        raise RuntimeError(
            "rank %s waited %ds for %s but it never appeared".format(
                rank, _WAIT_TIMEOUT, dst
            )
        )

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = "{}.part.{}".format(dst, os.getpid())
    req = urllib.request.Request(url, headers={"User-Agent": "torchkiln/ocr/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        bar = None
        try:
            from tqdm import tqdm

            bar = tqdm(total=total or None, unit="B", unit_scale=True, desc="download")
        except Exception:  # noqa: BLE001
            bar = None
        last_pct = -1
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if bar is not None:
                    bar.update(len(chunk))
                elif logger is not None and total:
                    pct = int(done * 100 / total)
                    if pct // 10 != last_pct // 10:
                        last_pct = pct
                        logger.info("downloading %s ... %d%%", os.path.basename(dst), pct)
        if bar is not None:
            bar.close()
    if total and done != total:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise IOError(
            "incomplete download for {} ({} / {} bytes)".format(dst, done, total)
        )
    os.replace(tmp, dst)
    return dst


def download_pretrained(filename, dst=None, logger=None):
    """Download ``filename`` from ModelScope into the cache and return its path."""
    dst = dst or os.path.join(pretrained_dir(), filename)
    if os.path.isfile(dst):
        return dst
    rel = "{}/{}".format(PRETRAINED_SUBDIR, filename)
    urls = [
        model_url(filename),
        "{base}/api/v1/models/{ns}/{model}/repo?Revision={rev}&FilePath={rel}".format(
            base=WEB_BASE,
            ns=MODELSCOPE_NAMESPACE,
            model=MODELSCOPE_MODEL,
            rev=MODELSCOPE_REVISION,
            rel=rel,
        ),
    ]
    if logger is not None:
        logger.info("Pretrained weights not found locally; downloading %s", filename)
    last_err = None
    for url in urls:
        for attempt in range(1, _RETRY_LIMIT + 1):
            try:
                path = _download_file(url, dst, logger)
                _hide_default_cache()
                if logger is not None:
                    logger.info("Saved pretrained weights to %s", path)
                return path
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if logger is not None:
                    logger.warning(
                        "Download attempt %d/%d failed (%s): %s",
                        attempt,
                        _RETRY_LIMIT,
                        url,
                        exc,
                    )
                time.sleep(1)
    raise RuntimeError(
        "Failed to download {} from ModelScope: {}".format(filename, last_err)
    )


def resolve_pretrained(name_or_path, logger=None):
    """Return a local path for ``name_or_path`` (downloading if necessary).

    Returns ``None`` when nothing was given, or when a *name* could not be
    found/downloaded and auto-download is disabled.

    Raises ``FileNotFoundError`` when ``name_or_path`` is clearly a path
    (contains a separator or a drive letter) but the file does not exist -- a
    silent fall-back to "train from scratch" would hide a typo.
    """
    if name_or_path is None:
        return None
    spec = str(name_or_path).strip()
    if spec == "" or spec.lower() in ("none", "null"):
        return None

    if spec.startswith("http://") or spec.startswith("https://"):
        filename = os.path.basename(spec.split("?")[0]) or "pretrained.pth"
        dst = os.path.join(pretrained_dir(), filename)
        if os.path.isfile(dst):
            if logger is not None:
                logger.info("Pretrained weights already cached: %s", dst)
            return dst
        # Reuse a same-named file that is already available locally
        # (e.g. the weights shipped inside this repo).
        for search_dir in _local_search_dirs():
            local = os.path.join(search_dir, filename)
            if os.path.isfile(local) and os.path.abspath(local) != os.path.abspath(dst):
                if logger is not None:
                    logger.info("Using local pretrained weights: %s", local)
                return local
        if os.environ.get(ENV_AUTO_DOWNLOAD, "1") == "0":
            return None
        if logger is not None:
            logger.info("Downloading pretrained weights to %s", dst)
        path = _download_file(spec, dst, logger)
        _hide_default_cache()
        if logger is not None:
            logger.info("Saved pretrained weights to %s", path)
        return path

    if os.path.isfile(spec):
        return spec

    looks_like_path = ("/" in spec) or ("\\" in spec) or (":" in spec)
    if looks_like_path:
        raise FileNotFoundError(
            "Pretrained weights file not found: {}\n"
            "  Pass an existing .pth path, or just the model name "
            "(e.g. PP-OCRv6_tiny_det) to use the cached/auto-downloaded "
            "official weights.".format(spec)
        )

    for d in _local_search_dirs():
        for cand in _candidates(spec):
            path = os.path.join(d, cand)
            if os.path.isfile(path):
                return path

    if os.environ.get(ENV_AUTO_DOWNLOAD, "1") == "0":
        return None

    filename = _candidates(spec)[0]
    try:
        return download_pretrained(filename, logger=logger)
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning(
                "Could not obtain pretrained weights '%s' (%s).", spec, exc
            )
        return None


def available_names():
    """List cached pretrained weight file names."""
    d = pretrained_dir()
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".pth"))


def load_state_dict_any(path):
    """Load a state_dict from *any* checkpoint flavour.

    Handles, in order:
      * plain tensor dict / `{'state_dict': ...}`            (our checkpoints)
      * `{'model': nn.Module, 'ema': ...}`                   (training resumes)
      * upstream ultralytics / yolov5 pickles (`.pt`)        -> `obj['model'].state_dict()`

    Upstream `.pt` files pickle the **whole model object**; the classes are not
    importable here, so they are replaced by placeholder `nn.Module` subclasses
    (torch restores `_parameters/_buffers` from the pickle state) and the tensor
    dict is taken from the reconstructed object.  Extension (`.pt` / `.pth`)
    is irrelevant - only the content matters.
    """
    import pickle
    import types

    import torch

    class _Placeholder(torch.nn.Module):
        # 上游 pickle 可能带参构造(如 ultralytics 的 _load_type),这里全部吞掉:
        # torch 之后仍会从 pickle state 里恢复真实参数/缓冲。
        def __init__(self, *args, **kwargs):
            super().__init__()

        def __call__(self, *args, **kwargs):
            # 上游 pickle 里可能有"占位类被当函数调用"的情况(ultralytics _load_type),
            # 这里直接返回 None;真实模块的参数/缓冲由 torch 从 pickle state 恢复。
            return None

    def _stub(module, name):
        # 上游 .pt 里 pickle 的类(ultralytics.nn.* 等)在本环境不可 import,
        # 用占位 nn.Module 顶替:torch 仍会从 pickle state 里恢复参数/缓冲。
        return type(str(name), (_Placeholder,), {"__module__": str(module)})

    class _AnyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _stub(module, name)

    def _load(p, use_stub):
        if use_stub:
            shim = types.ModuleType("pickle")
            shim.Unpickler = _AnyUnpickler
            shim.load = pickle.load
            return torch.load(
                p, map_location="cpu", pickle_module=shim, weights_only=False
            )
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            return None

    obj = _load(path, use_stub=False)
    if obj is None:
        obj = _load(path, use_stub=True)
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        for key in ("model", "ema"):
            node = obj.get(key)
            if isinstance(node, dict):
                return node
            if hasattr(node, "state_dict"):
                return node.state_dict()
        tensors = {k: v for k, v in obj.items() if hasattr(v, "shape")}
        if tensors:
            return tensors
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError("无法从 {} 解析 state_dict".format(path))
