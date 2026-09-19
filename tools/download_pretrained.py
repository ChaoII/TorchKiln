"""Download pretrained weights from ModelScope into the local cache.

Usage:
    python tools/download_pretrained.py                 # put all into the cache
    python tools/download_pretrained.py PP-OCRv6_tiny_det PP-OCRv5_mobile_rec
    python tools/download_pretrained.py --list          # show where each name resolves

The cache lives in ``~/.pytorchx/ocr/pretrained`` (override with
``PYTORCHOCR_HOME`` / ``PYTORCHOCR_PRETRAINED_DIR``).  Unlike training, this
tool always fills the *cache*: it will not silently reuse the weights that may
already sit in ``_downloads/official`` inside this repo.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from pytorchx.ocr.utils.pretrained import (  # noqa: E402
    download_pretrained,
    pretrained_dir,
    resolve_pretrained,
)

ALL_MODELS = [
    "PP-OCRv3_mobile_det",
    "PP-OCRv3_server_det",
    "PP-OCRv4_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_server_det",
    "PP-OCRv6_tiny_det",
    "PP-OCRv6_small_det",
    "PP-OCRv6_medium_det",
    "PP-OCRv3_mobile_rec",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_rec",
    "PP-OCRv5_mobile_rec",
    "PP-OCRv5_server_rec",
    "PP-OCRv6_tiny_rec",
    "PP-OCRv6_small_rec",
    "PP-OCRv6_medium_rec",
]


def _filename(name):
    return name if name.endswith(".pth") else name + "_ptocr.pth"


def main():
    parser = argparse.ArgumentParser(description="download pretrained weights")
    parser.add_argument("names", nargs="*", default=None, help="model names")
    parser.add_argument(
        "--list", action="store_true", help="show where each name resolves"
    )
    args = parser.parse_args()

    if args.list:
        print("cache dir:", pretrained_dir())
        for name in ALL_MODELS:
            path = resolve_pretrained(name)
            print("  {:<26s} {}".format(name, path or "-"))
        return

    names = args.names or ALL_MODELS
    ok = 0
    for name in names:
        filename = _filename(name)
        try:
            path = download_pretrained(filename)
            print("OK   {:<26s} {}".format(name, path))
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print("FAIL {:<26s} {}".format(name, exc))
    print("\n{}/{} in cache: {}".format(ok, len(names), pretrained_dir()))


if __name__ == "__main__":
    main()
