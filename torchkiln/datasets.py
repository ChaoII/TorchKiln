"""Dataset downloader for the ModelScope-hosted sample data.

仓库只入库 label 文本;图片打包在 ModelScope(见 ``datasets/manifest.yml``)。

    tkiln data list                 # 查看每个数据集:URL / label / 图片 是否就绪
    tkiln data get plate_det_demo   # 下载 + 解包到 datasets/plate_det_demo/
    tkiln data get --all            # 逐个下载(未填 URL 的会提示)
    tkiln data verify               # 只做就绪性检查(等价于 list)

未填 ``url`` 的条目会退化为 ``<defaults.prefix><name>.zip``(manifest 里已预填前缀),
下载失败会给出"请填写 url"的明确提示。
"""
from __future__ import absolute_import

import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

__all__ = ["load_manifest", "list_datasets", "get_dataset", "data_main"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "datasets", "manifest.yml")
DS_ROOT = os.path.join(ROOT, "datasets")


def load_manifest(path=None):
    import yaml

    p = path or MANIFEST
    if not os.path.isfile(p):
        raise FileNotFoundError("缺少数据集清单: {}".format(p))
    with io.open(p, encoding="utf-8") as f:
        m = yaml.safe_load(f) or {}
    return m.get("defaults") or {}, (m.get("datasets") or {})


def _resolve_url(entry, name, defaults):
    url = entry.get("url")
    if isinstance(url, str) and "://" in url:
        return url
    prefix = (defaults or {}).get("prefix") or ""
    if prefix:
        return prefix.rstrip("/") + "/" + name + ".zip"
    return None


def _status(name, entry, root=None):
    d = os.path.join(root or DS_ROOT, name)
    labels = os.path.isfile(os.path.join(d, "train.txt"))
    img_dir = os.path.join(d, "images")
    n_img = 0
    for dp, _dn, fn in os.walk(img_dir):
        n_img += len([f for f in fn if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))])
    return labels, n_img, d


def list_datasets(path=None):
    defaults, ds = load_manifest(path)
    print("{:<28} {:<12} {:<8} {:<7} {}".format("dataset", "task", "labels", "images", "url"))
    print("-" * 96)
    todo = []
    for name in sorted(ds):
        entry = ds[name] or {}
        labels, n_img, _ = _status(name, entry)
        url = _resolve_url(entry, name, defaults)
        print("{:<28} {:<12} {:<8} {:<7} {}".format(
            name, entry.get("task", "-"), "OK" if labels else "--",
            str(n_img) if n_img else "--", "OK" if url else "待填"))
        if not labels or not n_img:
            todo.append(name)
    if todo:
        print("\n未就绪:{}".format(", ".join(todo)))
        print("下载:tkiln data get <name>(或让 tools/make_demo_data.py 生成本地占位图)")
    return 0


def _download(url, dst, timeout=60):
    def hook(blocks, bsize, total):
        if total <= 0:
            return
        done = min(blocks * bsize, total)
        if blocks % 200 == 0:
            sys.stdout.write("\r  下载中 {:>5.1f}%".format(done * 100.0 / total))
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, reporthook=hook)
    sys.stdout.write("\r  下载完成 {:.1f} MB        \n".format(os.path.getsize(dst) / 1e6))


def _unpack(archive, target):
    tmp = tempfile.mkdtemp(prefix="tkiln_data_")
    try:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        entries = os.listdir(tmp)
        payload = tmp
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])):
            payload = os.path.join(tmp, entries[0])
        os.makedirs(target, exist_ok=True)
        for item in os.listdir(payload):
            src, dst = os.path.join(payload, item), os.path.join(target, item)
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def get_dataset(name, path=None, force=False, root=None):
    defaults, ds = load_manifest(path)
    if name not in ds:
        print("manifest 里没有数据集 {!r};可用:{}".format(name, ", ".join(sorted(ds))))
        return 2
    entry = ds[name] or {}
    target = os.path.join(root or DS_ROOT, name)
    labels, n_img, _ = _status(name, entry, root)
    if labels and n_img and not force:
        print("{} 已就绪(labels OK, {} 张图);需要重下加 --force".format(name, n_img))
        return 0
    url = _resolve_url(entry, name, defaults)
    if not url:
        print("{} 未填 url:请在 datasets/manifest.yml 里补上 ModelScope 地址".format(name))
        return 2
    print("{} <- {}".format(name, url))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tkiln_dl_") as td:
        archive = os.path.join(td, name + ".zip")
        try:
            _download(url, archive)
        except Exception as e:
            print("  下载失败:{}:{}".format(type(e).__name__, str(e)[:120]))
            print("  请确认 manifest 的 url(或 defaults.prefix)已填对")
            return 1
        try:
            _unpack(archive, target)
        except zipfile.BadZipFile:
            print("  解包失败:不是合法 zip(可直接放 .tar/目录,或改用 tools/make_demo_data.py)")
            return 1
    labels, n_img, _ = _status(name, entry, root)
    print("  解包到 {}\n  校验:labels={} images={}".format(
        target, "OK" if labels else "缺失", n_img))
    return 0 if labels and n_img else 1


def data_main(argv):
    argv = list(argv or [])
    cmd = argv[0].lower() if argv else "list"
    rest = argv[1:]
    force = "--force" in rest
    rest = [a for a in rest if a != "--force"]
    if cmd in ("list", "verify", "status"):
        return list_datasets()
    if cmd in ("get", "download", "pull"):
        if not rest or rest[0] == "--all":
            defaults, ds = load_manifest()
            rc = 0
            for name in sorted(ds):
                rc |= get_dataset(name, force=force)
            return rc
        rc = 0
        for name in rest:
            rc |= get_dataset(name, force=force)
        return rc
    print(__doc__)
    return 0
