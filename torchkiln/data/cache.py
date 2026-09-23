"""标签扫描缓存(对齐 ultralytics ``labels.cache``,但只缓存"扫描结果")。

原版 ``labels.cache`` 缓存的是"遍历数据集的产物"(每张图的尺寸、标注解析结果、损坏检查),
用 ``版本号 + 文件哈希(路径/大小/mtime)`` 做失效判断。我们也做同一件事,但按我们的需要裁剪:

* 缓存内容:每个样本的**标注解析结果**(以及扫描时得到的图像尺寸,如果调用方需要);
* 失效条件:``CACHE_VERSION`` 变化、标签文件列表/大小/mtime 变化 → 自动重扫;
* 缓存位置:``<data_dir>/.labels_cache_<split>.pkl``(可被 ``.gitignore`` 忽略);
* 收益:启动时跳过一次全量解析;训练中不再每样本打开小 txt(mosaic 下每步 4 次/样本 → 0)。

用法(数据集里)::

    self._cache = LabelCache(self.data_dir, label_files, split=self.mode, logger=logger,
                             parse=lambda rel: self._parse_one(rel))
    rec = self._cache.get(rel)          # 命中则直接返回解析结果,否则回退到 _parse_one

``parse`` 返回任何可 pickle 的对象(如 ``(boxes, kpts, valid)``);若返回 ``None`` 表示该样本无效。
"""

import hashlib
import io
import os
import pickle
import time

__all__ = ["LabelCache", "CACHE_VERSION"]

CACHE_VERSION = "tkiln-labels-1.0"


class LabelCache(object):
    def __init__(self, data_dir, label_files, split="train", logger=None,
                 parse=None, enabled=True):
        self.data_dir = data_dir or ""
        self.label_files = [str(p) for p in (label_files or [])]
        self.split = split
        self.logger = logger
        self.parse = parse
        self.enabled = bool(enabled and self.label_files)
        self.items = {}
        self.hit = False
        self._dirty = False
        self.path = os.path.join(
            self.data_dir or ".", ".labels_cache_%s.pkl" % split
        )
        if not self.enabled:
            return
        fp = self._fingerprint()
        t0 = time.time()
        if self._load(fp):
            self.hit = True
            if self.logger:
                self.logger.info(
                    "labels.cache hit: %d samples from %s (%.3fs)",
                    len(self.items), os.path.basename(self.path), time.time() - t0,
                )
            return
        self._scan()
        self._save(fp)
        if self.logger:
            self.logger.info(
                "labels.cache miss -> scanned %d samples in %.1fs, saved %s",
                len(self.items), time.time() - t0, os.path.basename(self.path),
            )

    # ---------------------------------------------------------------- inner
    def _fingerprint(self):
        h = hashlib.sha256()
        h.update(CACHE_VERSION.encode())
        h.update((self.data_dir or "").encode())
        for p in self.label_files:
            try:
                st = os.stat(p)
                h.update(("%s|%d|%d" % (p, st.st_size, int(st.st_mtime))).encode())
            except OSError:
                h.update(("%s|missing" % p).encode())
        return h.hexdigest()

    def _load(self, fp):
        if not os.path.isfile(self.path):
            return False
        try:
            with io.open(self.path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("version") != CACHE_VERSION or blob.get("fingerprint") != fp:
                return False
            self.items = blob.get("items") or {}
            return True
        except Exception:
            return False

    def _scan(self):
        """一次遍历:对每个标签文件里的每个样本调用 ``parse``。"""
        self.items = {}
        for lf in self.label_files:
            if not os.path.isfile(lf):
                continue
            with io.open(lf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rel = line.split()[0]
                    try:
                        self.items[rel] = (self.parse(rel) if self.parse else None)
                    except Exception:
                        self.items[rel] = None

    def _save(self, fp):
        if not self.items:
            return
        try:
            tmp = self.path + ".tmp"
            with io.open(tmp, "wb") as f:
                pickle.dump(
                    {"version": CACHE_VERSION, "fingerprint": fp, "items": self.items},
                    f, protocol=4,
                )
            os.replace(tmp, self.path)
        except Exception as e:
            if self.logger:
                self.logger.warning("labels.cache 写入失败(不影响训练): %s", e)

    # ---------------------------------------------------------------- public
    def get(self, rel):
        """返回缓存项;未命中返回 ``(False, None)``。"""
        if self.enabled and rel in self.items:
            return True, self.items[rel]
        return False, None

    def save_now(self, force=False):
        """把当前内存缓存落盘(训练中动态收集的项也能持久化)。"""
        if not self.enabled:
            return False
        if not force and self.hit and not self._dirty:
            return False
        self._save(self._fingerprint())
        self._dirty = False
        self.hit = True
        return True

    def mark_dirty(self):
        self._dirty = True

    def __contains__(self, rel):
        return self.enabled and rel in self.items

    def __len__(self):
        return len(self.items)
