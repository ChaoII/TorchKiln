"""Simplified RandAugment for the attribute models (PaddleClas ``TimmAutoAugment``).

PaddleX uses ``TimmAutoAugment(config_str="rand-m9-mstd0.5-inc1", prob=0.8)``:

* ``m9``      -> magnitude 9 (of 30)
* ``mstd0.5`` -> per-sample magnitude jitter of 0.5
* ``inc1``    -> magnitude grows by 1 every epoch
* ``prob 0.8``-> the whole policy is applied with 80% probability

The operations themselves follow timm's RandAugment list; the per-op level
tables are simplified to linear interpolation of the ``magnitude / 30`` ratio,
which is the PaddleClas/Timm behaviour for the continuous ops and a close
approximation for the discrete ones (documented deviation).
"""
from __future__ import absolute_import

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

__all__ = ["RandAugment"]


def _autocontrast(img, m):
    return ImageOps.autocontrast(img)


def _equalize(img, m):
    return ImageOps.equalize(img)


def _posterize(img, m):
    bits = int(round(8 - 4 * m))
    return ImageOps.posterize(img, max(1, min(8, bits)))


def _solarize(img, m):
    return ImageOps.solarize(img, int(256 - 256 * m))


def _color(img, m):
    return ImageEnhance.Color(img).enhance(1.0 + m)


def _contrast(img, m):
    return ImageEnhance.Contrast(img).enhance(1.0 + m)


def _brightness(img, m):
    return ImageEnhance.Brightness(img).enhance(1.0 + m)


def _sharpness(img, m):
    return ImageEnhance.Sharpness(img).enhance(1.0 + m)


def _rotate(img, m):
    return img.rotate(30.0 * m, resample=Image.BICUBIC, expand=False)


def _shear_x(img, m):
    return img.transform(img.size, Image.AFFINE, (1, m, 0, 0, 1, 0), resample=Image.BICUBIC)


def _shear_y(img, m):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, m, 1, 0), resample=Image.BICUBIC)


def _translate_x(img, m):
    px = int(m * img.size[0])
    return img.transform(img.size, Image.AFFINE, (1, 0, px, 0, 1, 0), resample=Image.BICUBIC)


def _translate_y(img, m):
    px = int(m * img.size[1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, px), resample=Image.BICUBIC)


class RandAugment(object):
    """``rand-m{magnitude}-mstd{magnitude_std}-inc{inc}`` with ``prob``."""

    OPS = [
        _autocontrast, _equalize, _posterize, _solarize, _color, _contrast,
        _brightness, _sharpness, _rotate, _shear_x, _shear_y,
        _translate_x, _translate_y,
    ]

    def __init__(self, p=0.8, num_ops=2, magnitude=9, magnitude_std=0.5, inc=1.0, **kwargs):
        self.p = float(p)
        self.num_ops = int(num_ops)
        self.magnitude = float(magnitude)
        self.magnitude_std = float(magnitude_std)
        self.inc = float(inc)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _magnitude(self):
        mag = self.magnitude + self.inc * self.epoch
        if self.magnitude_std > 0:
            mag += float(np.random.normal(0, self.magnitude_std))
        return float(np.clip(mag, 0, 30)) / 30.0

    def __call__(self, img):
        """``img``: HWC uint8 RGB numpy -> same shape."""
        if self.p <= 0 or np.random.rand() > self.p:
            return img
        pil = Image.fromarray(img)
        for _ in range(self.num_ops):
            op = self.OPS[np.random.randint(len(self.OPS))]
            try:
                pil = op(pil, self._magnitude())
            except Exception:
                pass
        return np.asarray(pil.convert("RGB"), dtype=img.dtype)
