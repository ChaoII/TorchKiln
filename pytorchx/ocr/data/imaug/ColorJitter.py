# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Color jitter operator, PyTorch/numpy port of PaddleOCR's ColorJitter.

Paddle implements ColorJitter by building Brightness/Contrast/Saturation/Hue
sub-transforms, randomly shuffling their order, and applying them. This port
reproduces the same factor ranges and random ordering using PIL.
"""
import random

import numpy as np
from PIL import Image, ImageEnhance

__all__ = ["ColorJitter"]


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, **kwargs):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _order(self):
        ops = []
        if self.brightness is not None:
            ops.append("brightness")
        if self.contrast is not None:
            ops.append("contrast")
        if self.saturation is not None:
            ops.append("saturation")
        if self.hue is not None:
            ops.append("hue")
        random.shuffle(ops)
        return ops

    def __call__(self, data):
        image = data["image"]
        img = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        for op in self._order():
            if op == "brightness":
                factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
                img = ImageEnhance.Brightness(img).enhance(factor)
            elif op == "contrast":
                factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
                img = ImageEnhance.Contrast(img).enhance(factor)
            elif op == "saturation":
                factor = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
                img = ImageEnhance.Color(img).enhance(factor)
            elif op == "hue":
                factor = random.uniform(-self.hue, self.hue)
                delta = int(round(factor * 255.0))
                if delta != 0:
                    # Shift hue on the single H channel only; avoids casting the
                    # whole image to float32 (which is huge for 1080p+ images).
                    hsv = np.array(img.convert("HSV"), dtype=np.uint8)
                    ch = hsv[:, :, 0].astype(np.int16)
                    ch += delta
                    np.mod(ch, 255, out=ch)
                    hsv[:, :, 0] = ch.astype(np.uint8)
                    img = Image.fromarray(hsv, mode="HSV").convert("RGB")
        try:
            import cv2

            data["image"] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            data["image"] = np.ascontiguousarray(np.array(img)[:, :, ::-1])
        return data
