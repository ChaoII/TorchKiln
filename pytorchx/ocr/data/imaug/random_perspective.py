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
"""RandomPerspective operator (numpy/cv2 port of PaddleOCR's implementation)."""
import math
import random

import cv2
import numpy as np

__all__ = ["RandomPerspective"]


class RandomPerspective(object):
    """Random perspective transform for OCR detection training.

    Operates on data["image"] (H,W,C) and data["polys"] (N, P, 2).
    """

    def __init__(
        self,
        prob=0.3,
        degrees=0.0,
        scale=0.2,
        shear=5.0,
        perspective=0.0002,
        fit_output=True,
        fill_value=(123.675, 116.28, 103.53),
        min_area_ratio=0.1,
        **kwargs,
    ):
        self.prob = prob
        self.degrees = degrees
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.fit_output = fit_output
        self.min_area_ratio = min_area_ratio
        if isinstance(fill_value, (int, float)):
            fill_value = (fill_value,) * 3
        self.fill_value = tuple(fill_value)

    def __call__(self, data):
        if random.random() > self.prob:
            return data

        im = data["image"]
        h, w = im.shape[:2]
        M_core = self._get_core_matrix(h, w)

        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, M_core)
        x_min, y_min = warped_corners.min(axis=0).ravel()
        x_max, y_max = warped_corners.max(axis=0).ravel()

        if self.fit_output:
            new_w = int(np.ceil(x_max) - np.floor(x_min))
            new_h = int(np.ceil(y_max) - np.floor(y_min))
            T_fit = np.eye(3, dtype=np.float32)
            T_fit[0, 2] = -np.floor(x_min)
            T_fit[1, 2] = -np.floor(y_min)
            M = T_fit @ M_core
            target_size = (new_w, new_h)
        else:
            T_orig = np.eye(3, dtype=np.float32)
            T_orig[0, 2] = w / 2.0
            T_orig[1, 2] = h / 2.0
            M = T_orig @ M_core
            target_size = (w, h)

        transformed_im = cv2.warpPerspective(
            im,
            M,
            target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.fill_value,
        )
        data["image"] = transformed_im

        polys = data["polys"]
        ignore_tags = data["ignore_tags"]
        texts = data["texts"]

        if len(polys) > 0:
            n = len(polys)
            points_per_poly = polys[0].shape[0] if hasattr(polys[0], "shape") else len(polys[0])
            all_points = np.array(polys, dtype=np.float32).reshape(-1, 1, 2)
            warped_pts = cv2.perspectiveTransform(all_points, M)
            warped_pts = warped_pts.reshape(n, points_per_poly, 2)

            orig_areas = np.array(
                [cv2.contourArea(p.astype(np.float32)) for p in np.array(polys, dtype=np.float32)]
            )
            new_areas = np.array([cv2.contourArea(p.astype(np.float32)) for p in warped_pts])

            tw, th = target_size
            centers_x = warped_pts[:, :, 0].mean(axis=1)
            centers_y = warped_pts[:, :, 1].mean(axis=1)
            valid = (
                (new_areas > orig_areas * self.min_area_ratio)
                & (centers_x > 0)
                & (centers_x < tw)
                & (centers_y > 0)
                & (centers_y < th)
            )
            valid_ids = np.where(valid)[0]
            data["polys"] = warped_pts[valid_ids]
            data["ignore_tags"] = [ignore_tags[i] for i in valid_ids]
            data["texts"] = [texts[i] for i in valid_ids]

        return data

    def _get_core_matrix(self, h, w):
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        ref_size = 640.0
        max_dim = max(h, w)
        p_normalized = self.perspective * (ref_size / max_dim)

        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-p_normalized, p_normalized)
        P[2, 1] = random.uniform(-p_normalized, p_normalized)

        s = random.uniform(1 - self.scale, 1 + self.scale)
        a = random.uniform(-self.degrees, self.degrees)
        R = np.eye(3, dtype=np.float32)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        return S @ R @ P @ C
