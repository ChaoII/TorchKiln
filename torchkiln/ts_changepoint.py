"""变点检测 / 概念漂移检测工具（纯统计，无训练、无模型）。

定位：**在线持续数据**场景的轻量监控 —— 不需要 GPU、不需要训练，
流式喂数据即可实时给出"是否发生变化 / 何时发生变化"的告警。

实现 4 个业界常用算法：
  * :class:`CUSUMDetector`        —— **累积和**（工业 SPC 标配，检测均值微小持续偏移）
  * :class:`PageHinkleyDetector`  —— **Page-Hinkley**（在线漂移检测，对流式数据友好）
  * :class:`ADWINDetector`        —— **ADWIN**（自适应窗口，按方差自动调窗；概念漂移常用）
  * :class:`ZScoreDetector`       —— **滚动 z-score**（最简单，检测突变/离群段）

用法（流式）::

    from torchkiln.ts_changepoint import CUSUMDetector
    det = CUSUMDetector(delta=0.5, threshold=5.0)
    for x in stream:
        if det.update(x):
            print("变点 @", det.last_change_index)

用法（离线批量）::

    from torchkiln.ts_changepoint import detect_changepoints
    res = detect_changepoints(arr, method="cusum", threshold=5.0)
    # {"indices": [...], "scores": [...], "method": "cusum"}
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

__all__ = ["CUSUMDetector", "PageHinkleyDetector", "ADWINDetector",
           "ZScoreDetector", "detect_changepoints", "DETECTORS"]


# --------------------------------------------------------------------------- #
# 1) CUSUM
# --------------------------------------------------------------------------- #
class CUSUMDetector:
    """累积和（CUSUM）变点检测。

    ``S_hi = max(0, S_hi + (x - mu) - delta)``、``S_lo = max(0, S_lo - (x - mu) - delta)``
    任一超过 ``threshold`` 即报警（**双侧**：检测均值上/下偏移）。

    Args:
        delta: 松弛量（容忍的正常波动；越大越不敏感）
        threshold: 报警阈值（单位 = 标准差的倍数，需先用 ``init_baseline`` 标定）
        warmup: 前多少点用于估计基线均值/标准差
    """

    name = "cusum"

    def __init__(self, delta: float = 0.5, threshold: float = 5.0,
                 warmup: int = 30, reset_on_change: bool = True):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self.reset_on_change = reset_on_change
        self.reset()

    def reset(self):
        self.n = 0
        self._buf = []
        self._mu = None
        self._sd = None
        self.s_hi = 0.0
        self.s_lo = 0.0
        self.last_change_index: Optional[int] = None
        self.scores: List[float] = []

    def init_baseline(self, values) -> "CUSUMDetector":
        """用一段"正常"数据标定均值/标准差（推荐先调）。"""
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        self._mu = float(np.mean(v))
        self._sd = float(np.std(v)) or 1.0
        return self

    def update(self, x: float) -> bool:
        """喂一个新点；返回**是否触发变点**。"""
        self.n += 1
        if self._mu is None:
            self._buf.append(float(x))
            if len(self._buf) >= self.warmup:
                self._mu = float(np.mean(self._buf))
                self._sd = float(np.std(self._buf)) or 1.0
            self.scores.append(0.0)
            return False
        z = (float(x) - self._mu) / self._sd
        self.s_hi = max(0.0, self.s_hi + z - self.delta)
        self.s_lo = max(0.0, self.s_lo - z - self.delta)
        s = max(self.s_hi, self.s_lo)
        self.scores.append(s)
        if s > self.threshold:
            self.last_change_index = self.n - 1
            if self.reset_on_change:
                self.s_hi = self.s_lo = 0.0
            return True
        return False


# --------------------------------------------------------------------------- #
# 2) Page-Hinkley
# --------------------------------------------------------------------------- #
class PageHinkleyDetector:
    """Page-Hinkley 检验（在线漂移检测）。

    ``m_t = sum(x_i - mean_t - delta)``、``PH = m_t - min(m_)``；
    ``PH > lambda`` 即认为发生漂移。对流式数据友好、计算 O(1)。
    """

    name = "page_hinkley"

    def __init__(self, delta: float = 0.005, threshold: float = 50.0,
                 alpha: float = 1.0):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)          # 均值的自适应更新率
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.m_t = 0.0
        self.m_min = 0.0
        self.m_max = 0.0
        self.last_change_index: Optional[int] = None
        self.scores: List[float] = []

    def update(self, x: float) -> bool:
        self.n += 1
        x = float(x)
        self.mean += (x - self.mean) / self.n
        self.m_t += x - self.mean - self.delta
        self.m_min = min(self.m_min, self.m_t)
        self.m_max = max(self.m_max, self.m_t)
        s = max(self.m_t - self.m_min, self.m_max - self.m_t)
        self.scores.append(s)
        if s > self.threshold:
            self.last_change_index = self.n - 1
            self.m_t = self.m_min = self.m_max = 0.0
            return True
        return False


# --------------------------------------------------------------------------- #
# 3) ADWIN（简化版：自适应窗口 + 方差口径）
# --------------------------------------------------------------------------- #
class ADWINDetector:
    """ADWIN（ADaptive WINdowing）简化实现 —— 自适应窗口检测概念漂移。

    维护一个窗口，不断尝试"切两半"，用 **Hoeffding 界** 判断两侧均值是否显著不同；
    若显著则报漂移并**丢弃旧半段**（窗口自适应变短）。``delta`` 为置信度参数。

    ⚠️ 相对原论文做了简化（单一窗口、二等分切点），适合工程在线监控；
    若要严格复现可用 `river` 的 ADWIN。
    """

    name = "adwin"

    def __init__(self, delta: float = 0.002, max_window: int = 1000,
                 min_window: int = 20):
        self.delta = float(delta)
        self.max_window = int(max_window)
        self.min_window = int(min_window)
        self.reset()

    def reset(self):
        self.n = 0
        self.window: List[float] = []
        self.last_change_index: Optional[int] = None
        self.scores: List[float] = []

    def update(self, x: float) -> bool:
        self.n += 1
        self.window.append(float(x))
        if len(self.window) > self.max_window:
            self.window.pop(0)
        w = np.asarray(self.window, dtype=np.float64)
        L = len(w)
        if L < self.min_window:
            self.scores.append(0.0)
            return False
        best, best_cut = 0.0, -1
        # 扫描若干切点（步长自适应，避免 O(n^2)）
        step = max(1, L // 50)
        for cut in range(self.min_window, L - self.min_window + 1, step):
            a, b = w[:cut], w[cut:]
            na, nb = len(a), len(b)
            m = abs(a.mean() - b.mean())
            m_hoeff = np.sqrt(
                (1.0 / (2 * na) + 1.0 / (2 * nb))
                * np.log(4.0 / self.delta))
            if m > m_hoeff and m > best:
                best, best_cut = m, cut
        self.scores.append(best)
        if best_cut > 0:
            self.last_change_index = self.n - len(w) + best_cut
            self.window = self.window[best_cut:]     # 丢弃旧段（自适应）
            return True
        return False


# --------------------------------------------------------------------------- #
# 4) 滚动 z-score
# --------------------------------------------------------------------------- #
class ZScoreDetector:
    """滚动 z-score —— 检测**突变/离群段**（最简单直观）。

    用最近 ``window`` 点的均值/标准差算 z，``|z| > threshold`` 即报警。
    """

    name = "zscore"

    def __init__(self, window: int = 50, threshold: float = 4.0,
                 two_sided: bool = True):
        self.window = int(window)
        self.threshold = float(threshold)
        self.two_sided = bool(two_sided)
        self.reset()

    def reset(self):
        self.n = 0
        self._buf: List[float] = []
        self.last_change_index: Optional[int] = None
        self.scores: List[float] = []

    def update(self, x: float) -> bool:
        self.n += 1
        self._buf.append(float(x))
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if len(self._buf) < max(5, self.window // 4):
            self.scores.append(0.0)
            return False
        mu = float(np.mean(self._buf))
        sd = float(np.std(self._buf)) or 1.0
        z = (float(x) - mu) / sd
        s = abs(z) if self.two_sided else z
        self.scores.append(s)
        if s > self.threshold:
            self.last_change_index = self.n - 1
            return True
        return False


DETECTORS: Dict[str, type] = {
    "cusum": CUSUMDetector,
    "page_hinkley": PageHinkleyDetector,
    "ph": PageHinkleyDetector,
    "adwin": ADWINDetector,
    "zscore": ZScoreDetector,
}


# --------------------------------------------------------------------------- #
# 离线批量接口
# --------------------------------------------------------------------------- #
def detect_changepoints(x, method: str = "cusum", ref_frac: float = 0.2,
                        **kwargs) -> dict:
    """对一段序列做变点检测，返回变点下标与逐点分数。

    Args:
        x: 1D 数组（或多列时按**列平均**压成 1D）
        method: ``cusum`` / ``page_hinkley`` / ``adwin`` / ``zscore``
        ref_frac: 用前 `ref_frac` 比例的数据标定基线（仅 cusum 用）
        **kwargs: 透传给检测器

    Returns:
        ``{"indices": [...], "scores": np.ndarray, "method": str}``
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.reshape(-1)
    key = str(method).lower()
    if key not in DETECTORS:
        raise ValueError("unknown method %r; 可选 %s" % (method, list(DETECTORS)))
    det = DETECTORS[key](**kwargs)
    if key == "cusum" and ref_frac > 0:
        n_ref = max(1, int(len(arr) * ref_frac))
        det.init_baseline(arr[:n_ref])
    idx = []
    for i, v in enumerate(arr):
        if det.update(float(v)):
            idx.append(i)
    return {"indices": idx, "scores": np.asarray(det.scores), "method": key}
