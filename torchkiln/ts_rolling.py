"""滚动（在线）时序预测接口 —— 面向"持续数据/功率预测"的部署场景。

与离线滑窗（`TSDataset` 一次性切 windows）的区别：
  * :class:`RollingPredictor` 维护一个**滚动缓冲**，来一条新数据就预测一次、
    再把真实值回填进缓冲（one-step / multi-step 两种模式）。
  * 提供 :meth:`predict`（单点）与 :meth:`run_stream`（跑整个流，返回逐点预测 + 指标）。

用法::

    from torchkiln.ts_rolling import RollingPredictor
    rp = RollingPredictor(model, in_chunk_len=96, out_chunk_len=24,
                          feature_cols=["power", "irradiance"], device="cuda:0")
    rp.warmup(history_array)                 # 灌入历史窗口
    y = rp.predict()                         # 预测未来 out_chunk_len 步
    rp.push_actual(next_values)              # 真实值到达 -> 回填缓冲
    # 或一次性跑流：
    res = rp.run_stream(stream_array[:, cols], actuals=stream_array[:, target_idx])
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch

__all__ = ["RollingPredictor"]


class RollingPredictor:
    """滚动预测器（单变量或多变量目标均可）。

    Args:
        model: 训练好的 ts_forecast 模型（输入 dict ``{"past_target": (1,L,D)}``，
            有协变量时还需 ``known_cov_numeric`` 等；见 :meth:`predict` 的 ``extra``）。
        in_chunk_len: 回看窗口长度 ``L``。
        out_chunk_len: 预测步数 ``H``。
        target_idx: 目标列在输入特征中的下标（默认 0）。
        device: ``'cuda:0'`` / ``'cpu'``。
        standardize: ``(mean, std)`` 元组（在**原始尺度**上做 z-score）；
            ``None`` 表示不标准化。预测会**反标准化**回原始尺度。
    """

    def __init__(self, model, in_chunk_len: int, out_chunk_len: int,
                 target_idx: int = 0, device: str = "cpu",
                 standardize=None, n_target_dims: Optional[int] = None):
        self.model = model
        self.L = int(in_chunk_len)
        self.H = int(out_chunk_len)
        self.target_idx = int(target_idx)
        self.device = torch.device(device)
        self.mean, self.std = (standardize if standardize is not None else (None, None))
        self.n_target_dims = n_target_dims
        self.buffer: Optional[np.ndarray] = None      # (>=L, D) 原始尺度
        self.model.eval()

    # ---------------- 缓冲管理 ----------------
    def warmup(self, history: np.ndarray):
        """灌入历史窗口（``(T, D)``，``T >= L``；只保留最近 ``L`` 条）。"""
        history = np.asarray(history, dtype=np.float64)
        if history.ndim == 1:
            history = history[:, None]
        self.buffer = history[-self.L:].copy()
        return self

    def push_actual(self, values: np.ndarray):
        """真实值到达（``(k, D)`` 或 ``(k,)`` 单目标）-> 滚入缓冲。"""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 0:
            v = v.reshape(1, 1)
        elif v.ndim == 1:
            v = v.reshape(-1, 1) if self.buffer.shape[1] == 1 else v[None, :]
        if self.buffer is None:
            self.buffer = v[-self.L:].copy()
        else:
            self.buffer = np.concatenate([self.buffer, v], axis=0)[-self.L:]
        return self

    # ---------------- 预测 ----------------
    def _standardize(self, x):
        if self.mean is None:
            return x
        return (x - self.mean) / np.where(np.asarray(self.std) < 1e-12, 1.0, self.std)

    def _inverse(self, y):
        if self.mean is None:
            return y
        m = np.asarray(self.mean)
        s = np.asarray(self.std)
        return y * np.where(s < 1e-12, 1.0, s) + m

    def predict(self, extra: Optional[dict] = None) -> np.ndarray:
        """用当前缓冲预测未来 ``H`` 步，返回**原始尺度**的 ``(H, n_target)``。

        ``extra``：协变量（与训练时同名，如 ``known_cov_numeric``），
        形状需含 ``(L+H)` 的时间维（known）或 ``L``（observed）；会自动加 batch 维。
        """
        if self.buffer is None or len(self.buffer) < self.L:
            raise RuntimeError("缓冲不足：先 warmup() 或 push_actual() 到 %d 条" % self.L)
        past = self.buffer[-self.L:].copy()
        past_z = self._standardize(past)
        if self.n_target_dims is not None:
            past_z = past_z[:, :self.n_target_dims]
        batch = {"past_target": torch.as_tensor(past_z[None], dtype=torch.float32,
                                                device=self.device)}
        if self.mean is not None:
            m = np.asarray(self.mean)[:self.n_target_dims]
            s = np.asarray(self.std)[:self.n_target_dims]
            batch["_rolling_mean"] = torch.as_tensor(m, dtype=torch.float32,
                                                     device=self.device)
            batch["_rolling_std"] = torch.as_tensor(s, dtype=torch.float32,
                                                    device=self.device)
        if extra:
            for k, v in extra.items():
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == len(past.shape):
                    arr = arr[None]
                batch[k] = torch.as_tensor(arr, device=self.device)
        with torch.no_grad():
            out = self.model(batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
        y = out.detach().cpu().numpy()[0]                  # (H, n_target)
        return self._inverse(y)

    # ---------------- 跑整个流 ----------------
    def run_stream(self, stream: np.ndarray, actuals: Optional[np.ndarray] = None,
                   step: int = 1, verbose: bool = False) -> dict:
        """从 ``stream`` 逐窗预测（``stream`` 为**原始尺度** ``(T, D)``）。

        每次用最近 ``L`` 条预测未来 ``H`` 步，然后推进 ``step`` 步
        （``step=1`` 是最细的滚动；``step=H`` 即不重叠的块预测）。

        返回 ``{"pred": (N,H,n_target), "true": (N,H,n_target), "idx": (N,)}``
        以及可选的指标（MSE/RMSE/nRMSE/MAE）。
        """
        stream = np.asarray(stream, dtype=np.float64)
        if stream.ndim == 1:
            stream = stream[:, None]
        n_t = self.n_target_dims or stream.shape[1]
        preds, trues, idxs = [], [], []
        for i in range(self.L, len(stream), step):
            self.warmup(stream[max(0, i - self.L):i])
            p = self.predict()                              # (H, n_target)
            preds.append(p)
            j0, j1 = i, min(i + self.H, len(stream))
            seg = stream[j0:j1, :n_t]
            if len(seg) < self.H:                          # 尾部补齐
                seg = np.concatenate([seg, np.full((self.H - len(seg), n_t), np.nan)])
            trues.append(seg)
            idxs.append(i)
            if verbose and len(preds) % 20 == 0:
                print("  rolled %d/%d" % (len(preds), (len(stream) - self.L) // step))
        P = np.stack(preds) if preds else np.zeros((0, self.H, n_t))
        T = np.stack(trues) if trues else np.zeros((0, self.H, n_t))
        res = {"pred": P, "true": T, "idx": np.array(idxs)}
        if actuals is not None:
            A = np.asarray(actuals, dtype=np.float64).reshape(-1)
            if len(A) == len(P):
                m = ~np.isnan(A)
                ap, aa = P[m], A[m]
            else:
                m = ~np.isnan(T)
                ap, aa = P[m], T[m]
            if len(ap):
                err = ap - aa
                rmse = float(np.sqrt((err ** 2).mean()))
                rms = float(np.sqrt((aa ** 2).mean()))
                res.update({"MSE": float((err ** 2).mean()),
                            "MAE": float(np.abs(err).mean()),
                            "RMSE": rmse,
                            "nRMSE": rmse / (rms if rms > 1e-12 else 1.0)})
        return res
