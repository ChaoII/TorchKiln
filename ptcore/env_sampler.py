"""跨平台环境采样器：CPU / 内存 / 磁盘 / 网络 + GPU。

供简单模式训练循环按间隔采样（``maybe_sample`` 节流），采样结果作为
``env`` 事件的 fields（事件层的 type/seq/ts 注入不在本模块职责内）。

数据源：
- CPU / 内存 / 磁盘 / 网络：``psutil``（硬依赖，跨平台，无 Linux-only API）；
- GPU：``torch.cuda`` 为主（is_available / device_count / mem_get_info /
  memory_allocated / memory_reserved），``pynvml`` 仅在可选安装时增强
  （util / temp / power），未安装时这些字段为 None；
- 磁盘 / 网络速率：对 ``psutil.disk_io_counters()`` / ``psutil.net_io_counters()``
  的累计值取差值除以时间差；首次采样记基线并返回 0.0。

健壮性约定：任何单项采样失败都降级为空字段 / 保守值并继续，绝不向调用方上抛异常；
无 CUDA 时 ``gpu == []``。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import psutil

__all__ = ["EnvSampler"]

_MB = 1024 * 1024


def _default_disk_path() -> str:
    """默认采样磁盘所在路径：Windows 取 cwd 所在盘符，POSIX 取 /。"""
    if os.name == "nt":
        return os.environ.get("SystemDrive", "C:") + os.sep
    return "/"


class EnvSampler:
    """环境采样器。

    :param interval: ``maybe_sample`` 的最小采样间隔（秒），默认 5.0。
    :param device_indices: 仅采样指定 CUDA 卡号列表；None 表示全部可见卡。
    """

    def __init__(self, interval: float = 5.0, device_indices: list[int] | None = None) -> None:
        self._interval = float(interval)
        self._device_indices = [int(i) for i in device_indices] if device_indices is not None else None
        self._last_sample: float | None = None
        # (monotonic 时间戳, 累计读字节, 累计写字节)；None = 尚未记基线
        self._disk_prev: tuple[float, int, int] | None = None
        # (monotonic 时间戳, 累计收字节, 累计发字节)；None = 尚未记基线
        self._net_prev: tuple[float, int, int] | None = None
        self._nvml_initialized = False
        self._nvml: Any | None = None  # pynvml 模块或 None（惰性初始化，失败不再重试）

    # ------------------------------------------------------------------
    # 采样入口
    # ------------------------------------------------------------------

    def sample(self) -> dict:
        """立即采样一次，返回 env 事件字段 dict（**不含 type/seq/ts**）。"""
        return {
            "gpu": self._sample_gpu(),
            "cpu": self._sample_cpu(),
            "cpu_per_core": self._sample_cpu_per_core(),
            "mem": self._sample_mem(),
            "disk": self._sample_disk(),
            "net": self._sample_net(),
        }

    def maybe_sample(self) -> dict | None:
        """按 ``interval`` 节流：距上次采样（``time.monotonic``）不足 interval 返回 None。"""
        now = time.monotonic()
        if self._last_sample is not None and now - self._last_sample < self._interval:
            return None
        self._last_sample = now
        return self.sample()

    def close(self) -> None:
        """关闭 pynvml 句柄（若已打开）；可安全重复调用。"""
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                logging.debug("nvmlShutdown 失败，忽略", exc_info=True)
            self._nvml = None

    # ------------------------------------------------------------------
    # 各项采样（每项独立 try/except，失败降级不上抛）
    # ------------------------------------------------------------------

    def _sample_cpu(self) -> float:
        try:
            return float(psutil.cpu_percent(None))
        except Exception:
            logging.debug("CPU 采样失败，降级为 0", exc_info=True)
            return 0.0

    def _sample_cpu_per_core(self) -> list[float]:
        try:
            return [float(v) for v in psutil.cpu_percent(None, percpu=True)]
        except Exception:
            logging.debug("逐核 CPU 采样失败，降级为空列表", exc_info=True)
            return []

    def _sample_mem(self) -> dict:
        try:
            vm = psutil.virtual_memory()
            return {"used_mb": vm.used / _MB, "total_mb": vm.total / _MB}
        except Exception:
            logging.debug("内存采样失败，降级为 0", exc_info=True)
            return {"used_mb": 0.0, "total_mb": 0.0}

    def _sample_disk(self) -> dict:
        read_mb_s = write_mb_s = free_mb = 0.0
        try:
            free_mb = psutil.disk_usage(_default_disk_path()).free / _MB
        except Exception:
            logging.debug("disk_usage 采样失败，free_mb 降级为 0", exc_info=True)
        try:
            c = psutil.disk_io_counters()
            if c is not None:
                now = time.monotonic()
                prev = self._disk_prev
                if prev is not None:
                    dt = now - prev[0]
                    if dt > 0:
                        read_mb_s = max(0.0, (c.read_bytes - prev[1]) / _MB / dt)
                        write_mb_s = max(0.0, (c.write_bytes - prev[2]) / _MB / dt)
                self._disk_prev = (now, c.read_bytes, c.write_bytes)
        except Exception:
            logging.debug("disk_io_counters 采样失败，速率降级为 0", exc_info=True)
            read_mb_s = write_mb_s = 0.0
        return {"read_mb_s": read_mb_s, "write_mb_s": write_mb_s, "free_mb": free_mb}

    def _sample_net(self) -> dict:
        rx_mb_s = tx_mb_s = 0.0
        try:
            c = psutil.net_io_counters()
            if c is not None:
                now = time.monotonic()
                prev = self._net_prev
                if prev is not None:
                    dt = now - prev[0]
                    if dt > 0:
                        rx_mb_s = max(0.0, (c.bytes_recv - prev[1]) / _MB / dt)
                        tx_mb_s = max(0.0, (c.bytes_sent - prev[2]) / _MB / dt)
                self._net_prev = (now, c.bytes_recv, c.bytes_sent)
        except Exception:
            logging.debug("net_io_counters 采样失败，速率降级为 0", exc_info=True)
            rx_mb_s = tx_mb_s = 0.0
        return {"rx_mb_s": rx_mb_s, "tx_mb_s": tx_mb_s}

    def _sample_gpu(self) -> list[dict]:
        try:
            import torch
        except Exception:
            return []
        try:
            if not torch.cuda.is_available():
                return []
            if self._device_indices is not None:
                indices = self._device_indices
            else:
                indices = list(range(torch.cuda.device_count()))
        except Exception:
            logging.debug("torch.cuda 可用性检查失败，gpu 降级为 []", exc_info=True)
            return []
        out = []
        for idx in indices:
            entry: dict = {
                "id": idx,
                "util": None,
                "mem_reserved_mb": 0.0,
                "mem_allocated_mb": 0.0,
                "mem_total_mb": None,
                "temp_c": None,
                "power_w": None,
            }
            try:
                _free, total = torch.cuda.mem_get_info(idx)
                entry["mem_total_mb"] = total / _MB
                entry["mem_allocated_mb"] = torch.cuda.memory_allocated(idx) / _MB
                entry["mem_reserved_mb"] = torch.cuda.memory_reserved(idx) / _MB
            except Exception:
                logging.debug("torch.cuda 卡 %s 采样失败，内存字段降级", idx, exc_info=True)
            self._enrich_with_nvml(idx, entry)
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # pynvml（可选依赖，仅增强 util/temp/power）
    # ------------------------------------------------------------------

    def _get_nvml(self):
        """惰性初始化 pynvml（可选依赖）：成功返回模块，否则返回 None。

        只在首次采样时尝试 import/初始化，失败后不再重试；
        任何异常（ImportError / nvmlInit 失败）都吞掉并降级。
        """
        if self._nvml_initialized:
            return self._nvml
        self._nvml_initialized = True
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            logging.debug("pynvml 不可用（未安装或初始化失败），GPU util/temp/power 将为 None",
                          exc_info=True)
            self._nvml = None
        return self._nvml

    def _enrich_with_nvml(self, idx: int, entry: dict) -> None:
        """有 pynvml 时补齐 util/temp/power；单项失败保持 None。"""
        pynvml = self._get_nvml()
        if pynvml is None:
            return
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            entry["util"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            entry["temp_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            entry["power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            logging.debug("pynvml 卡 %s 采样失败，util/temp/power 保持 None", idx, exc_info=True)
