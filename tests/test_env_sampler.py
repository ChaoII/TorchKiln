"""EnvSampler（ptcore/env_sampler.py）测试。

GPU 路径全部用 monkeypatch 模拟，本文件不依赖真实 GPU；
pynvml 的“缺失”场景通过屏蔽 builtins.__import__ 模拟。
"""
from __future__ import annotations

import builtins

from ptcore.env_sampler import EnvSampler


def test_sample_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    s = EnvSampler(interval=0.1)
    data = s.sample()
    assert data["gpu"] == [] and 0 <= data["cpu"] <= 100
    assert "used_mb" in data["mem"] and "rx_mb_s" in data["net"]
    assert isinstance(data["cpu_per_core"], list)
    assert set(data["disk"]) == {"read_mb_s", "write_mb_s", "free_mb"}
    assert set(data["net"]) == {"rx_mb_s", "tx_mb_s"}


def test_maybe_sample_throttles():
    s = EnvSampler(interval=60.0)
    assert s.maybe_sample() is not None
    assert s.maybe_sample() is None          # 60s 内第二次不采


def test_first_sample_rates_zero(monkeypatch):
    """磁盘/网络速率：首次采样记基线返回 0，不抛 KeyError。"""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    s = EnvSampler(interval=0.0)
    data = s.sample()
    assert data["disk"]["read_mb_s"] == 0.0 and data["disk"]["write_mb_s"] == 0.0
    assert data["net"]["rx_mb_s"] == 0.0 and data["net"]["tx_mb_s"] == 0.0
    assert data["disk"]["free_mb"] >= 0
    assert data["mem"]["total_mb"] > 0


def test_pynvml_missing_degrades(monkeypatch):
    """pynvml 缺失（import 抛 ImportError）→ util/temp/power 为 None 而非报错。

    说明（对 brief 的最小修正）：本机无可用 CUDA（torch CPU 构建），
    brief 原测试假设“有 CUDA 但无 pynvml”，无 CUDA 时 gpu=[] 会 IndexError。
    故按控制者 Ruling 补 monkeypatch 模拟 1 张卡，保留断言语义。
    """
    import torch

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pynvml":
            raise ImportError("pynvml not installed (simulated)")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx: (1 << 30, 2 << 30))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda idx: 1 << 20)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx: 2 << 20)

    s = EnvSampler()
    try:
        data = s.sample()
    finally:
        s.close()
    assert data["gpu"][0]["util"] is None    # 无 pynvml → util 为 None 而非报错
    assert data["gpu"][0]["temp_c"] is None
    assert data["gpu"][0]["power_w"] is None
    assert data["gpu"][0]["mem_total_mb"] == 2048.0   # (2<<30) 字节 / MB
    assert data["gpu"][0]["mem_allocated_mb"] == 1.0
    assert data["gpu"][0]["mem_reserved_mb"] == 2.0
