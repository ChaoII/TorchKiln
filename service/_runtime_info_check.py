"""``GET /api/v1/info`` 的 ``runtime`` 字段回归守卫。

为什么需要这个文件：平台侧（AIStation）后端**不装 torch**，torch / CUDA / cuDNN
版本只有装了 torch 的训练进程报得出来——所以由本服务上报，平台只做汇总展示。
这个字段因此成了跨仓契约的一部分，格式或可用性悄悄变掉，平台上就会显示天书。

守四件事：

1. **cuDNN 版本整数的格式化**（90107 -> 9.1.7）：格式错了前端会显示天书。注意
   编码规则在 cuDNN 8 处换过一次（>=8 是 5 位、<=7 是 4 位），只按 5 位算会把
   老版本读成 0.76.5——这个坑已经用 7605 钉住了；
2. **``framework_info`` 必须带 ``runtime`` 键**，且探测失败时**降级成 None 而不是
   抛异常**——无 torch 的单测环境、CPU-only 构建、驱动缺失都不该把 ``GET /info``
   打挂（平台侧会因一个版本字段拿不到整份 info）；
3. **缓存生效**：``import torch`` 冷启动要 20 秒以上，只能发生一次；
4. **主进程绝不 import torch**（本轮最关键的设计约束）：本机实测 ``import torch``
   要 23 秒且**长期持有 GIL**，放本进程或线程池都会把 uvicorn 事件循环饿死——
   训练进行中那 23 秒 SSE 心跳与日志转发全部停摆，平台会误判服务已死。
   因此探测必须走**独立子进程**，这条断言就是它的守卫。
"""
import os
import sys

REPO_ROOT = os.environ.get("TKILN_REPO_ROOT", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from service import registry  # noqa: E402

FAIL = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    if not cond:
        FAIL.append(name)


print("=== cuDNN 版本整数 -> 人类可读 ===")
# 规则在 cuDNN 8 处换过：>=8 是 5 位 major*10000+minor*100+patch，
# <=7 是 4 位 major*1000+minor*100+patch。只按 5 位算会把老版本读成 0.76.5。
for raw, want in ((90107, "9.1.7"), (80902, "8.9.2"), (120109, "12.1.9"),
                  (7605, "7.6.5"), (8801, "8.8.1")):
    got = registry._fmt_cudnn(raw)
    check(f"{raw} -> {want}", got == want, f"实际 {got}")

print("\n=== 主进程不许 import torch（探测必须走子进程）===")
check("测试前 sys.modules 无 torch", "torch" not in sys.modules)
probe = registry.runtime_info()
check("探测后 sys.modules 仍无 torch", "torch" not in sys.modules,
      "有则说明有人在本进程 import torch，会饿死事件循环")

print("\n=== 真实探测：键齐全、类型正确 ===")
for key in ("python", "torch", "cuda", "cudnn", "device", "device_count"):
    check(f"runtime.{key} 键存在", key in probe, str(sorted(probe.keys())))
check("device_count 是 int", isinstance(probe.get("device_count"), int),
      repr(probe.get("device_count")))
check("cudnn 已格式化（含点号）",
      probe.get("cudnn") is None or "." in str(probe["cudnn"]),
      repr(probe.get("cudnn")))
check("python 版本可读", bool(probe.get("python")), repr(probe.get("python")))

print("\n=== framework_info 必须带 runtime，且旧字段未被顶掉 ===")
info = registry.framework_info(REPO_ROOT)
check("runtime 键存在", "runtime" in info, str(sorted(info.keys())))
check("runtime 内容与 runtime_info() 一致", info.get("runtime") == probe)
check("旧字段未被顶掉", info.get("framework") == "torchkiln"
      and info.get("spec_version") == "1.0", str(info.get("framework")))

print("\n=== 缓存：探测只发生一次 ===")
a = registry.runtime_info()
b = registry.runtime_info()
check("两次调用同一对象", a is b, f"{id(a)} vs {id(b)}")
check("缓存命中后不重跑子进程", registry._RUNTIME_CACHE is a)

print("\n=== 降级：子进程探测失败时返回含 None 的字典，绝不抛异常 ===")
saved_cache = registry._RUNTIME_CACHE
registry._RUNTIME_CACHE = None
import subprocess as _sp  # noqa: E402

_orig_run = _sp.run


def _boom(*args, **kwargs):
    raise RuntimeError("模拟子进程起不来")


_sp.run = _boom
try:
    degraded = registry.runtime_info()
    check("不抛异常", True)
    check("全字段降级为 None", all(degraded.get(k) is None for k in
                                   ("python", "torch", "cuda", "cudnn", "device")),
          str(degraded))
    check("device_count 降级为 0", degraded.get("device_count") == 0)
    check("缓存已写入（下次不再重试）", registry._RUNTIME_CACHE is degraded)
finally:
    _sp.run = _orig_run
    registry._RUNTIME_CACHE = saved_cache

print("\n=== warm_runtime_info 吞掉所有异常 ===")
registry._RUNTIME_CACHE = None
try:
    registry.warm_runtime_info()
    check("不抛异常", True)
except Exception as exc:  # noqa: BLE001
    check("不抛异常", False, repr(exc))
finally:
    registry._RUNTIME_CACHE = saved_cache

print("\n" + "=" * 60)
if FAIL:
    print(f"失败 {len(FAIL)} 项: " + "; ".join(FAIL))
    sys.exit(1)
print("全部通过：/info 的 runtime 格式稳定、探测失败可降级、走子进程不碰事件循环")
