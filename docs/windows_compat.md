# Windows 平台无关化清单

本清单记录 TorchKiln 在 **Windows 原生可跑** 这件事上的**已确认兼容项**、**已知限制**与**修复记录**，
并配套一个**防回退审计测试** `tests/test_windows_compat.py`（`test_no_linux_only_apis`），
持续扫描 `torchkiln/` 与 `ptcore/` 源码，禁止重新引入 Linux-only 的真实代码。

> 服务侧（`service/`）仍运行在 Linux 容器里，本清单聚焦的是**训练/推理主链路**
> （`torchkiln/`、`ptcore/`、以及 Windows 宿主上要能起得来/杀得掉 `service/` 的进程逻辑）
> 在 Windows 原生环境下的可用性。

## 已确认兼容项

| 兼容点 | 位置 | 说明 |
|--------|------|------|
| Windows 每 epoch batch 数 −1 语义 | `ptcore/trainers/base.py:303-304`、`784-785` | 对齐 PaddleOCR：Windows 下训练循环每 epoch 只取 `len(loader)-1` 个 batch；LR 调度折算 `opt_steps_per_epoch` 与 `max_iter` 均据此修正，避免 Windows 上学习率相位错乱。 |
| `persistent_workers` 保活 | `ptcore/trainers/base.py:252-259`（train）、`272-275`（eval） | Windows 上 spawn worker 代价高，`num_workers>0` 时开启 `persistent_workers` + `prefetch_factor`，跨 epoch 保活 worker，避免每 epoch 重复 spawn 拖慢训练。 |
| ctypes 隐藏缓存目录 | `ptcore/pretrained.py:85-95` | `_hide_dir` 仅在 `sys.platform == "win32"` 时通过 `ctypes.windll.kernel32.SetFileAttributesW` 把 `~/.torchkiln` 预训练缓存目录标记为隐藏，best-effort、失败不抛。 |
| `newline="\n"` 固定换行 | `ptcore/metrics_sink.py:208-209`（另 `223`） | metrics JSONL 写盘固定 `newline="\n"`，Windows 宿主 `tail`/消费方读日志不会被 `\r\n` 干扰。 |
| 进程判活 / 终止信号的 nt 分支 | `service/runner.py:320-333`（`pid_alive`）、`401-411`（`send_signal`） | `os.name == "nt"` 时用 `ctypes` `OpenProcess`/`QueryFullProcessImageName` 判活、用 `proc.kill()/terminate()` 终止；非 nt 才走 `os.killpg` 按进程组杀（Windows 无进程组）。 |
| `_posix_join` 规避反斜杠 | `service/runner.py:161-174` | 用 `/` 手工拼路径而**不用** `os.path.join`——spec 路径可能由 Windows 侧构造，`os.path.join` 在 Windows 上产生 `\` 反斜杠，传进 Linux 容器会找不到文件且本地无任何信号。 |

> 附：`ptcore/env_sampler.py:33` 的 `if os.name == "nt":` 分支同样是已确认的 Windows 适配点（采样器在 Windows 上的降级路径）。

## 已知限制

1. **PSE C 扩展需要 VS C++ 工具链**
   `torchkiln/ocr/postprocess/pse_postprocess/pse/__init__.py` 在模块导入时就地编译 Cython 扩展 `pse`
   （`python setup.py build_ext --inplace`，`language='c++'`）。在 Windows 上编译失败会抛出 `RuntimeError`
   并提示：**需安装 Visual Studio 2019+ 的 `desktop development with C++` 全部默认组件**（含 C++ 编译器与
   v143 工具集）。纯 OCR 检测/识别主链路不依赖该扩展；只有用到 PSE 后处理时才需要。
2. **Windows `spawn` 需要 `__main__` 保护**
   Windows 下 `multiprocessing` 默认启动方式是 `spawn`（不是 fork）：子进程会**重新 import 主模块**，
   因此所有 `multiprocessing` 相关入口（DataLoader `num_workers>0`、`multiprocessing.Value/Queue` 等）
   必须放在 `if __name__ == "__main__":` 保护之下，否则子进程会无限递归 spawn。简单模式
   `python -m torchkiln.simple_run` 由 `service/runner.py` 以子进程方式拉起，其内部训练循环的
   多进程 worker 需遵守此约定。
3. **`multiprocessing.Value` 需入口保护**
   同理，任何在模块顶层创建 `multiprocessing.Value/Array/Queue/Manager` 的写法，在 Windows
   `spawn` 下都会触发子进程重新执行模块顶层代码，必须包裹在 `if __name__ == "__main__":` 中或
   在 worker 函数内部惰性创建。

## 简单模式跑法

简单模式下，AIStation 侧用一行命令拉起一个单作业进程（train / eval / predict / export 四选一，
由 `spec.kind` 分派），事件通过 HTTP loopback 回调上报：

```bash
python -m torchkiln.simple_run <spec.json> --callback-url <url> --token <t> [--task-id <int>] [--repo-root <root>]
```

入口实现见 `torchkiln/simple_run.py`（模块 docstring 有完整说明）；更完整的平台对接、
JobSpec 契约与事件协议见后续文档（`docs/USER_GUIDE.md`、`docs/TASK_ARCHITECTURE.md`）。

## 修复记录

本次（Windows 平台无关化审计）清理的两处：

1. **删除死代码 `import signal`** — `torchkiln/ocr/data/__init__.py:10`
   该 `import signal` 为历史残留（全文件无任何 `signal.` 调用），属 Linux 习惯的裸导入，
   在 Windows 下无害但属于无效依赖，删除以消除审计命中与潜在困惑。
2. **PSE C 扩展编译改为 `shell=False` + 列表参数 + `cwd=`** —
   `torchkiln/ocr/postprocess/pse_postprocess/pse/__init__.py`
   原实现为 `os.chdir('torchkiln/...')` + `subprocess.call('... setup.py ...', shell=True)`：
   `shell=True` 存在命令注入面、POSIX 相对路径在 Windows cmd 下脆弱、且 `os.chdir` 会改动
   进程 cwd（全局副作用）。改为 `subprocess.call([sys.executable, "setup.py", "build_ext",
   "--inplace"], cwd=<包目录>)`（`shell=False`、列表参数、用 `cwd=` 替代 `os.chdir`），
   编译失败时的报错与降级路径保持等价，仅增强为中英双语提示（Windows 需 VS C++ 工具链）。

## 防回退审计测试

`tests/test_windows_compat.py::test_no_linux_only_apis` 基于 `tokenize` + `ast` 只检查**真实代码
token**（忽略注释与 docstring，控制误报），扫描 `torchkiln/` 与 `ptcore/` 全部 `.py`：

- 调用 `os.fork(`；
- `import resource` / `fcntl` / `posixpath` / `pwd` / `grp`（Linux-only 标准库）及其模块属性访问；
- 非 docstring 字符串字面量里的 `/dev/shm`、`/dev/null`（POSIX 固定路径）；
- **裸 `import signal`（无 `os.name` / `sys.platform` / `platform.system` 平台保护）**。

命中即失败，防止后续改动重新引入 Windows 不可用的平台专属代码。
