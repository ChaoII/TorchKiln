"""Windows 平台无关化审计测试（防止回退）。

目标：确保 ``torchkiln/`` 与 ``ptcore/`` 的源码里没有 Linux-only 的**真实代码**，
让 TorchKiln 在 Windows 原生可跑（简单模式 ``python -m torchkiln.simple_run``）。

扫描规则（全部基于真实代码 token，忽略注释 / docstring，控制误报）：

1. 调用 ``os.fork(`` —— Windows 无 ``fork``。
2. ``import resource`` / ``import fcntl`` / ``import posixpath`` /
   ``import pwd`` / ``import grp``（含 ``from X import ...``）—— 这些是
   Linux-only 标准库，在 Windows 上 ``ImportError``。
3. 模块属性访问 ``fcntl.`` / ``pwd.`` / ``grp.`` / ``posixpath`` —— 兜底
   ``from X import *`` 之类绕过 import 检测的写法。
4. 非 docstring 字符串字面量里的 POSIX 固定路径 ``/dev/shm`` / ``/dev/null``。
5. **裸 ``import signal``（无平台保护）**：``signal`` 在 Windows 上无害，但
   没有 ``os.name`` / ``sys.platform`` / ``platform.system`` 保护的裸
   ``import signal`` 属于 Linux 习惯（常见死代码）。本条让"清理前 RED /
   清理后 GREEN"成立：把无保护的 ``import signal`` 判为失败项。

实现说明（防误报的关键）：

* 用 ``tokenize`` 逐 token 匹配 NAME / OP / STRING / NUMBER，**丢弃 COMMENT、
  缩进与换行**——注释与 docstring 不参与 NAME/OP 模式匹配。
* ``/dev/shm`` / ``/dev/null`` 用 ``ast`` 只检查**非 docstring** 的
  ``ast.Constant`` 字符串常量——docstring 里出现不算命中。
* import 检测走 ``ast``（``Import`` / ``ImportFrom``），比子串匹配可靠。

配套文档：``docs/windows_compat.md``。
"""
import ast
import io
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("torchkiln", "ptcore")

# Linux-only 标准库：import 即失败
_LINUX_ONLY_MODULES = ("resource", "fcntl", "posixpath", "pwd", "grp")
# 非 docstring 字符串里禁止出现的 POSIX 固定路径
_POSIX_PATHS = ("/dev/shm", "/dev/null")
# 视为"平台保护"存在的真实代码引用
_GUARDS = (("os", "name"), ("sys", "platform"), ("platform", "system"))


def _real_tokens(src):
    """返回 (过滤后的 (type, string) 列表, 是否成功)。

    只保留 NAME / OP / STRING / NUMBER；COMMENT、行内空白、缩进与换行全部丢弃，
    保证注释 / docstring 不会参与 NAME/OP 模式匹配。
    """
    toks = []
    try:
        gen = tokenize.generate_tokens(io.StringIO(src).readline)
        for t in gen:
            if t.type in (tokenize.NAME, tokenize.OP,
                          tokenize.STRING, tokenize.NUMBER):
                toks.append((t.type, t.string))
        return toks, True
    except (tokenize.TokenError, IndentationError, SyntaxError, UnicodeDecodeError):
        return toks, False


def _flat(toks):
    return [s for _ty, s in toks]


def _contains_seq(flat, seq):
    n = len(seq)
    return any(flat[i:i + n] == seq for i in range(len(flat) - n + 1))


def _has_platform_guard(toks):
    """文件里是否存在 os.name / sys.platform / platform.system 的真实代码引用。"""
    flat = _flat(toks)
    return any(_contains_seq(flat, [a, ".", b]) for a, b in _GUARDS)


def _imported_modules(tree):
    """返回被 import 的顶层模块名集合（含 ``from X import ...``）。"""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
    return found


def _docstring_ids(tree):
    """module/class/function 首个语句里的字符串常量（docstring）节点 id 集合。"""
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docs.add(id(body[0].value))
    return docs


def _posix_path_strings(tree):
    """返回非 docstring 字符串字面量里命中的 (路径, 原文) 列表。"""
    docs = _docstring_ids(tree)
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docs):
            for p in _POSIX_PATHS:
                if p in node.value:
                    hits.append((p, node.value))
    return hits


def _file_offenses(path):
    """返回该文件的命中列表（每项为一句中文描述）；空列表 = 通过。"""
    src = path.read_text(encoding="utf-8", errors="replace")
    toks, ok = _real_tokens(src)
    if not ok:
        return []  # 无法 tokenize（语法错误/编码异常）时跳过，避免引入噪音
    flat = _flat(toks)
    names = {s for ty, s in toks if ty == tokenize.NAME}

    offenses = []
    # 1) os.fork( 调用
    if _contains_seq(flat, ["os", ".", "fork", "("]):
        offenses.append("调用 os.fork(")
    # 3) 模块属性访问（兜底）
    for mod in ("fcntl", "pwd", "grp"):
        if _contains_seq(flat, [mod, "."]):
            offenses.append(f"模块属性访问 {mod}.")
    if "posixpath" in names:
        offenses.append("使用 posixpath")

    # 2) / 4) / 5) 需要 AST
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return offenses

    # 2) Linux-only 模块 import
    imported = _imported_modules(tree)
    for mod in _LINUX_ONLY_MODULES:
        if mod in imported:
            offenses.append(f"import {mod}（Linux-only 模块）")
    # 4) 非 docstring 字符串里的 POSIX 固定路径
    for p, _val in _posix_path_strings(tree):
        offenses.append(f"字符串含 {p}")
    # 5) 裸 import signal（无平台保护）
    if "signal" in imported and not _has_platform_guard(toks):
        offenses.append("裸 import signal（无 os.name/sys.platform/platform.system 保护）")
    return offenses


def test_no_linux_only_apis():
    problems = []
    for root in SCAN_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            for off in _file_offenses(path):
                problems.append(f"{path.relative_to(REPO)}: {off}")
    assert not problems, (
        "发现 Linux-only 用法（Windows 平台无关化被破坏）：\n"
        + "\n".join(problems)
    )
