"""``tools/infer/predict_yolo.py`` 的输入枚举逻辑测试（目录批量）。

背景
----
2026-10 AIStation 侧接 TorchKiln 时发现：``tkiln predict`` 只接受**单张图片**
（``cv2.imread(args.input)``），而平台的「预测」功能是把整个数据集目录挂进容器、
期望框架批量处理。那条链路自接入起就从未跑通过。

于是给它加了目录批量：递归遍历 + 按**相对路径**输出（数据集布局是
``images/train/a.jpg`` + ``images/val/a.jpg``，平铺会互相覆盖）。

这里只测纯函数 ``_list_images``，不碰 ``Runtime.load``（那需要真权重与 GPU）。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.infer.predict_yolo import IMAGE_EXTS, _list_images  # noqa: E402


def _write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


# ---------------------------------------------------------------------------
# 单文件
# ---------------------------------------------------------------------------


def test_single_file_yields_verbatim_output(tmp_path):
    """单文件 → 输出用 ``--output`` 原样路径（``rel`` 为 None）。"""
    p = _write(str(tmp_path / "a.jpg"))
    assert _list_images(p) == [(p, None)]


def test_single_non_image_file_still_yielded(tmp_path):
    """单文件时不做扩展名过滤——用户显式指定了就是他的事，报错交给 imread。"""
    p = _write(str(tmp_path / "a.bin"))
    assert _list_images(p) == [(p, None)]


def test_missing_path_treated_as_single_file(tmp_path):
    """路径不存在：仍然当作单文件返回，由 imread 报「读不出来」。

    刻意不在这里抛：这样调用方的错误信息里带着用户给的原始路径，
    比在这里报「没找到图片」更贴近真实原因。
    """
    p = str(tmp_path / "nope.jpg")
    assert _list_images(p) == [(p, None)]


# ---------------------------------------------------------------------------
# 目录：递归
# ---------------------------------------------------------------------------


def test_directory_flat(tmp_path):
    _write(str(tmp_path / "a.jpg"))
    _write(str(tmp_path / "b.png"))
    got = _list_images(str(tmp_path))
    assert [rel for _abs, rel in got] == ["a.jpg", "b.png"]


def test_directory_recurses_into_subdirs(tmp_path):
    """数据集布局是 images/train、images/val —— 只扫顶层会得到空列表。"""
    _write(str(tmp_path / "images" / "train" / "a.jpg"))
    _write(str(tmp_path / "images" / "val" / "b.jpg"))
    got = _list_images(str(tmp_path))
    assert [rel for _abs, rel in got] == ["images/train/a.jpg", "images/val/b.jpg"]


def test_directory_deeply_nested(tmp_path):
    _write(str(tmp_path / "a" / "b" / "c" / "d.jpg"))
    assert [rel for _a, rel in _list_images(str(tmp_path))] == ["a/b/c/d.jpg"]


def test_empty_directory(tmp_path):
    assert _list_images(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# 目录：按扩展名过滤
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", IMAGE_EXTS)
def test_all_supported_extensions(tmp_path, ext):
    _write(str(tmp_path / f"img{ext}"))
    assert len(_list_images(str(tmp_path))) == 1


def test_extension_match_is_case_insensitive(tmp_path):
    """``.JPG`` / ``.PNG`` 也要认——数据集导出方的命名大小写不统一。"""
    _write(str(tmp_path / "a.JPG"))
    _write(str(tmp_path / "b.PNG"))
    assert len(_list_images(str(tmp_path))) == 2


def test_non_images_excluded(tmp_path):
    _write(str(tmp_path / "a.jpg"))
    for junk in ("labels.txt", "data.yaml", "train.txt", "log.log", "w.bin"):
        _write(str(tmp_path / junk))
    assert [rel for _a, rel in _list_images(str(tmp_path))] == ["a.jpg"]


def test_extension_needs_dot_boundary(tmp_path):
    """``.jpg.bak`` 不是图片（以末尾扩展名判断，不做子串匹配）。"""
    _write(str(tmp_path / "a.jpg.bak"))
    assert _list_images(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# 相对路径：同名不覆盖
# ---------------------------------------------------------------------------


def test_same_basename_in_different_subdirs_kept_separate(tmp_path):
    """``images/train/a.jpg`` 与 ``images/val/a.jpg`` 必须都能出现。

    输出若平铺就会互相覆盖，表现为「跑完了但少了一半结果」。
    """
    _write(str(tmp_path / "images" / "train" / "a.jpg"))
    _write(str(tmp_path / "images" / "val" / "a.jpg"))
    got = _list_images(str(tmp_path))
    assert len(got) == 2
    assert {rel for _a, rel in got} == {"images/train/a.jpg", "images/val/a.jpg"}


def test_relative_path_uses_forward_slash(tmp_path):
    """相对路径用 ``/`` 分隔，便于直接当跨平台标识（如对象存储 key）用。"""
    _write(str(tmp_path / "x" / "y" / "a.jpg"))
    _abs, rel = _list_images(str(tmp_path))[0]
    assert rel == "x/y/a.jpg"


def test_absolute_paths_are_absolute(tmp_path):
    _write(str(tmp_path / "sub" / "a.jpg"))
    abs_path, _rel = _list_images(str(tmp_path))[0]
    assert os.path.isabs(abs_path)
    assert os.path.isfile(abs_path)


# ---------------------------------------------------------------------------
# 顺序稳定
# ---------------------------------------------------------------------------


def test_order_is_deterministic_across_subdirs(tmp_path):
    """顺序按相对路径排，不受文件系统枚举顺序影响。"""
    for name in ("m", "a", "z"):
        _write(str(tmp_path / name / f"{name}.jpg"))
        _write(str(tmp_path / f"top_{name}.jpg"))

    first = [rel for _a, rel in _list_images(str(tmp_path))]
    second = [rel for _a, rel in _list_images(str(tmp_path))]
    assert first == second == sorted(first)


def test_same_dir_sorted_by_name(tmp_path):
    for n in ("c.jpg", "a.jpg", "b.jpg"):
        _write(str(tmp_path / n))
    assert [rel for _a, rel in _list_images(str(tmp_path))] == ["a.jpg", "b.jpg", "c.jpg"]
