"""``torchkiln.infer_api`` 的纯逻辑测试（不加载任何真实权重）。

为什么用「假 model + 假 post」：真权重要训练 1~40 秒且输出不确定，
掩膜/多边形这类分支根本走不到；而这里要验的恰恰是
**letterbox 坐标回映射、掩膜 dtype、polygon 提取**这些纯几何逻辑——
用可精确构造的输入才能断言到具体数值。

跑法::

    cd D:\\TorchKiln
    python -m pytest tests/test_infer_api.py -q
"""
from __future__ import absolute_import

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchkiln.infer_api import (  # noqa: E402
    Runtime,
    _mask_to_polygon,
    _r,
    _unletterbox_mask,
    _unletterbox_xyxy,
)


# --------------------------------------------------------------------- 工具


def _rt(task_name, post, size=64, names=None, model=None):
    """构造一个不碰 torch.load 的 Runtime（model/post 由测试自己给）。"""
    return Runtime(
        config={"Architecture": {"task": task_name},
                "Global": {"model_name": "stub-" + task_name},
                "Train": {"dataset": {"transform": {"image_size": size},
                                       "names": names}}},
        task_name=task_name, names=names, size=size,
        task=object(), post=post, model=model or (lambda x: x),
        device=torch.device("cpu"), config_path="stub.yml", weights="stub.pth")


class _Post(object):
    """把预置结果原样返回的假后处理（顺便断言 infer 传进来的 raw）。"""

    def __init__(self, result, saw_size=None):
        self._result = result
        self._saw_size = saw_size if saw_size is not None else {}

    def __call__(self, raw, size=None):
        self._saw_size["size"] = size
        return self._result


# --------------------------------------------------------- 坐标回映射（数学）


def test_unletterbox_xyxy_identity_letterbox_is_noop():
    """图片尺寸正好等于训练尺寸时 letterbox 不缩放也不 padding，坐标必须原样。"""
    p1, p2 = _unletterbox_xyxy([10.0, 20.0, 30.0, 40.0], ratio=1.0, pad=(0.0, 0.0))
    assert p1 == pytest.approx((10.0, 20.0))
    assert p2 == pytest.approx((30.0, 40.0))


def test_unletterbox_xyxy_undoes_scale_and_pad():
    """缩放 + 上下 padding 后，框要能精确还原到原图坐标。

    取 ratio=0.5、pad=(7, 11)：原图 (10, 20) ~ (40, 40) 经 letterbox 后
    x*0.5+7、y*0.5+11 -> [12, 21, 27, 31]。
    """
    p1, p2 = _unletterbox_xyxy([12.0, 21.0, 27.0, 31.0], ratio=0.5, pad=(7.0, 11.0))
    assert p1 == pytest.approx((10.0, 20.0))
    assert p2 == pytest.approx((40.0, 40.0))


def test_unletterbox_mask_accepts_bool():
    """回归：分割掩膜是 ``torch.bool``，OpenCV 的 resize 不支持 bool。

    少了这层 dtype 转换，segment 推理会直接抛
    ``src data type = bool is not supported``（真机上就是这样炸出来的）。
    """
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 4:16] = True          # 10 行 x 12 列 = 120
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    out = _unletterbox_mask(mask, img, ratio=1.0, pad=(0.0, 0.0), w0=20, h0=20)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 1}
    assert out[10, 10] == 1
    assert out.sum() == 120


def test_unletterbox_mask_crops_padding_regions():
    """pad 区域必须被裁掉，否则原图边缘会多出一圈不属于目标的像素。"""
    # letterbox 后 24x24，其中 pad=(4,4)、有效区 16x16、ratio=1 -> 原图 16x16
    m = np.zeros((24, 24), dtype=np.uint8)
    m[4:20, 4:20] = 7
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    out = _unletterbox_mask(m, img, ratio=1.0, pad=(4.0, 4.0), w0=16, h0=16)
    assert out.shape == (16, 16)
    # 全是有效值，不含 padding 的 0
    assert out.min() == 7


def test_unletterbox_mask_uses_float_for_linear():
    """深度图走线性插值，保留浮点（转 uint8 会把 0.1m 与 0.9m 都压成 0/1）。"""
    m = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    out = _unletterbox_mask(m, img, ratio=1.0, pad=(0.0, 0.0), w0=16, h0=16, nearest=False)
    assert out.dtype == np.float32
    assert out.max() == pytest.approx(1.0, abs=1e-5)
    assert out.min() == pytest.approx(0.0, abs=1e-5)


# ------------------------------------------------------------- polygon 提取


def test_mask_to_polygon_rectangle():
    """一个 10x10 实心方块的外轮廓应是矩形多边形，且顶点落在方块边界上。"""
    m = np.zeros((40, 40), dtype=np.uint8)
    m[10:20, 5:15] = 1
    poly = _mask_to_polygon(m)
    assert len(poly) >= 4
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) == 5 and max(xs) == 14
    assert min(ys) == 10 and max(ys) == 19


def test_mask_to_polygon_empty_returns_empty_list():
    assert _mask_to_polygon(np.zeros((10, 10), dtype=np.uint8)) == []


def test_mask_to_polygon_tiny_blob_below_min_area():
    """比 min_area 还小的碎片不该产生多边形（否则客户端会画出一堆噪点框）。"""
    m = np.zeros((40, 40), dtype=np.uint8)
    m[0:1, 0:1] = 1
    assert _mask_to_polygon(m) == []


def test_r_is_json_safe_float():
    """numpy 标量不能直接进 json.dumps，_r 负责转成原生 float。"""
    v = _r(np.float32(0.123456))
    assert isinstance(v, float)
    assert v == pytest.approx(0.1235, abs=1e-4)
    json.dumps({"v": v})  # 不抛异常即通过


# ------------------------------------------------- infer() 的结构化输出映射


def test_infer_detect_maps_bbox_and_names():
    post = _Post([{"bboxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                   "scores": torch.tensor([0.87]),
                   "labels": torch.tensor([1])}])
    rt = _rt("detect", post, size=64, names=["cat", "dog"])
    res = rt.infer(np.zeros((64, 64, 3), dtype=np.uint8))

    assert res.data["task"] == "detect"
    assert res.data["image_width"] == 64 and res.data["image_height"] == 64
    assert len(res.data["detections"]) == 1
    d = res.data["detections"][0]
    assert d["class"] == "dog" and d["label_id"] == 1
    assert d["confidence"] == pytest.approx(0.87, abs=1e-3)
    assert d["bbox"] == [10.0, 20.0, 30.0, 40.0]
    json.dumps(res.data, ensure_ascii=False)  # 必须是 JSON 安全的


def test_infer_detect_without_names_falls_back_to_id():
    """配置没写 names 时退回类别 id（与旧 predict_yolo 行为一致），不能崩。"""
    post = _Post([{"bboxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                   "scores": torch.tensor([0.5]),
                   "labels": torch.tensor([3])}])
    res = _rt("detect", post).infer(np.zeros((64, 64, 3), dtype=np.uint8))
    assert res.data["detections"][0]["class"] == "3"


def test_infer_classify_top1_lands_in_detections():
    """分类没有框，但仍给 detections[0]=top1，让只认 detections 的客户端有值。"""
    # 分类的 model 输出是 (B, num_classes) 的 logits，不是 (B,3,H,W) 的特征图，
    # 所以这里用固定 logits 的假 model（恒等会把图片当 logits softmax，错）。
    logits = torch.tensor([[0.0, 1.0, 0.0]])
    rt = _rt("classify", _Post([logits]), names=["a", "b", "c"],
             model=lambda x: logits)
    res = rt.infer(np.zeros((64, 64, 3), dtype=np.uint8))
    assert res.data["task"] == "classify"
    assert len(res.data["labels"]) == 3
    assert res.data["labels"][0]["class"] == "b", "top-1 应是 logits 最大的类"
    assert res.data["detections"][0]["class"] == "b"
    assert res.data["detections"][0]["confidence"] == pytest.approx(
        float(torch.softmax(logits, dim=1)[0, 1]), abs=1e-3)
    assert res.data["top1"] == "b"


def test_infer_segment_emits_polygon_from_mask():
    """分割：掩膜（bool！）要被解成原图坐标的多边形。"""
    mask = torch.zeros((1, 64, 64), dtype=torch.bool)
    mask[0, 10:30, 20:50] = True
    post = _Post([{"bboxes": torch.tensor([[20.0, 10.0, 49.0, 29.0]]),
                   "scores": torch.tensor([0.9]),
                   "labels": torch.tensor([0]),
                   "masks": mask}])
    res = _rt("segment", post, names=["road"]).infer(np.zeros((64, 64, 3), dtype=np.uint8))
    d = res.data["detections"][0]
    assert d["class"] == "road"
    assert "polygon" in d, "掩膜非空却没出多边形，说明取掩膜/回映射那一段错了"
    xs = [p[0] for p in d["polygon"]]
    ys = [p[1] for p in d["polygon"]]
    # 掩膜原点在 letterbox 尺寸(=训练尺寸=原图尺寸)上，ratio=1、pad=0，
    # 所以多边形应与掩膜位置一致（20..49 / 10..29）
    assert min(xs) == 20 and max(xs) == 49
    assert min(ys) == 10 and max(ys) == 29


def test_infer_obb_points_and_envelope_bbox():
    """旋转框的后处理输出是 ``(N, 5)`` 的 xywhr（cx, cy, w, h, angle）。"""
    post = _Post([{"bboxes": torch.tensor([[32.0, 32.0, 20.0, 10.0, 0.0]]),
                   "scores": torch.tensor([0.77]),
                   "labels": torch.tensor([0])}])
    res = _rt("obb", post, names=["car"]).infer(np.zeros((64, 64, 3), dtype=np.uint8))
    d = res.data["detections"][0]
    assert len(d["points"]) == 4, "旋转框应是 4 个角点"
    xs = [p[0] for p in d["points"]]
    ys = [p[1] for p in d["points"]]
    # angle=0 -> 中心 (32,32)、半宽 10、半高 5 -> x 22..42、y 27..37
    assert min(xs) == pytest.approx(22.0, abs=0.5)
    assert max(xs) == pytest.approx(42.0, abs=0.5)
    assert min(ys) == pytest.approx(27.0, abs=0.5)
    assert max(ys) == pytest.approx(37.0, abs=0.5)
    # bbox 是外接矩形，方便只认 bbox 的通用客户端也能画对位置
    assert d["bbox"] == [_r(min(xs), 2), _r(min(ys), 2),
                        _r(max(xs), 2), _r(max(ys), 2)]


def test_infer_pose_keypoints_are_unletterboxed():
    post = _Post([{"bboxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                   "scores": torch.tensor([0.6]),
                   "labels": torch.tensor([0]),
                   "kpts": torch.tensor([[[12.0, 14.0, 0.8], [40.0, 44.0, 0.5]]])}])
    res = _rt("pose", post).infer(np.zeros((64, 64, 3), dtype=np.uint8))
    kps = res.data["detections"][0]["keypoints"]
    assert len(kps) == 2
    assert (kps[0]["x"], kps[0]["y"], kps[0]["score"]) == (12.0, 14.0, pytest.approx(0.8, abs=1e-3))
    assert (kps[1]["x"], kps[1]["y"]) == (40.0, 44.0)


def test_infer_semantic_returns_mask_png_and_no_detections():
    m = torch.zeros((1, 64, 64), dtype=torch.uint8)
    m[0, 0:32, 0:32] = 1
    res = _rt("semantic", _Post([m])).infer(np.zeros((64, 64, 3), dtype=np.uint8))
    assert res.data["detections"] == []
    assert res.data["mask_png"], "整幅分割必须给出可渲染的掩膜"
    assert {c["id"] for c in res.data["classes"]} == {0, 1}


def test_infer_depth_reports_range_and_png():
    d = torch.linspace(0.5, 3.5, 64 * 64).reshape(1, 64, 64)
    res = _rt("depth", _Post([d])).infer(np.zeros((64, 64, 3), dtype=np.uint8))
    assert res.data["detections"] == []
    assert res.data["depth"]["min"] == pytest.approx(0.5, abs=0.05)
    assert res.data["depth"]["max"] == pytest.approx(3.5, abs=0.05)
    assert res.data["depth_png"]


def test_infer_lane_row_converts_row_x_to_pixel_points():
    """lane_row 每行只存归一化 x，y 由行号隐含 -> 输出要补成像素坐标点。"""
    xs = torch.tensor([[0.1, 0.5, 0.9]])
    res = _rt("lane_row", _Post([xs])).infer(np.zeros((100, 200, 3), dtype=np.uint8))
    lane = res.data["lanes"][0]
    assert len(lane["points"]) == 3
    assert lane["points"][0][0] == pytest.approx(0.1 * 199, abs=0.5)
    assert lane["points"][0][1] == pytest.approx(0.0, abs=0.5)
    assert lane["points"][2][1] == pytest.approx(99.0, abs=0.5)
    assert lane["row_x"] == [pytest.approx(0.1, abs=1e-3), 0.5, 0.9]


def test_infer_passes_size_only_to_fullmap_tasks():
    """整幅输出的 task 才需要原图尺寸；框类 task 不该收到该参数。"""
    saw = {}
    rt = _rt("semantic", _Post([torch.zeros((1, 64, 64))], saw_size=saw), size=64)
    rt.infer(np.zeros((64, 64, 3), dtype=np.uint8))
    assert saw["size"] is not None, "semantic 需要 size 才能把掩膜还原到原图尺寸"


def test_infer_rejects_empty_image():
    with pytest.raises(ValueError):
        _rt("detect", _Post([])).infer(np.zeros((0, 0, 3), dtype=np.uint8))


def test_infer_converts_non_uint8_and_grayscale_input():
    """平台侧可能喂 RGB、float 或灰度图，不该在这里崩。"""
    post = _Post([{"bboxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
                   "scores": torch.tensor([0.1]), "labels": torch.tensor([0])}])
    rt = _rt("detect", post)
    for img in (np.zeros((64, 64), dtype=np.uint8),                       # 灰度
                np.zeros((64, 64, 3), dtype=np.float32),                  # float
                np.zeros((64, 64, 3), dtype=np.uint8)[:, :, ::-1]):      # RGB
        out = rt.infer(img)
        assert out.data["image_width"] == 64 and out.data["image_height"] == 64


def test_describe_reports_model_and_task():
    info = _rt("detect", _Post([]), names=["a", "b"]).describe()
    assert info["task"] == "detect"
    assert info["model_name"] == "stub-detect"
    assert info["num_classes"] == 2
    assert info["image_size"] == 64
