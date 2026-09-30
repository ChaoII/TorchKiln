"""``tkiln serve`` 的 HTTP 契约测试（不打真权重，用假 Runtime）。

这份契约是**平台侧依赖的**：AIStation 生成的前端「在线推理」调用、对外文档、
以及既有 YOLO/PaddleX 部署服务全都按这个格式写死了。所以它属于"改了会静默
破坏外部调用方"的地方，必须用测试钉住：

* ``GET /health`` -> ``{"status": "ok", "model_name": ...}``
* ``POST /predict`` -> ``success`` / ``detections`` / ``image_width`` /
  ``image_height`` / ``inference_time_ms``，且 ``detections`` 每项恒含
  ``class`` 与 ``confidence``
* ``X-API-Key`` 配了就必须带，带错返回 401

跑法::

    cd D:\\TorchKiln
    python -m pytest tests/test_serve.py -q
"""
from __future__ import absolute_import

import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")
_testclient = pytest.importorskip("starlette.testclient")
torch = pytest.importorskip("torch")

from starlette.testclient import TestClient  # noqa: E402

from torchkiln.infer_api import InferResult  # noqa: E402
from torchkiln.serve import build_app  # noqa: E402


class _StubRuntime(object):
    """只实现 serve 用到的两个方法：``describe()`` 与 ``infer()``。"""

    def __init__(self, task="detect", detections=None, raise_exc=None):
        self._task = task
        self._detections = detections if detections is not None else [
            {"class": "dog", "confidence": 0.91, "bbox": [1.0, 2.0, 3.0, 4.0],
             "label_id": 1},
        ]
        self._raise = raise_exc

    def describe(self):
        return {"model_name": "stub-det", "task": self._task, "image_size": 640,
                "num_classes": 2, "device": "cpu", "config_path": "stub.yml"}

    def infer(self, img):
        if self._raise is not None:
            raise self._raise
        h, w = img.shape[:2]
        return InferResult({
            "image_width": int(w), "image_height": int(h), "task": self._task,
            "model_name": "stub-det", "detections": self._detections,
        })


def _png(w=40, h=30):
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    return {"file": ("x.png", io.BytesIO(buf.tobytes()), "image/png")}


def _client(api_key=None, runtime=None):
    return TestClient(build_app(runtime or _StubRuntime(), api_key=api_key))


# ------------------------------------------------------------------ /health


def test_health_reports_ok_and_model_name():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_name": "stub-det"}


def test_info_exposes_task_and_size():
    r = _client().get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "detect" and body["image_size"] == 640


# ----------------------------------------------------------------- /predict


def test_predict_returns_platform_contract_fields():
    r = _client().post("/predict", files=_png())
    assert r.status_code == 200
    body = r.json()
    # 这四个字段是平台既有调用方依赖的，缺一个就是破坏兼容
    for k in ("success", "detections", "image_width", "image_height",
              "inference_time_ms"):
        assert k in body, "缺少平台依赖的字段: " + k
    assert body["success"] is True
    assert body["image_width"] == 40 and body["image_height"] == 30
    assert isinstance(body["inference_time_ms"], (int, float))
    d = body["detections"][0]
    assert "class" in d and "confidence" in d


def test_predict_result_is_json_serializable():
    """/predict 的返回要能直接进 json（平台前端按 JSON 解析）。"""
    r = _client().post("/predict", files=_png())
    r.json()          # 解析失败会直接抛异常
    assert r.headers["content-type"].startswith("application/json")


def test_predict_accepts_rgb_and_gray_uploads():
    for w, h, chans in ((40, 30, 3), (20, 20, 1)):
        ok, buf = cv2.imencode(".png", np.zeros((h, w, chans), dtype=np.uint8))
        assert ok
        r = _client().post("/predict",
                           files={"file": ("x.png", io.BytesIO(buf.tobytes()), "image/png")})
        assert r.status_code == 200
        assert r.json()["image_width"] == w


# --------------------------------------------------------------------- 鉴权


def test_api_key_required_when_configured():
    r = _client(api_key="k").post("/predict", files=_png())
    assert r.status_code == 401, "配了 api_key 就不能放过无头的请求"


def test_wrong_api_key_rejected():
    r = _client(api_key="k").post("/predict", files=_png(),
                                  headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_correct_api_key_accepted():
    r = _client(api_key="k").post("/predict", files=_png(),
                                  headers={"X-API-Key": "k"})
    assert r.status_code == 200


def test_no_api_key_configured_allows_unauthenticated():
    """本地自测不该被鉴权挡住（api_key=None 时不校验）。"""
    r = _client(api_key=None).post("/predict", files=_png())
    assert r.status_code == 200


# -------------------------------------------------------------- 错误处理


def test_invalid_image_returns_400_not_500():
    r = _client().post("/predict",
                       files={"file": ("x.png", io.BytesIO(b"not-an-image"), "image/png")})
    assert r.status_code == 400


def test_empty_file_returns_400():
    r = _client().post("/predict",
                       files={"file": ("x.png", io.BytesIO(b""), "image/png")})
    assert r.status_code == 400


def test_inference_exception_returns_500_with_reason():
    """推理内部报错要回 500 + 人话原因，而不是裸异常页。

    平台健康检查只看 200，调用方需要看到失败原因才排得动。
    """
    rt = _StubRuntime(raise_exc=RuntimeError("后处理炸了"))
    r = _client(runtime=rt).post("/predict", files=_png())
    assert r.status_code == 500
    assert "后处理炸了" in r.json()["detail"]


# ------------------------------------------------- 整幅任务也保持契约形状


def test_fullmap_task_still_returns_empty_detections():
    """semantic/depth 没有框，但 detections 必须是**空列表**而不是缺字段。"""
    rt = _StubRuntime(task="semantic", detections=[])
    r = _client(runtime=rt).post("/predict", files=_png())
    body = r.json()
    assert body["detections"] == []
    assert body["task"] == "semantic"
    assert body["success"] is True
