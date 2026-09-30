"""``tkiln serve``：把权重挂成 HTTP 推理服务。

平台（AIStation）的「部署」需要的是一个**长期运行的推理端点**，而
``tools/infer/predict_yolo.py`` 是单图脚本、只吐可视化图片。早期平台侧的做法是
自己生成一份推理脚本重写 letterbox/后处理/坐标回映射——那样推理与训练的预处理
很容易对不齐，表现为"能跑但精度悄悄变差"，极难排查。

这里改成：推理逻辑只有 :mod:`torchkiln.infer_api` 一份，``predict`` 与 ``serve``
都调它；本模块只负责 HTTP 外壳（解码图片、鉴权、计时、序列化）。

接口与平台既有的 YOLO/PaddleX 推理服务保持兼容，因此平台侧的调用方、文档、
前端都不需要改：

* ``GET  /health``  -> ``{"status": "ok", "model_name": ...}``
* ``POST /predict``  -> ``multipart/form-data``，字段名 ``file``，
  头 ``X-API-Key``（设了才校验），返回::

      {"success": true, "detections": [...], "image_width": W, "image_height": H,
       "inference_time_ms": 12.3, "task": "detect", "model_name": "..."}

  ``detections`` 每项恒含 ``class`` / ``confidence``，几何字段按任务给：
  ``bbox``（detect/segment/pose）、``polygon``（segment）、``points``（obb）、
  ``keypoints``（pose）。整幅输出的任务（semantic / depth / lane_seg /
  lane_row）没有框，改在同级给 ``mask_png`` / ``depth_png`` / ``lanes``
  （base64 PNG 或坐标列表），``detections`` 仍返回空列表以便客户端统一处理。

用法::

    tkiln serve -c configs/yolo/yolo11-det.yml --weights output/x/best_accuracy.pth
    tkiln serve -c yolo11-seg --weights w.pth --port 9000 --api-key secret
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import cv2
import numpy as np


def build_app(runtime, api_key=None, title="TorchKiln Inference"):
    """用已加载好的 :class:`~torchkiln.infer_api.Runtime` 组装 FastAPI 应用。

    模型必须在**建应用之前**加载完：加载失败就抛异常让进程退出，而不是先占住
    端口、再让平台的健康检查超时——那样平台侧看到的是"服务起不来"，看不出
    到底哪一步坏了，还白占一次端口。
    """
    from fastapi import FastAPI, File, HTTPException, Security, UploadFile
    from fastapi.security import APIKeyHeader

    app = FastAPI(title=title, version="1.0")
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    info = runtime.describe()
    # 没配 api_key（None）时，auto_error=False 拿到的是 None，比较结果自然成立，
    # 即本地自测无需带头；配了才强制校验。
    expected_key = api_key

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_name": info["model_name"]}

    @app.get("/info")
    async def meta():
        return info

    @app.post("/predict")
    async def predict(file: UploadFile = File(...),
                      api_key: str = Security(key_header)):
        if api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        start = time.time()
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty file")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        try:
            res = runtime.infer(img)
        except Exception as exc:  # noqa: BLE001
            # 推理失败要回 500 + 原因，而不是让异常裸奔成 HTML 错误页：
            # 平台的健康检查只看 200，调用方需要看到人话错误。
            raise HTTPException(status_code=500, detail="inference failed: {}: {}".format(
                type(exc).__name__, exc))
        out = dict(res.data)
        out["success"] = True
        out["inference_time_ms"] = round((time.time() - start) * 1000, 1)
        return out

    app.state.api_key = api_key
    app.state.runtime = runtime
    return app


def main(argv=None):
    """``tkiln serve`` 的 CLI 入口。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="tkiln serve", description="启动 HTTP 推理服务")
    ap.add_argument("-c", "--config", required=True, help="配置路径**或模型名**")
    ap.add_argument("-o", "--opt", nargs="*", action="append", default=None,
                    help="超参覆盖，如 -o Global.conf=0.3")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--api-key", default=None, help="设置后 /predict 需要 X-API-Key 头")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    from ptcore.config import flatten_opts

    from torchkiln.infer_api import Runtime

    print("[serve] loading config={} weights={} ...".format(args.config, args.weights),
          flush=True)
    runtime = Runtime.load(args.config, args.weights, device=args.device,
                           overrides=flatten_opts(args.opt))
    print("[serve] loaded: {}".format(runtime.describe()), flush=True)
    if args.api_key:
        print("[serve] /predict 需要 X-API-Key 头", flush=True)

    import uvicorn

    uvicorn.run(build_app(runtime, api_key=args.api_key), host=args.host,
                port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
