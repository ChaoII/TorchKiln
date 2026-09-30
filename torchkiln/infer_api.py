"""图像推理的可复用 API（CLI ``predict`` 与 HTTP ``serve`` 共用同一份实现）。

为什么要有这一层：``tools/infer/predict_yolo.py`` 原本把
「letterbox -> 前向 -> 后处理 -> 映射回原图 -> 画图」全写在 ``main()`` 里，
只接受**文件路径**、只输出**可视化图片**。平台侧要提供 HTTP 推理服务就得
把 letterbox、后处理、坐标回映射、各任务的输出格式**再实现一遍**——而这正是
最容易出的错：推理侧的预处理与训练侧不一致时，结果看着"能跑"但精度悄悄
变差，排查成本极高。

所以这里把推理拆成两半，两边共用：

* :meth:`Runtime.infer` —— 只产出**结构化结果**（纯 JSON 安全），不做任何画图；
* :meth:`Runtime.visualize` —— 拿结构化结果画图，供 CLI 存图用。

这样 ``tkiln predict``（单图可视化）与 ``tkiln serve``（HTTP 服务）的数值
必然同源，不存在"两条推理路径给出不同结果"的可能。

支持的 task（与 ``predict_yolo.py`` 一致）：classify / detect / segment /
obb / semantic / depth / lane_seg / lane_row。OCR、点云类任务的推理入口
仍是各自独立脚本（det+rec 双模型、单目 3D 需要投影），本模块**不覆盖**——
宁可显式报"不支持"，也不要拿一个近似实现顶替。

示例::

    from torchkiln.infer_api import Runtime

    rt = Runtime.load("yolo11-det", "output/x/best_accuracy.pth", device="cuda:0")
    res = rt.infer(img_bgr)                 # img_bgr: HxWx3 uint8 (BGR)
    print(res.data["detections"])           # JSON 安全的结构化结果
    cv2.imwrite("out.jpg", rt.visualize(img_bgr, res))
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import os
import sys

import cv2
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ptcore.config_schema import resolve_config  # noqa: E402
from ptcore.precision import enable_paddle_like_precision  # noqa: E402
from ptcore.pretrained import resolve_pretrained  # noqa: E402
from torchkiln.det.ops import letterbox  # noqa: E402
from torchkiln.det.rbox import rbox2poly_np  # noqa: E402
from torchkiln.ocr.utils.config import flatten_opts, parse_args_to_config  # noqa: E402
from torchkiln.trainer import build_task  # noqa: E402

enable_paddle_like_precision()

#: :meth:`Runtime.infer` 支持的 task。其余 task 的推理入口形态不同
#: （OCR 是 det+rec 双模型、点云类要投影），显式拒绝好过拿近似实现顶替。
SUPPORTED_TASKS = (
    "classify", "detect", "segment", "obb",
    "semantic", "depth", "lane_seg", "lane_row",
)

#: 后处理需要原图尺寸的 task（输出是整幅 mask / 深度图 / 车道图，不是框）
_FULLMAP_TASKS = ("semantic", "depth", "lane_seg")


def _color(i):
    """按类别 id 生成稳定的 BGR 颜色（与 ``predict_yolo.py`` 保持一致）。"""
    rng = np.random.RandomState(int(i) * 9973 + 17)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def _r(x, nd=4):
    """转成 JSON 安全的 float（numpy 类型不能直接 json 序列化）。"""
    return round(float(x), nd)


def _listify(v):
    """tensor / ndarray -> list[float]。"""
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return [float(t) for t in np.asarray(v).reshape(-1)]


def _png_b64(arr_u8):
    """整幅灰度图 -> PNG 的 base64（客户端直接 <img src="data:image/png;base64,...">）。"""
    ok, buf = cv2.imencode(".png", arr_u8)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


class InferResult(object):
    """一次推理的结果。

    :attr:`data` 是 **JSON 安全**的 dict（HTTP 服务直接返回它）；
    :attr:`viz` 存画图需要的中间量（掩膜/深度图/车道 x 等 numpy 数组），
    只给 :meth:`Runtime.visualize` 用，不进 JSON。
    """

    __slots__ = ("data", "viz", "notes")

    def __init__(self, data, viz=None, notes=None):
        self.data = data
        self.viz = viz or {}
        self.notes = notes or []

    def __repr__(self):
        return "InferResult({})".format(self.data)


class Runtime(object):
    """加载好权重与配置的推理运行时（进程内常驻，线程安全由调用方保证）。"""

    def __init__(self, config, task_name, names, size, task, post, model, device,
                 config_path=None, weights=None):
        self.config = config
        self.task_name = task_name
        self.names = names
        self.size = size
        self.task = task
        self.post = post
        self.model = model
        self.device = device
        self.config_path = config_path
        self.weights = weights

    # ------------------------------------------------------------------ 加载

    @classmethod
    def load(cls, config, weights, device="cuda:0", overrides=None):
        """加载运行时。

        :param config: 配置**路径或模型名**（走 :func:`resolve_config`，与
            ``tkiln schema`` 同一套解析，所以 ``yolo11-det`` 和
            ``configs/yolo/yolo11-det.yml`` 等价）。
        :param weights: 权重路径（支持 ``best_accuracy.pth`` 之外的官方权重名，
            内部经 :func:`resolve_pretrained` 解析）。
        :param device: ``cuda:0`` / ``cpu``；CUDA 不可用时自动退回 CPU。
        :param overrides: ``-o Key.Sub=value`` 形式的超参覆盖（通常用不到，
            部署时用训练那份配置原样加载即可）。
        """
        path, _how = resolve_config(config)
        cfg = parse_args_to_config(path, flatten_opts(overrides or []))
        task_name = (cfg.get("Architecture") or {}).get("task", "classify")
        if task_name not in SUPPORTED_TASKS:
            raise ValueError(
                "task={!r} 不在本模块的推理服务支持范围内（支持 {}）。"
                "OCR/点云类任务的推理入口形态不同（多模型/需投影），"
                "请用对应的 tools/infer/predict_*.py".format(task_name, ", ".join(SUPPORTED_TASKS))
            )
        dev = torch.device(
            device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

        task = build_task(cfg)
        post = task.build_post_process(cfg)
        model = task.build_model(cfg, post)
        wpath = resolve_pretrained(weights)
        if not wpath:
            raise FileNotFoundError("Could not resolve weights: {}".format(weights))
        state = torch.load(wpath, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        own = model.state_dict()
        # 与 predict_yolo.py 同一套宽松加载：shape 不一致的层（多任务头、预训练
        # 权重里多出来的键）直接跳过，不因 strict 不匹配而整个加载失败。
        state = {k: v for k, v in state.items()
                 if k in own and tuple(own[k].shape) == tuple(v.shape)}
        model.load_state_dict(state, strict=False)
        model.eval().to(dev)

        ds = (cfg.get("Train", {}).get("dataset") or {})
        size = int((ds.get("transform") or {}).get("image_size", 640))
        return cls(cfg, task_name, ds.get("names"), size, task, post, model, dev,
                   config_path=path, weights=wpath)

    @property
    def model_name(self):
        return (self.config.get("Global") or {}).get("model_name") or \
            os.path.splitext(os.path.basename(self.config_path or ""))[0]

    @property
    def num_classes(self):
        if isinstance(self.names, dict):
            return len(self.names)
        if isinstance(self.names, (list, tuple)):
            return len(self.names)
        return 0

    def class_name(self, idx):
        """类别 id -> 名称（配置里没写 names 时退回 id 本身）。"""
        if isinstance(self.names, dict):
            # 允许 {0: "cat", 1: "dog"} 或 {"0": ...} 两种写法
            return str(self.names.get(int(idx), self.names.get(str(int(idx)), idx)))
        if isinstance(self.names, (list, tuple)) and 0 <= int(idx) < len(self.names):
            return str(self.names[int(idx)])
        return str(int(idx))

    def describe(self):
        """服务自描述信息（``/health`` 与日志用）。"""
        return {
            "model_name": self.model_name,
            "task": self.task_name,
            "image_size": self.size,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "config_path": self.config_path,
        }

    # ------------------------------------------------------------------ 推理

    def infer(self, img0):
        """对一张 BGR 图推理，返回 :class:`InferResult`。

        :param img0: HxWx3 uint8 **BGR**（OpenCV 惯例）原图。
        """
        if img0 is None or not isinstance(img0, np.ndarray) or img0.size == 0:
            raise ValueError("empty image")
        if img0.ndim == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        if img0.dtype != np.uint8:
            img0 = np.clip(img0, 0, 255).astype(np.uint8)

        h0, w0 = img0.shape[:2]
        img, ratio, pad = letterbox(img0, self.size)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device)

        with torch.no_grad():
            raw = self.model(x)
            if self.task_name in _FULLMAP_TASKS:
                res = self.post(raw, size=(img.shape[0], img.shape[1]))
            else:
                res = self.post(raw)

        base = {"image_width": int(w0), "image_height": int(h0),
                "task": self.task_name, "model_name": self.model_name}
        fn = getattr(self, "_infer_" + self.task_name, None)
        if fn is None:
            raise ValueError("unsupported task: {!r}".format(self.task_name))
        data, viz, notes = fn(res, raw, img, ratio, pad, h0, w0, base)
        return InferResult(data, viz, notes)

    # -- 各 task 的结构化输出 -------------------------------------------

    def _infer_classify(self, res, raw, img, ratio, pad, h0, w0, base):
        from torchkiln.tasks._cls import is_multi_label_loss_cfg

        multi = ("MultiLabel" in type(self.post).__name__
                 or is_multi_label_loss_cfg(self.config.get("Loss")))
        if multi:
            r = res[0]
            sc = r["scores"]
            labs = [int(i) for i in _listify(r["labels"])]
            pairs = [(self.class_name(i), _r(float(sc[i]))) for i in labs]
            notes = ["multi-label hits: " + (", ".join(
                "{}:{:.3f}".format(n, c) for n, c in pairs) or "(none)")]
        else:
            probs = torch.softmax(raw, dim=1)[0].detach().cpu().numpy()
            top = probs.argsort()[::-1][:5]
            pairs = [(self.class_name(int(i)), _r(probs[i])) for i in top]
            notes = ["top-5: " + ", ".join("{}:{:.3f}".format(n, c) for n, c in pairs)]

        # 分类没有框，但仍给出 detections[0] = top-1：通用客户端
        # （只认 detections 列表）不至于拿到空结果而误判"模型没输出"。
        dets = [{"class": n, "confidence": c} for n, c in pairs[:1]]
        base.update({"detections": dets, "labels": [
            {"class": n, "confidence": c} for n, c in pairs], "top1": pairs[0][0] if pairs else None})
        return base, {"raw": raw}, notes

    def _infer_detect(self, res, raw, img, ratio, pad, h0, w0, base):
        r = res[0]
        boxes = _to_numpy(r["bboxes"])
        scores = _to_numpy(r["scores"])
        labels = _to_numpy(r["labels"])
        dets = []
        for box, score, lab in zip(boxes, scores, labels):
            p1, p2 = _unletterbox_xyxy(box, ratio, pad)
            dets.append({"class": self.class_name(lab), "confidence": _r(score),
                         "bbox": [_r(p1[0], 2), _r(p1[1], 2), _r(p2[0], 2), _r(p2[1], 2)],
                         "label_id": int(lab)})
        base["detections"] = dets
        return base, {}, ["detections: {}".format(len(dets))]

    def _infer_segment(self, res, raw, img, ratio, pad, h0, w0, base):
        r = res[0]
        boxes = _to_numpy(r["bboxes"])
        scores = _to_numpy(r["scores"])
        labels = _to_numpy(r["labels"])
        masks = r.get("masks") if isinstance(r, dict) else None
        masks = _to_numpy(masks) if masks is not None and len(masks) else None

        dets = []
        polies = []
        for i, (box, score, lab) in enumerate(zip(boxes, scores, labels)):
            p1, p2 = _unletterbox_xyxy(box, ratio, pad)
            det = {"class": self.class_name(lab), "confidence": _r(score),
                   "bbox": [_r(p1[0], 2), _r(p1[1], 2), _r(p2[0], 2), _r(p2[1], 2)],
                   "label_id": int(lab)}
            if masks is not None and i < len(masks):
                m = _unletterbox_mask(masks[i], img, ratio, pad, w0, h0)
                poly = _mask_to_polygon(m)
                if poly:
                    det["polygon"] = poly
                    polies.append(np.array(poly, dtype=np.int32))
            dets.append(det)
        base["detections"] = dets
        return base, {"masks": polies}, ["detections: {}".format(len(dets))]

    def _infer_obb(self, res, raw, img, ratio, pad, h0, w0, base):
        r = res[0]
        boxes = _to_numpy(r["bboxes"])
        scores = _to_numpy(r["scores"])
        labels = _to_numpy(r["labels"])
        dets = []
        polys = []
        for box, score, lab in zip(boxes, scores, labels):
            bx = np.asarray(box, dtype=np.float32).copy()
            # 旋转框的中心点要减 pad 再除 scale，w/h 只需除 scale
            bx[0] = (bx[0] - pad[0]) / ratio
            bx[1] = (bx[1] - pad[1]) / ratio
            bx[2] = float(bx[2]) / ratio
            bx[3] = float(bx[3]) / ratio
            poly = rbox2poly_np(bx[None])[0].astype(np.float32)
            xs, ys = poly[:, 0], poly[:, 1]
            dets.append({
                "class": self.class_name(lab), "confidence": _r(score),
                # points 是旋转框本身；bbox 给轴对齐外接框，让只认 bbox 的
                # 通用客户端也能画出正确的位置。
                "points": [[_r(px, 2), _r(py, 2)] for px, py in poly],
                "bbox": [_r(xs.min(), 2), _r(ys.min(), 2), _r(xs.max(), 2), _r(ys.max(), 2)],
                "label_id": int(lab),
            })
            polys.append(poly.astype(np.int32))
        base["detections"] = dets
        return base, {"polys": polys}, ["detections: {}".format(len(dets))]

    def _infer_pose(self, res, raw, img, ratio, pad, h0, w0, base):
        r = res[0]
        boxes = _to_numpy(r["bboxes"])
        scores = _to_numpy(r["scores"])
        labels = _to_numpy(r["labels"])
        kpts = _to_numpy(r["kpts"]) if r.get("kpts") is not None else None
        dets = []
        for i, (box, score, lab) in enumerate(zip(boxes, scores, labels)):
            p1, p2 = _unletterbox_xyxy(box, ratio, pad)
            det = {"class": self.class_name(lab), "confidence": _r(score),
                   "bbox": [_r(p1[0], 2), _r(p1[1], 2), _r(p2[0], 2), _r(p2[1], 2)],
                   "label_id": int(lab)}
            if kpts is not None and i < len(kpts):
                kp = np.asarray(kpts[i], dtype=np.float32)  # (nk, 2 或 3)
                pts = []
                for k in range(kp.shape[0]):
                    x = (float(kp[k][0]) - pad[0]) / ratio
                    y = (float(kp[k][1]) - pad[1]) / ratio
                    sc = float(kp[k][2]) if kp.shape[1] > 2 else 1.0
                    pts.append({"x": _r(x, 2), "y": _r(y, 2), "score": _r(sc, 3)})
                det["keypoints"] = pts
            dets.append(det)
        base["detections"] = dets
        return base, {}, ["detections: {}".format(len(dets))]

    def _infer_semantic(self, res, raw, img, ratio, pad, h0, w0, base):
        mask = _to_map(res[0]).astype(np.uint8)
        mask = _unletterbox_mask(mask, img, ratio, pad, w0, h0, nearest=True)
        classes = sorted(int(c) for c in np.unique(mask))
        base.update({
            "detections": [],  # 整幅分割没有"目标列表"，保持字段存在以便客户端统一处理
            "mask_png": _png_b64(mask),
            "classes": [{"id": c, "class": self.class_name(c)} for c in classes],
            "mask_width": int(mask.shape[1]), "mask_height": int(mask.shape[0]),
        })
        return base, {"mask": mask}, ["classes: {}".format(classes)]

    def _infer_lane_seg(self, res, raw, img, ratio, pad, h0, w0, base):
        mask = _to_map(res[0]).astype(np.uint8)
        mask = _unletterbox_mask(mask, img, ratio, pad, w0, h0, nearest=True)
        classes = sorted(int(c) for c in np.unique(mask))
        base.update({
            "detections": [],
            "mask_png": _png_b64(mask),
            "classes": [{"id": c, "class": self.class_name(c)} for c in classes],
            "mask_width": int(mask.shape[1]), "mask_height": int(mask.shape[0]),
        })
        return base, {"mask": mask}, ["classes: {}".format(classes)]

    def _infer_lane_row(self, res, raw, img, ratio, pad, h0, w0, base):
        xs = _to_map(res[0]).astype(np.float32)  # (L, R)，每行是归一化 x
        lanes = []
        lines = []
        for li in range(xs.shape[0]):
            pts = []
            row = []
            for ri in range(xs.shape[1]):
                xv = float(xs[li, ri])
                row.append(xv)
                if xv < 0:
                    continue
                # y 由行号隐含：第 ri 行落在整图高度的 ri/R 处
                pts.append([_r(xv * (w0 - 1), 2),
                            _r(ri / max(xs.shape[1] - 1, 1) * (h0 - 1), 2)])
            if len(pts) >= 2:
                lanes.append({"lane_id": li, "points": pts, "row_x": [_r(v, 4) for v in row]})
                lines.append(np.array(pts, dtype=np.int32))
        base.update({"detections": [], "lanes": lanes})
        return base, {"lines": lines}, ["lane_row lanes: {}, rows: {}".format(xs.shape[0], xs.shape[1])]

    def _infer_depth(self, res, raw, img, ratio, pad, h0, w0, base):
        d = _to_map(res[0]).astype(np.float32)
        d = _unletterbox_mask(d, img, ratio, pad, w0, h0, nearest=False)
        dmin, dmax = float(d.min()), float(d.max())
        dn = ((d - dmin) / max(1e-6, dmax - dmin) * 255).astype(np.uint8)
        base.update({
            "detections": [],
            "depth": {"min": _r(dmin), "max": _r(dmax), "mean": _r(float(d.mean()))},
            "depth_png": _png_b64(dn),
        })
        return base, {"depth": d}, ["depth range: {:.3f} .. {:.3f} m".format(dmin, dmax)]

    # ------------------------------------------------------------------ 画图

    def visualize(self, img0, result):
        """把 :class:`InferResult` 画到原图上（CLI 存图用；服务不调它）。"""
        vis = img0.copy()
        viz = result.viz
        t = self.task_name
        if t == "classify":
            txt = (", ".join("{}:{:.3f}".format(d["class"], d["confidence"])
                            for d in result.data.get("labels", [])[:5]) or "none")
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif t == "detect":
            for d in result.data["detections"]:
                c = _color(d["label_id"])
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                cv2.putText(vis, "{} {:.2f}".format(d["class"], d["confidence"]),
                            (int(x1), max(12, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        elif t == "segment":
            for d, poly in zip(result.data["detections"], viz.get("masks", [])):
                c = _color(d["label_id"])
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                cv2.putText(vis, "{} {:.2f}".format(d["class"], d["confidence"]),
                            (int(x1), max(12, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                if poly is not None and len(poly):
                    cv2.polylines(vis, [poly], True, c, 1)
        elif t == "obb":
            for d, poly in zip(result.data["detections"], viz.get("polys", [])):
                c = _color(d["label_id"])
                cv2.polylines(vis, [poly], True, c, 2)
                cv2.putText(vis, "{} {:.2f}".format(d["class"], d["confidence"]),
                            (int(poly[0][0]), max(12, int(poly[0][1]) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        elif t == "pose":
            for d in result.data["detections"]:
                c = _color(d["label_id"])
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                cv2.putText(vis, "{} {:.2f}".format(d["class"], d["confidence"]),
                            (int(x1), max(12, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                for kp in d.get("keypoints", []):
                    cv2.circle(vis, (int(kp["x"]), int(kp["y"])), 3, c, -1)
        elif t in ("semantic", "lane_seg"):
            mask = viz.get("mask")
            if mask is not None:
                colored = np.zeros_like(vis)
                for c in np.unique(mask):
                    if int(c) == 0:
                        continue
                    colored[mask == c] = _color(int(c))
                vis = cv2.addWeighted(vis, 0.5, colored, 0.5, 0)
        elif t == "lane_row":
            for line in viz.get("lines", []):
                c = _color(len(line) % 97)
                for a, b in zip(line, line[1:]):
                    cv2.line(vis, tuple(a), tuple(b), c, 2, cv2.LINE_AA)
        elif t == "depth":
            d = viz.get("depth")
            if d is not None:
                dmin, dmax = float(d.min()), float(d.max())
                dn = ((d - dmin) / max(1e-6, dmax - dmin) * 255).astype(np.uint8)
                vis = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
        return vis


# ---------------------------------------------------------------------- 工具


def _to_numpy(v):
    if v is None:
        return None
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _to_map(v):
    """整幅输出（语义掩膜 / 深度图 / 车道图）-> ``(H, W)`` 或 ``(L, R)``。

    ⚠️ 只去掉 batch 维，不能用 ``np.squeeze``：全量 squeeze 会把
    ``lane_row`` 的 ``(1, 3)``（1 条车道 x 3 行）压成一维，后面按行取就 IndexError；
    而不减 batch 维时 ``cv2.resize`` 会把 ``(1, H, W)`` 当成「H 行 W 列、每像素
    1 个通道」的 3 通道图，resize 完变成 ``(H, W, 1)``，到 ``cv2.imencode``
    才炸（``channels == 1 || 3 || 4`` 断言失败）——错误发生在离病因很远的地方。
    """
    m = np.asarray(_to_numpy(v))
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    return m


def _unletterbox_xyxy(box, ratio, pad):
    """letterbox 后的 xyxy -> 原图坐标。"""
    b = np.asarray(box, dtype=np.float32)
    p1 = ((b[0] - pad[0]) / ratio, (b[1] - pad[1]) / ratio)
    p2 = ((b[2] - pad[0]) / ratio, (b[3] - pad[1]) / ratio)
    return p1, p2


def _unletterbox_mask(mask, img, ratio, pad, w0, h0, nearest=True):
    """letterbox 后的整幅 mask/深度图 -> 原图尺寸。

    顺序与 ``predict_yolo.py`` 一致：先缩到 letterbox 尺寸 -> 裁掉 pad ->
    再缩到原图尺寸。少了中间那步裁剪，pad 区域会污染边缘一圈像素。

    ⚠️ 必须先转 dtype：分割任务的 ``masks`` 是 ``torch.bool``，而 OpenCV 的
    resize 明确不支持 bool（``src data type = bool is not supported``）。
    最近邻走 uint8（掩膜/类别索引），线性插值走 float32（深度图）。
    """
    m = np.asarray(mask)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    m = m.astype(np.uint8) if nearest else m.astype(np.float32)
    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=interp)
    m = m[int(pad[1]):int(pad[1] + h0 * ratio), int(pad[0]):int(pad[0] + w0 * ratio)]
    m = cv2.resize(m, (w0, h0), interpolation=interp)
    return m


def _mask_to_polygon(mask, min_area=8.0):
    """二值 mask -> 最大外轮廓多边形（原图坐标系，客户端可直接画）。

    返回 ``[[x, y], ...]``；找不到足够大的轮廓时返回 ``[]``。
    比在 JSON 里塞整幅掩膜小几个数量级，标注前端也正是要多边形。
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.max() == 0:
        return []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return []
    return [[int(p[0][0]), int(p[0][1])] for p in c]
