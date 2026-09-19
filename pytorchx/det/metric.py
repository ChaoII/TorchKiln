"""Detection metric: COCO-style mAP (IoU 0.50:0.05:0.95), 101-point AP."""
from __future__ import absolute_import

import numpy as np

__all__ = ["DetMetric"]


def _iou_matrix(box, boxes):
    """IoU of one (4,) xyxy box against (N,4) xyxy boxes."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return inter / (area_a + area_b - inter + 1e-9)


def _ap_101(recall, precision):
    """COCO 101-point interpolated AP (aligned with ultralytics ``compute_ap``)."""
    if recall.size == 0:
        return 0.0
    last_r = float(recall[-1]) if recall.size else 1.0
    mrec = np.concatenate(([0.0], recall, [last_r], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    grid = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(grid, mrec, mpre), grid))


class DetMetric(object):
    def __init__(
        self,
        iou_thresholds=None,
        main_indicator="mAP50-95",
        box_format="xyxy",
        **kwargs
    ):
        self.iou_thresholds = (
            list(iou_thresholds)
            if iou_thresholds is not None
            else [round(0.5 + 0.05 * i, 2) for i in range(10)]
        )
        self.main_indicator = main_indicator
        self.box_format = box_format
        self.reset()

    def reset(self):
        self.preds = []
        self.gts = []

    def _iou(self, box, boxes):
        """IoU of one box against many, honouring ``box_format``."""
        if self.box_format == "xywhr":
            from pytorchx.det.rbox import probiou

            import torch as _t

            if boxes.shape[0] == 0:
                return np.zeros((0,), dtype=np.float32)
            b1 = _t.from_numpy(box.reshape(1, 5).astype(np.float32))
            b2 = _t.from_numpy(boxes.astype(np.float32))
            with _t.no_grad():
                iou = probiou(b1, b2).numpy()
            return iou.astype(np.float32)
        return _iou_matrix(box, boxes)

    def __call__(self, post_result, batch):
        targets = batch[1]
        mask = batch[2]
        targets = targets.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        for i, pred in enumerate(post_result):
            self.preds.append(
                {
                    "bboxes": pred["bboxes"].detach().cpu().numpy().astype(np.float32),
                    "scores": pred["scores"].detach().cpu().numpy().astype(np.float32),
                    "labels": pred["labels"].detach().cpu().numpy().astype(np.int64),
                }
            )
            m = mask[i].astype(bool)
            self.gts.append(targets[i][m].astype(np.float32))  # (N, 1+box_dim)

    def get_metric(self):
        num_images = len(self.gts)
        n_classes = 0
        for g in self.gts:
            if g.shape[0]:
                n_classes = max(n_classes, int(g[:, 0].max()) + 1)
        for p in self.preds:
            if p["labels"].size:
                n_classes = max(n_classes, int(p["labels"].max()) + 1)

        per_thr = {}
        for thr in self.iou_thresholds:
            aps = []
            for c in range(n_classes):
                scores, tps, n_gt = [], [], 0
                for i in range(num_images):
                    g = self.gts[i]
                    gt_c = g[g[:, 0] == c][:, 1:] if g.shape[0] else g.reshape(0, 5)
                    n_gt += gt_c.shape[0]

                    p = self.preds[i]
                    sel = p["labels"] == c
                    pb, ps = p["bboxes"][sel], p["scores"][sel]
                    if pb.shape[0]:
                        order = np.argsort(-ps)
                        pb, ps = pb[order], ps[order]

                    matched = np.zeros(gt_c.shape[0], dtype=bool)
                    for j in range(pb.shape[0]):
                        # 对齐 ultralytics:只在**未匹配**的 GT 中找最优,
                        # 否则已匹配 GT 会抢占而漏掉其它满足阈值的 GT,系统性压低 mAP。
                        if gt_c.shape[0]:
                            ious = np.array(self._iou(pb[j], gt_c), dtype=np.float32)
                            ious[matched] = -1.0
                            k = int(ious.argmax())
                        else:
                            ious = np.zeros((0,), dtype=np.float32)
                            k = -1
                        if k >= 0 and ious[k] > thr:
                            matched[k] = True
                            tps.append(1)
                        else:
                            tps.append(0)
                        scores.append(ps[j])
                if n_gt == 0:
                    continue
                if not scores:
                    aps.append(0.0)
                    continue
                order = np.argsort(-np.array(scores))
                tp = np.array(tps, dtype=np.float32)[order]
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(1.0 - tp)
                recall = cum_tp / max(n_gt, 1)
                precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
                aps.append(_ap_101(recall, precision))
            per_thr[thr] = float(np.mean(aps)) if aps else 0.0

        metrics = {"mAP50": per_thr.get(0.5, 0.0)}
        metrics["mAP50-95"] = (
            float(np.mean(list(per_thr.values()))) if per_thr else 0.0
        )
        metrics["mAP75"] = per_thr.get(0.75, 0.0)
        return metrics
