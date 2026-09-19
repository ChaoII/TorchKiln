"""Detection post-processing: decode + confidence filter + NMS.

Supports axis-aligned boxes (``box_type="xyxy"``) and oriented boxes
(``box_type="xywhr"``), with or without DFL regression (``reg_max > 1``).
"""
from __future__ import absolute_import

import torch

from pytorchx.det.ops import dist2bbox, make_anchors, nms, split_head
from pytorchx.det.rbox import dist2rbox, nms_rotated

__all__ = ["DetPostProcess"]


class DetPostProcess(object):
    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.7,
        max_det=300,
        strides=(8, 16, 32),
        box_type="xyxy",
        reg_max=1,
        reg_layout="ltrb_angle",
        ne=1,
        reg_channels=None,
        end2end=False,
        **kwargs
    ):
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.strides = tuple(int(s) for s in strides)
        self.box_type = box_type
        self.reg_max = int(reg_max)
        self.reg_layout = reg_layout
        self.ne = int(ne)
        self.end2end = bool(end2end)
        if reg_channels is None:
            if box_type == "xywhr":
                reg_channels = (
                    4 * self.reg_max + self.ne
                    if reg_layout == "upstream"
                    else 4 + self.ne
                )
            else:
                reg_channels = 4 * self.reg_max
        self.reg_channels = int(reg_channels)

    def _mode(self):
        if self.box_type != "xywhr":
            return "det"
        return "obb" if self.reg_layout == "upstream" else "obb_ours"

    def _num_classes(self, channels):
        return channels - self.reg_channels

    def _decode(self, dist, anchor_points, stride_tensor):
        if self.box_type == "xywhr":
            return dist2rbox(dist, anchor_points, stride_tensor)
        return dist2bbox(dist, anchor_points, xywh=False) * stride_tensor

    def _nms(self, boxes, scores):
        if self.end2end:
            # the one-to-one branch is already de-duplicated
            return torch.arange(boxes.shape[0], device=boxes.device)
        if self.box_type == "xywhr":
            return nms_rotated(boxes, scores, self.iou_thres)
        return nms(boxes, scores, self.iou_thres)

    def __call__(self, preds):
        num_classes = self._num_classes(preds[0].shape[1])
        distri, scores, _ = split_head(
            preds, num_classes, self.reg_max, self._mode(), self.ne
        )
        scores = scores.sigmoid()
        anchor_points, stride_tensor = make_anchors(preds, self.strides)
        boxes = self._decode(distri, anchor_points, stride_tensor)

        results = []
        for b in range(boxes.shape[0]):
            bx, sc = boxes[b], scores[b]
            conf, labels = sc.max(-1)
            keep = conf > self.conf_thres
            bx, conf, labels = bx[keep], conf[keep], labels[keep]
            if bx.numel() == 0:
                results.append(
                    {
                        "bboxes": bx.reshape(0, bx.shape[1] if bx.dim() > 1 else 4),
                        "scores": conf,
                        "labels": labels,
                    }
                )
                continue
            kept = []
            for cls in labels.unique():
                idx = (labels == cls).nonzero(as_tuple=True)[0]
                sel = self._nms(bx[idx], conf[idx])
                kept.append(idx[sel])
            kept = torch.cat(kept) if kept else torch.empty(0, dtype=torch.long)
            if kept.numel() > self.max_det:
                order = conf[kept].argsort(descending=True)[: self.max_det]
                kept = kept[order]
            results.append(
                {
                    "bboxes": bx[kept],
                    "scores": conf[kept],
                    "labels": labels[kept],
                }
            )
        return results
