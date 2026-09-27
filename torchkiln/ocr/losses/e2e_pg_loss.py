"""PGNet 端到端训练的损失（torch 版，对齐 PaddleOCR ``ppocr/losses/e2e_pg_loss.py``）。

四个分量：
  score_loss   : DiceLoss(f_score, tcl_maps)                 —— 文本区域分割
  border_loss  : border(4通道) 的 Huber-on-distance 加权     —— 文本边界
  direction_loss: direction(2通道) 的同上                    —— 文本方向
  ctc_loss     : 字符图 f_char 按字符点位置 gather 后做 CTC   —— 字符识别
  loss_all = score + border + direction + 5 * ctc
"""
from __future__ import absolute_import

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .det_basic_loss import DiceLoss

__all__ = ["PGLoss", "pre_process", "org_tcl_rois"]


def org_tcl_rois(batch_size, pos_lists, pos_masks, label_lists, tcl_bs):
    """把 (pos, mask, label) 列表按 GPU 分组后，重复/丢弃到恰好 tcl_bs 个。"""
    pos_lists_, pos_masks_, label_lists_ = [], [], []
    img_bs = batch_size
    ngpu = int(batch_size / img_bs) if img_bs else 1
    img_ids = np.array(pos_lists, dtype=np.int32)[:, 0, 0].copy()
    pos_lists_split = [[] for _ in range(ngpu)]
    pos_masks_split = [[] for _ in range(ngpu)]
    label_lists_split = [[] for _ in range(ngpu)]

    for i in range(img_ids.shape[0]):
        img_id = img_ids[i]
        gpu_id = int(img_id / img_bs) if img_bs else 0
        # Robustness guard (torchkiln): PaddleOCR assumes ``img_id < batch_size``
        # because ``PGProcessTrain`` cycles its own counter with the configured
        # batch size.  If a user configures a mismatching loader batch size (or
        # a partial last batch slips through), clamp instead of crashing.
        if gpu_id >= ngpu:
            gpu_id = ngpu - 1
        img_id = img_id % img_bs if img_bs else img_id
        pos_list = pos_lists[i].copy()
        pos_list[:, 0] = img_id
        pos_lists_split[gpu_id].append(pos_list)
        pos_masks_split[gpu_id].append(pos_masks[i].copy())
        label_lists_split[gpu_id].append(copy.deepcopy(label_lists[i]))

    for i in range(ngpu):
        vp_len = len(pos_lists_split[i])
        if vp_len <= tcl_bs:
            for j in range(0, tcl_bs - vp_len):
                pos_lists_split[i].append(pos_lists_split[i][j].copy())
                pos_masks_split[i].append(pos_masks_split[i][j].copy())
                label_lists_split[i].append(copy.deepcopy(label_lists_split[i][j]))
        else:
            for _ in range(0, vp_len - tcl_bs):
                c_len = len(pos_lists_split[i])
                pop_id = np.random.permutation(c_len)[0]
                pos_lists_split[i].pop(pop_id)
                pos_masks_split[i].pop(pop_id)
                label_lists_split[i].pop(pop_id)

    for i in range(ngpu):
        pos_lists_.extend(pos_lists_split[i])
        pos_masks_.extend(pos_masks_split[i])
        label_lists_.extend(label_lists_split[i])
    return pos_lists_, pos_masks_, label_lists_


def pre_process(label_list, pos_list, pos_mask, max_text_length, max_text_nums,
                pad_num, tcl_bs):
    """打包字符点位置/掩码/标签，产出 CTC 需要的 (pos, mask, label, label_len)。"""
    label_np = label_list.cpu().numpy() if isinstance(label_list, torch.Tensor) else np.asarray(label_list)
    batch = label_np.shape[0]
    pos_np = pos_list.cpu().numpy() if isinstance(pos_list, torch.Tensor) else np.asarray(pos_list)
    mask_np = pos_mask.cpu().numpy() if isinstance(pos_mask, torch.Tensor) else np.asarray(pos_mask)

    pos_list_t, pos_mask_t, label_list_t = [], [], []
    for i in range(batch):
        for j in range(max_text_nums):
            if mask_np[i, j].any():
                pos_list_t.append(pos_np[i][j])
                pos_mask_t.append(mask_np[i][j])
                label_list_t.append(label_np[i][j])
    pos_list, pos_mask, label_list = org_tcl_rois(
        batch, pos_list_t, pos_mask_t, label_list_t, tcl_bs)

    label = []
    tt = [l.tolist() for l in label_list]
    for i in range(tcl_bs):
        k = 0
        for j in range(max_text_length):
            if tt[i][j][0] != pad_num:
                k += 1
            else:
                break
        label.append(k)
    # 注意: label_list 每项 shape (T,1)，squeeze 最后一维
    label = torch.tensor(label, dtype=torch.int64)
    pos_list = torch.tensor(np.array(pos_list))
    pos_mask = torch.tensor(np.array(pos_mask))
    label_list = torch.squeeze(torch.tensor(np.array(label_list)), dim=2)
    label_list = label_list.to(torch.int32)
    return pos_list, pos_mask, label_list, label


class PGLoss(nn.Module):
    def __init__(self, tcl_bs, max_text_length, max_text_nums, pad_num,
                 eps=1e-6, **kwargs):
        super(PGLoss, self).__init__()
        self.tcl_bs = tcl_bs
        self.max_text_nums = max_text_nums
        self.max_text_length = max_text_length
        self.pad_num = pad_num
        self.dice_loss = DiceLoss(eps=eps)

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = torch.split(l_border, [4, 1], dim=1)
        f_border_split = f_border
        b, c, h, w = l_border_norm.shape
        l_border_norm_split = l_border_norm.expand(b, 4 * c, h, w)
        b, c, h, w = l_score.shape
        l_border_score = l_score.expand(b, 4 * c, h, w)
        b, c, h, w = l_mask.shape
        l_border_mask = l_mask.expand(b, 4 * c, h, w)
        border_diff = l_border_split - f_border_split
        abs_border_diff = torch.abs(border_diff)
        border_sign = (abs_border_diff < 1.0).to(torch.float32)
        border_sign = border_sign.detach()      # stop_gradient
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (
            abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        return torch.sum(border_out_loss * l_border_score * l_border_mask) / (
            torch.sum(l_border_score * l_border_mask) + 1e-5)

    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = torch.split(
            l_direction, [2, 1], dim=1)
        f_direction_split = f_direction
        b, c, h, w = l_direction_norm.shape
        l_direction_norm_split = l_direction_norm.expand(b, 2 * c, h, w)
        b, c, h, w = l_score.shape
        l_direction_score = l_score.expand(b, 2 * c, h, w)
        b, c, h, w = l_mask.shape
        l_direction_mask = l_mask.expand(b, 2 * c, h, w)
        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = torch.abs(direction_diff)
        direction_sign = (abs_direction_diff < 1.0).to(torch.float32)
        direction_sign = direction_sign.detach()
        direction_in_loss = (
            0.5 * abs_direction_diff * abs_direction_diff * direction_sign
            + (abs_direction_diff - 0.5) * (1.0 - direction_sign))
        direction_out_loss = l_direction_norm_split * direction_in_loss
        return torch.sum(
            direction_out_loss * l_direction_score * l_direction_mask) / (
            torch.sum(l_direction_score * l_direction_mask) + 1e-5)

    def ctcloss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        f_char = f_char.permute(0, 2, 3, 1)          # (B,H,W,C)
        tcl_pos = tcl_pos.reshape(-1, 3).to(torch.int64)
        # gather_nd 等价: 高级索引
        f_tcl_char = f_char[tcl_pos[:, 0], tcl_pos[:, 1], tcl_pos[:, 2]]
        f_tcl_char = f_tcl_char.reshape(-1, 64, self.pad_num + 1)
        f_tcl_char_fg, f_tcl_char_bg = torch.split(
            f_tcl_char, [self.pad_num, 1], dim=2)
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0
        b, c, l = tcl_mask.shape
        tcl_mask_fg = tcl_mask.expand(b, c, self.pad_num * l).detach()
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * (-20.0)
        f_tcl_char_mask = torch.cat([f_tcl_char_fg, f_tcl_char_bg], dim=2)
        f_tcl_char_ld = f_tcl_char_mask.permute(1, 0, 2)   # (N,B,C)
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = torch.full((B,), N, dtype=torch.int64,
                                   device=f_tcl_char_ld.device)
        # 注意: paddle.nn.functional.ctc_loss **内部会做 log_softmax**
        # （所以 PaddleOCR 直接喂原始 logits）；torch 的 F.ctc_loss 不做，
        # 必须显式 log_softmax，否则损失会变成负数。
        cost = F.ctc_loss(
            F.log_softmax(f_tcl_char_ld, dim=2), tcl_label, input_lengths,
            label_t, blank=self.pad_num, reduction="none")
        return cost.mean()

    def forward(self, predicts, labels):
        (images, tcl_maps, tcl_label_maps, border_maps, direction_maps,
         training_masks, label_list, pos_list, pos_mask) = labels[:9]
        pos_list, pos_mask, label_list, label_t = pre_process(
            label_list, pos_list, pos_mask, self.max_text_length,
            self.max_text_nums, self.pad_num, self.tcl_bs)

        f_score = predicts["f_score"]
        f_border = predicts["f_border"]
        f_direction = predicts["f_direction"]
        f_char = predicts["f_char"]

        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        border_loss = self.border_loss(f_border, border_maps, tcl_maps,
                                       training_masks)
        direction_loss = self.direction_loss(f_direction, direction_maps,
                                             tcl_maps, training_masks)
        ctc_loss = self.ctcloss(f_char, pos_list.to(f_char.device),
                                pos_mask.to(f_char.device),
                                label_list.to(f_char.device),
                                label_t.to(f_char.device))
        loss_all = score_loss + border_loss + direction_loss + 5 * ctc_loss
        return {
            "loss": loss_all,
            "score_loss": score_loss,
            "border_loss": border_loss,
            "direction_loss": direction_loss,
            "ctc_loss": ctc_loss,
        }
