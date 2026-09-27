"""KWS 的 max_pooling_loss —— torch 移植。

对齐 PaddleSpeech ``paddlespeech/kws/models/loss.py``（``padding_mask`` /
``fill_mask_elements`` / ``max_pooling_loss``）。

语义：
  * 命中关键词（``target[i] == j``）→ **max-pooling**：有效帧概率取最大，``-log(max)``；
    padding 位置置 0（不影响 max）
  * 其它关键词/填充（filler）→ **min-pooling**：``1-p`` 取最小，``-log(min)``；
    padding 位置置 1（不影响 min）
  * 两者都先 ``clip(1e-8, 1.0)`` 防 log(0)；最后 ``/ num_utts``
  * 同时返回 ``(loss, num_correct, acc)``（acc: 有效帧 max-logit > 0.5 且 idx 命中）

⚠️ 与 Paddle 的两处对应关系（都踩过）：
  * ``paddle.max(axis=1)`` 只返回**值** → torch 必须 ``.max(dim=1).values``
  * ``paddle.clip(a, lo, hi)`` → ``torch.clamp(a, lo, hi)``
"""
from __future__ import annotations

import torch

__all__ = ["padding_mask", "fill_mask_elements", "max_pooling_loss"]


def padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """(N,) 有效长度 -> (N, max_len) bool，**True 表示 padding**（超出有效长度）。"""
    batch_size = lengths.shape[0]
    max_len = int(lengths.max().item())
    seq = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    seq = seq.unsqueeze(0).expand(batch_size, max_len)
    return seq >= lengths.unsqueeze(1)


def fill_mask_elements(condition: torch.Tensor, value: float,
                       x: torch.Tensor) -> torch.Tensor:
    assert condition.shape == x.shape, (condition.shape, x.shape)
    values = torch.ones_like(x, dtype=x.dtype) * value
    return torch.where(condition, values, x)


def max_pooling_loss(logits: torch.Tensor, target: torch.Tensor,
                     lengths: torch.Tensor, min_duration: int = 0):
    """logits: (N, T, K) 概率；target: (N,) 关键词 id（<0 表示 filler）；lengths: (N,)。"""
    mask = padding_mask(lengths)
    num_utts = logits.shape[0]
    num_keywords = logits.shape[2]

    loss = logits.new_zeros(())
    for i in range(num_utts):
        for j in range(num_keywords):
            if int(target[i].item()) == j:
                # 命中关键词: max-pooling
                prob = logits[i, :, j]
                m = mask[i]
                if min_duration > 0:
                    m[:min_duration] = True          # 与 Paddle 同样是原地修改
                prob = fill_mask_elements(m, 0.0, prob)
                prob = prob.clamp(1e-8, 1.0)
                max_prob = prob.max()
                loss = loss + (-torch.log(max_prob))
            else:
                # 其它关键词 / filler: min-pooling
                prob = 1 - logits[i, :, j]
                prob = fill_mask_elements(mask[i], 1.0, prob)
                prob = prob.clamp(1e-8, 1.0)
                min_prob = prob.min()
                loss = loss + (-torch.log(min_prob))
    loss = loss / num_utts

    # 批内准确率
    mask2 = mask.unsqueeze(-1)
    lg = fill_mask_elements(mask2, 0.0, logits)
    max_logits = lg.max(dim=1).values                # Paddle max(1) 只返回值
    num_correct = 0
    for i in range(num_utts):
        max_p = float(max_logits[i].max().item())
        idx = int(max_logits[i].argmax(0).item())
        if max_p > 0.5 and idx == int(target[i].item()):
            num_correct += 1
        if max_p < 0.5 and int(target[i].item()) < 0:
            num_correct += 1
    acc = num_correct / num_utts
    return loss, num_correct, acc
