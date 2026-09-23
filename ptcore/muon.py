"""MuSGD / Muon optimizer, ported 1:1 from ultralytics ``ultralytics/optim/muon.py``.

ultralytics ``optimizer=auto`` 在迭代数 >10000 时选 MuSGD(lr0=0.01, momentum=0.9)：
对 ndim∈{2,4} 的参数（矩阵/卷积核）用 Newton-Schulz 正交化的 Muon 更新，
其余用 SGD；两者按 muon=0.2 / sgd=1.0 加权。
"""
from __future__ import absolute_import

import torch
from torch import optim

__all__ = ["MuSGD", "muon_update", "zeropower_via_newtonschulz5"]


def zeropower_via_newtonschulz5(G, eps=1e-7):
    """Newton-Schulz 迭代近似正交化 (对齐 ultralytics)。"""
    assert G.ndim in {2, 3}
    X = G.reshape(-1, G.size(-2), G.size(-1)).bfloat16()
    X /= X.norm(dim=(-2, -1), keepdim=True) + eps
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.transpose(-2, -1)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a)
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.reshape(G.shape)


def muon_update(grad, momentum, beta=0.95, nesterov=True):
    """Muon 更新 (对齐 ultralytics,批量正交化)。"""
    single = isinstance(grad, torch.Tensor)
    grads, momentums = ([grad], [momentum]) if single else (grad, momentum)
    torch._foreach_mul_(momentums, beta)
    torch._foreach_add_(momentums, grads, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(momentums, beta))
        torch._foreach_add_(updates, grads, alpha=1 - beta)
    else:
        updates = list(momentums)
    buckets = {}
    for i, u in enumerate(updates):
        dense = u.is_contiguous()
        nhwc = not dense and u.ndim == 4 and u.is_contiguous(memory_format=torch.channels_last)
        m = u.permute(0, 2, 3, 1).flatten(1) if nhwc else (u.flatten(1) if u.ndim > 2 else u.contiguous())
        scale = max(1, grads[i].size(-2) / grads[i].size(-1)) ** 0.5
        buckets.setdefault((m.size(1), scale, m.device, m.dtype), []).append(
            (i, m, u.stride() if dense or nhwc else None)
        )
    groups = []
    for key, items in buckets.items():
        items.sort(key=lambda t: -t[1].size(0))
        start = 0
        for j in range(1, len(items) + 1):
            if j == len(items) or items[start][1].size(0) > 16 * items[j][1].size(0):
                groups.append((key, items[start:j]))
                start = j
    for (cols, scale, device, dtype), items in groups:
        X = torch.zeros(len(items), items[0][1].size(0), cols, device=device, dtype=dtype)
        torch._foreach_add_([X[j, : m.size(0)] for j, (_, m, _) in enumerate(items)], [m for _, m, _ in items])
        X = zeropower_via_newtonschulz5(X).contiguous().to(grads[items[0][0]].dtype).mul_(scale)
        for j, (i, m, stride) in enumerate(items):
            x = X[j, : m.size(0)]
            updates[i] = x.as_strided(grads[i].shape, stride) if stride else x.reshape(grads[i].shape)
    return updates[0] if single else updates


class MuSGD(optim.Optimizer):
    """Hybrid Muon + SGD optimizer (对齐 ultralytics)。"""

    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0,
                 nesterov=False, use_muon=False, muon=0.5, sgd=0.5):
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay,
                    "nesterov": nesterov, "use_muon": use_muon}
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
            for p in params:
                if len(self.state[p]) == 0:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                    if group["use_muon"]:
                        self.state[p]["momentum_buffer_SGD"] = torch.zeros_like(p)
            if group["use_muon"]:
                updates = muon_update(
                    [p.grad for p in params],
                    [self.state[p]["momentum_buffer"] for p in params],
                    beta=momentum, nesterov=nesterov,
                )
                torch._foreach_add_(params, updates, alpha=-(lr * self.muon))
                buffers = [self.state[p]["momentum_buffer_SGD"] for p in params]
                lr *= self.sgd
            else:
                buffers = [self.state[p]["momentum_buffer"] for p in params]
            grads = [p.grad for p in params]
            if group["weight_decay"] != 0:
                grads = torch._foreach_add(grads, params, alpha=group["weight_decay"])
            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, grads)
            updates = torch._foreach_add(grads, buffers, alpha=momentum) if nesterov else buffers
            torch._foreach_add_(params, updates, alpha=-lr)
        return loss
