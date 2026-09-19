from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import math

import torch

__all__ = ["build_optimizer", "build_lr_scheduler"]


def _paddle_cosine_lambda(base_lr, total_steps, warmup_steps, lrf=None):
    """Cosine LR schedule with linear warmup.

    Without ``lrf`` this replicates Paddle's ``CosineAnnealingDecay`` wrapped in
    ``LinearWarmup`` (LR decays to 0):

      - k == 0 and warmup_steps > 0  -> 0.0
      - 1 <= k <  warmup_steps       -> base_lr * k / warmup_steps
      - k >= warmup_steps            -> base_lr * 0.5*(1+cos(pi*(k-warmup)/T))

    With ``lrf`` (e.g. 0.01, the YOLO convention) the cosine decays from 1.0 to
    ``lrf`` instead of 0.
    """

    def lr_lambda(step):
        if warmup_steps > 0:
            if step == 0:
                return 0.0
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            inner = step - warmup_steps
        else:
            inner = step
        if total_steps <= 0:
            return 1.0
        cos = 0.5 * (1.0 + math.cos(math.pi * float(inner) / float(total_steps)))
        if lrf is None:
            return cos
        return lrf + (1.0 - lrf) * cos

    return lr_lambda


def _linear_lambda(total_steps, warmup_steps, lrf):
    def lr_lambda(step):
        """全程线性衰减(对齐 ultralytics ``lf(epoch) = max(1-x/epochs,0)*(1-lrf)+lrf``)。

        warmup 由 ``_GroupWarmupLR`` 负责,其终点 = ``initial_lr * lf(epoch)``(含 epoch 衰减),
        因此这里**不**再内嵌 warmup 段,否则 LR 会在 warmup 升到全量 lr0。
        """

        if total_steps <= 0:
            return 1.0
        frac = min(1.0, float(step) / float(total_steps))
        return 1.0 + (lrf - 1.0) * frac

    return lr_lambda


def _build_base_lr_scheduler(lr_config, epochs, step_each_epoch, optimizer):
    lr_config = copy.deepcopy(lr_config)
    lr_name = lr_config.pop("name", "Const")
    base_lr = float(lr_config.get("learning_rate", 0.001))
    warmup_epoch = lr_config.get("warmup_epoch", 0)
    total_steps = int(step_each_epoch) * int(epochs)
    warmup_steps = round(warmup_epoch * step_each_epoch)
    lrf = lr_config.get("lrf", None)
    lrf = float(lrf) if lrf is not None else None

    warmup_bias_lr = float(lr_config.get("warmup_bias_lr", 0.1) or 0.0)
    warmup_momentum = lr_config.get("warmup_momentum")

    if lr_name in ("Cosine", "cos_lr"):
        lr_lambda = _paddle_cosine_lambda(base_lr, total_steps, warmup_steps, lrf)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if lr_name == "Linear":
        lr_lambda = _linear_lambda(total_steps, warmup_steps, 0.01 if lrf is None else lrf)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if lr_name == "OneCycle":
        max_lr = float(lr_config.get("max_lr", base_lr))
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=max(1, total_steps),
            pct_start=float(lr_config.get("pct_start", 0.1)),
        )
    if lr_name in ("Const", "Constant"):
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    raise NotImplementedError("lr scheduler {} not supported yet".format(lr_name))


def build_optimizer(config, epochs, step_each_epoch, model):
    """Build ``(optimizer, lr_scheduler)``.

    Supported optimizers: ``Adam`` (PaddleOCR default), ``AdamW``, ``SGD``,
    ``RAdam``, ``NAdam``. Schedulers: ``Cosine`` (Paddle semantics, optional
    ``lrf``), ``Linear``, ``OneCycle``, ``Const``.
    """
    config = copy.deepcopy(config)
    lr_config = config.pop("lr")
    base_lr = float(lr_config.get("learning_rate", 0.001))

    # regularization (L2 weight decay)
    weight_decay = 0.0
    if "regularizer" in config and config["regularizer"] is not None:
        reg_config = config.pop("regularizer")
        reg_name = reg_config.pop("name")
        if not reg_name.endswith("Decay"):
            reg_name += "Decay"
        assert reg_name == "L2Decay", "only L2Decay regularizer is supported"
        weight_decay = float(reg_config.get("factor", 0.0))
    elif "weight_decay" in config:
        weight_decay = float(config.pop("weight_decay"))

    optim_name = config.pop("name")
    # ---- ultralytics 对齐 ----
    # (1) 名义批量 nbs:lr/momentum/weight_decay 按 batch*accumulate/nbs 缩放
    nbs = float(config.get("nbs") or 0.0)
    batch = float(
        (((config.get("Train") or {}).get("dataset") or {}).get("loader") or {}).get(
            "batch_size_per_card", 0
        ) or 0
    )
    if nbs and batch:
        accumulate = max(round(nbs / batch), 1)
        scale = batch * accumulate / nbs
    else:
        scale = 1.0
    base_lr = base_lr * scale
    weight_decay = weight_decay * scale
    # (2) 参数分组:ndim==1(BN 权重/gamma、所有 bias)不做 weight decay
    # (2) 参数分组(ultralytics 顺序):0=bias(不 decay,且 warmup 到 warmup_bias_lr)
    g_bias, g_weight, g_bn = [], [], []
    import torch.nn as _nn
    _bn_types = tuple(v for k, v in _nn.__dict__.items() if isinstance(v, type) and 'Norm' in k)
    for name, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        if name.endswith('.bias'):
            g_bias.append(p_)
        elif isinstance(dict(model.named_modules()).get(name.rsplit('.', 1)[0]), _bn_types):
            g_bn.append(p_)
        else:
            g_weight.append(p_)
    param_groups = [
        {'params': g_bias, 'weight_decay': 0.0},                          # 0: bias
        {'params': g_weight, 'weight_decay': weight_decay},               # 1: 权重
        {'params': g_bn, 'weight_decay': 0.0},                            # 2: BN
    ]
    beta1 = float(config.get("beta1", 0.9))
    beta2 = float(config.get("beta2", 0.999))
    eps = float(config.get("epsilon", config.get("eps", 1e-8)))
    momentum = float(config.get("momentum", 0.937))

    if optim_name == "Adam":
        optimizer = torch.optim.Adam(
            param_groups, lr=base_lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
        )
    elif optim_name in ("AdamW", "adamw"):
        optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
        )
    elif optim_name in ("SGD", "sgd"):
        optimizer = torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=momentum,
            nesterov=bool(config.get("nesterov", True)),
            weight_decay=weight_decay,
        )
    elif optim_name in ("RAdam", "radam"):
        optimizer = torch.optim.RAdam(
            param_groups, lr=base_lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
        )
    elif optim_name in ("NAdam", "nadam"):
        optimizer = torch.optim.NAdam(
            param_groups, lr=base_lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
        )
    else:
        raise NotImplementedError("optimizer {} not supported yet".format(optim_name))

    lr_scheduler = build_lr_scheduler(lr_config, epochs, step_each_epoch, optimizer)
    return optimizer, lr_scheduler


class _GroupWarmupLR(object):
    """ultralytics 风格 warmup:bias 组从 warmup_bias_lr 降到 lr0,其余从 0 升到 lr0;
    同时把 momentum 从 warmup_momentum 线性升到目标值。warmup 结束后交给基础调度器。"""

    def __init__(self, base, optimizer, warmup_steps, warmup_bias_lr, warmup_momentum):
        self.base = base
        self.optimizer = optimizer
        self.warmup_steps = max(int(warmup_steps), 0)
        self.warmup_bias_lr = float(warmup_bias_lr or 0.0)
        self.warmup_momentum = warmup_momentum
        self.step_count = 0
        # LambdaLR 构造时会把各组 lr 重置为 lr*lambda(0)=0,因此必须取
        # ``initial_lr``(构造时保存的原始 lr),否则 warmup 会把权重组从 0 插值到 0。
        self._initial = [
            float(g.get("initial_lr", g.get("lr", 0.0)))
            for g in optimizer.param_groups
        ]
        self._momenta = [g.get("momentum") for g in optimizer.param_groups]

    def _interp(self, t, a, b):
        if self.warmup_steps <= 0:
            return b
        r = min(max(t, 0), self.warmup_steps) / float(self.warmup_steps)
        return a + (b - a) * r

    def step(self, *a, **k):
        self.step_count += 1
        t = self.step_count
        # 始终推进基础调度器,保证 warmup 结束后相位正确(否则基础调度器
        # 计数器停在 0,warmup 后等于再走一遍 warmup,LR 远低于设定值)。
        self.base.step(*a, **k)
        if self.warmup_steps and t <= self.warmup_steps:
            # 对齐 ultralytics `set_lr`:warmup 终点 = ``initial_lr * lf(epoch)``(随 epoch 衰减),
            # 而非固定 lr0。base.get_last_lr() 已是当前 epoch 的目标 lr。
            target = self.base.get_last_lr()
            for i, g in enumerate(self.optimizer.param_groups):
                start = self.warmup_bias_lr if i == 0 else 0.0     # 0 组 = bias
                g["lr"] = self._interp(t, start, target[i])
                if self._momenta[i] is not None and self.warmup_momentum is not None:
                    g["momentum"] = self._interp(t, float(self.warmup_momentum),
                                                  float(self._momenta[i]))
            return None
        if self.warmup_momentum is not None:
            for i, g in enumerate(self.optimizer.param_groups):
                if self._momenta[i] is not None:
                    g["momentum"] = self._momenta[i]
        return None

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    get_lr = get_last_lr

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, d):
        return self.base.load_state_dict(d)


def build_lr_scheduler(lr_config, epochs, step_each_epoch, optimizer):
    """在基础调度器外包一层 ultralytics 风格的分组 warmup(bias 组 warmup_bias_lr)。"""
    base = _build_base_lr_scheduler(lr_config, epochs, step_each_epoch, optimizer)
    warmup_epoch = float(lr_config.get("warmup_epoch", 0) or 0)
    warmup_steps = round(warmup_epoch * step_each_epoch)
    if warmup_steps <= 0:
        return base
    return _GroupWarmupLR(
        base, optimizer, warmup_steps,
        lr_config.get("warmup_bias_lr", 0.1),
        lr_config.get("warmup_momentum", None),
    )
