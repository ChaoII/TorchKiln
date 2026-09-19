from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch

__all__ = ["ModelEMA"]


class ModelEMA:
    """Exponential Moving Average for model parameters (torch port).

    ema_param = decay * ema_param + (1 - decay) * cur_param
    """

    def __init__(
        self,
        model,
        decay=0.9998,
        gamma=2000,
        ema_decay_type="threshold",
        ema_filter_no_grad=False,
    ):
        self.decay = decay
        self.gamma = gamma
        self.ema_decay_type = ema_decay_type
        self.step = 0
        self._decay = decay

        self.ema_black_list = set()
        if ema_filter_no_grad:
            bn_state_names = set()
            for name, layer in model.named_modules():
                if isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    prefix = name + "." if name else ""
                    bn_state_names.add(prefix + "running_mean")
                    bn_state_names.add(prefix + "running_var")
            for n, p in model.named_parameters():
                if not p.requires_grad and n not in bn_state_names:
                    self.ema_black_list.add(n)

        self.state_dict = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                # EMA 从当前模型开始(对齐 ultralytics deepcopy(model)),并统一为 fp32
                self.state_dict[k] = v.detach().clone().float()
            else:
                self.state_dict[k] = v.clone()

    def _get_decay(self):
        if self.ema_decay_type == "threshold":
            return min(self.decay, (1 + self.step) / (10 + self.step))
        elif self.ema_decay_type == "exponential":
            return self.decay * (1 - math.exp(-(self.step + 1) / self.gamma))
        else:
            return self.decay

    @torch.no_grad()
    def update(self, model):
        decay = self._get_decay()
        self._decay = decay
        model_dict = model.state_dict()
        for k, v in self.state_dict.items():
            if k not in self.ema_black_list and k in model_dict:
                cur = model_dict[k]
                if not torch.is_floating_point(cur):
                    v.copy_(cur)
                    continue
                v.mul_(decay).add_(cur.float(), alpha=(1 - decay))
        self.step += 1

    @torch.no_grad()
    def apply(self):
        if self.step == 0:
            return {k: v.clone() for k, v in self.state_dict.items()}
        state = {}
        for k, v in self.state_dict.items():
            if k in self.ema_black_list or not torch.is_floating_point(v):
                state[k] = v.clone()
            else:
                if self.ema_decay_type != "exponential":
                    v = v / (1 - self._decay**self.step)
                state[k] = v.clone()
        return state

    def state_dict_for_save(self):
        return {"ema_state": self.state_dict, "step": self.step}

    def set_state_dict(self, d):
        loaded = d.get("ema_state", d)
        for k, v in self.state_dict.items():
            if k in loaded:
                v.copy_(loaded[k].to(device=v.device, dtype=v.dtype))
        self.step = int(d.get("step", 0))
