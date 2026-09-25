"""SCINet forecaster — PyTorch port of PaddleTS ``_StackedSCINetModule``.

Input is a dict ``{"past_target"}`` and output is ``(B, out_chunk_len, target_dim)``.
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

__all__ = ["SCINet"]


class _Splitter(nn.Module):
    def _even(self, x):
        return x[:, ::2, :]

    def _odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        return self._even(x), self._odd(x)


class _Interactor(nn.Module):
    def __init__(self, in_planes, kernel_size, dropout_rate, num_group, hidden_size):
        super().__init__()
        self._in_planes = in_planes
        self._kernel_size = kernel_size
        self._dropout_rate = dropout_rate
        self._hidden_size = hidden_size
        self._num_group = num_group
        self._dilation = 1
        self._split_layer = _Splitter()
        self._psi = self._build_single_internal_module()
        self._phi = self._build_single_internal_module()
        self._eta = self._build_single_internal_module()
        self._rho = self._build_single_internal_module()

    def _build_single_internal_module(self):
        if self._kernel_size % 2 == 0:
            pad_l = self._dilation * (self._kernel_size - 2) // 2 + 1
            pad_r = self._dilation * self._kernel_size // 2 + 1
        else:
            pad_l = self._dilation * (self._kernel_size - 1) // 2 + 1
            pad_r = self._dilation * (self._kernel_size - 1) // 2 + 1
        prev_size = 1
        return nn.Sequential(
            nn.ConstantPad1d((pad_l, pad_r), 0.0),
            nn.Conv1d(self._in_planes * prev_size,
                      self._in_planes * self._hidden_size,
                      self._kernel_size, dilation=self._dilation, stride=1,
                      groups=self._num_group),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=self._dropout_rate),
            nn.Conv1d(self._in_planes * self._hidden_size, self._in_planes,
                      kernel_size=3, stride=1, groups=self._num_group),
            nn.Tanh())

    def forward(self, x):
        x_even, x_odd = self._split_layer(x)
        x_even = x_even.transpose(1, 2)
        x_odd = x_odd.transpose(1, 2)
        x_scaled_even = x_even * torch.exp(self._phi(x_odd))
        x_scaled_odd = x_odd * torch.exp(self._psi(x_even))
        x_even_update = x_scaled_even + self._eta(x_scaled_odd)
        x_odd_update = x_scaled_odd - self._rho(x_scaled_even)
        return x_even_update.transpose(1, 2), x_odd_update.transpose(1, 2)


class _SCINetTree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout_rate,
                 num_group, hidden_size):
        super().__init__()
        self._current_level = current_level
        self._workingblock = _Interactor(in_planes, kernel_size, dropout_rate,
                                         num_group, hidden_size)
        if self._current_level != 0:
            self._scinet_tree_even = _SCINetTree(
                in_planes, current_level - 1, kernel_size, dropout_rate,
                num_group, hidden_size)
            self._scinet_tree_odd = _SCINetTree(
                in_planes, current_level - 1, kernel_size, dropout_rate,
                num_group, hidden_size)

    def _concat_and_realign(self, even, odd):
        even = even.transpose(0, 1)
        odd = odd.transpose(0, 1)
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        min_len = min(odd_len, even_len)
        all_time_steps = []
        for i in range(min_len):
            all_time_steps.append(even[i].unsqueeze(0))
            all_time_steps.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            all_time_steps.append(even[-1].unsqueeze(0))
        concat = torch.cat(all_time_steps, dim=0)
        return concat.transpose(0, 1)

    def forward(self, x):
        x_even_update, x_odd_update = self._workingblock(x)
        if self._current_level == 0:
            return self._concat_and_realign(x_even_update, x_odd_update)
        return self._concat_and_realign(self._scinet_tree_even(x_even_update),
                                        self._scinet_tree_odd(x_odd_update))


class SCINet(nn.Module):
    """Stacked SCINet (PaddleTS ``_StackedSCINetModule``)."""

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1,
                 num_stack=1, num_level=3, num_decoder_layer=1, concat_len=0,
                 kernel_size=5, dropout_rate=0.5, num_group=1, hidden_size=1):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._in_dim = target_dim
        self._num_stack = num_stack
        self._num_level = num_level
        self._num_decoder_layer = num_decoder_layer
        self._concat_len = concat_len
        self._div_len = self._in_chunk_len // 6
        self._overlap_len = self._in_chunk_len // 4
        self._div_num = 6
        self._encoder1 = _SCINetTree(target_dim, num_level - 1, kernel_size,
                                     dropout_rate, num_group, hidden_size)
        if self._num_stack == 2:
            self._encoder2 = _SCINetTree(target_dim, num_level - 1, kernel_size,
                                         dropout_rate, num_group, hidden_size)
        self._decoder1 = nn.Conv1d(in_chunk_len, out_chunk_len, 1, stride=1)
        self._div_projection = nn.ModuleList()
        if self._num_decoder_layer > 1:
            self._decoder1 = nn.Linear(in_chunk_len, out_chunk_len)
            for _ in range(self._num_decoder_layer - 1):
                div_proj = nn.ModuleList()
                for i in range(self._div_num):
                    lens = min(i * self._div_len + self._overlap_len, in_chunk_len) - i * self._div_len
                    div_proj.append(nn.Linear(lens, self._div_len))
                self._div_projection.append(div_proj)
        if self._num_stack == 2:
            in_ch = self._concat_len + out_chunk_len if self._concat_len > 0 \
                else in_chunk_len + out_chunk_len
            self._decoder2 = nn.Conv1d(in_ch, out_chunk_len, 1)

    def forward(self, x_dict):
        x = x_dict["past_target"]
        if "observed_cov_numeric" in x_dict:
            x = torch.cat([x, x_dict["observed_cov_numeric"]], dim=2)
        if "known_cov_numeric" in x_dict:
            past_known = x_dict["known_cov_numeric"][:, :self._in_chunk_len]
            x = torch.cat([x, past_known], dim=2)
        res1 = x
        x = self._encoder1(x)
        x = x + res1
        if self._num_decoder_layer == 1:
            # (B, T, D): paddle Conv1D uses NCL => channel dim is the time axis
            x = self._decoder1(x)
        else:
            x = x.transpose(1, 2)
            for div_projection in self._div_projection:
                output = torch.zeros_like(x)
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:, :, i * self._div_len: min(i * self._div_len + self._overlap_len, self._in_chunk_len)]
                    output[:, :, i * self._div_len: (i + 1) * self._div_len] = div_layer(div_x)
                x = output
            x = self._decoder1(x).transpose(1, 2)
        if self._num_stack == 1:
            return x, None
        mid_output = x
        if self._concat_len > 0:
            x = torch.cat([res1[:, -self._concat_len:, :], x], dim=1)
        else:
            x = torch.cat([res1, x], dim=1)
        res2 = x
        x = self._encoder2(x)
        x = x + res2
        x = self._decoder2(x)
        return x, mid_output
