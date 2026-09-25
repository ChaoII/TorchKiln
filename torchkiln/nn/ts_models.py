"""Time-series DL models — PyTorch ports of PaddleTS (``paddlets``).

Interface mirrors PaddleTS: input is a dict of tensors
``{"past_target", "known_cov_numeric", "observed_cov_numeric"}`` and the output
is ``(B, out_chunk_len, target_dim)``.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

__all__ = ["NBEATS", "NBEATS_TS", "MLP", "DLinear", "TCN", "NHiTS"]


class _TrendGenerator(nn.Module):
    def __init__(self, expansion_coefficient_dim, target_length):
        super().__init__()
        basis = torch.stack(
            [(torch.arange(target_length) / target_length) ** i
             for i in range(expansion_coefficient_dim)], dim=1).T
        self.register_buffer("_basis", basis)

    def forward(self, x):
        return x @ self._basis


class _SeasonalityGenerator(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        half = int(target_length / 2 - 1)
        cos_v = [torch.cos(torch.arange(target_length) / target_length * 2 * np.pi * i)
                 for i in range(1, half + 1)]
        sin_v = [torch.sin(torch.arange(target_length) / target_length * 2 * np.pi * i)
                 for i in range(1, half + 1)]
        basis = torch.stack([torch.ones(target_length)] + cos_v + sin_v, dim=1).T
        self.register_buffer("_basis", basis)

    def forward(self, x):
        return x @ self._basis


class _Block(nn.Module):
    def __init__(self, num_layers, layer_width, expansion_coefficient_dim,
                 backcast_length, in_chunk_len, target_length, target_dim, g_type):
        super().__init__()
        self._target_dim = target_dim
        self._relu = nn.ReLU()
        stack = [nn.Linear(in_chunk_len, layer_width)]
        stack += [nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self._linear_layer_stack_list = nn.ModuleList(stack)
        if g_type == "seasonality":
            self._backcast_linear_layer = nn.Linear(
                layer_width, (2 * int(backcast_length / 2 - 1) + 1) * target_dim)
            self._forecast_linear_layer = nn.Linear(
                layer_width, (2 * int(target_length / 2 - 1) + 1) * target_dim)
        else:
            self._backcast_linear_layer = nn.Linear(
                layer_width, expansion_coefficient_dim * target_dim)
            self._forecast_linear_layer = nn.Linear(
                layer_width, expansion_coefficient_dim * target_dim)
        if g_type == "generic":
            self._backcast_g = nn.Linear(expansion_coefficient_dim, backcast_length)
            self._forecast_g = nn.Linear(expansion_coefficient_dim, target_length)
        elif g_type == "trend":
            self._backcast_g = _TrendGenerator(expansion_coefficient_dim, backcast_length)
            self._forecast_g = _TrendGenerator(expansion_coefficient_dim, target_length)
        elif g_type == "seasonality":
            self._backcast_g = _SeasonalityGenerator(backcast_length)
            self._forecast_g = _SeasonalityGenerator(target_length)
        else:
            raise ValueError("g_type not supported")

    def forward(self, backcast, known_cov, observed_cov):
        b = backcast.shape[0]
        feat = [backcast.reshape(b, -1)]
        if known_cov is not None:
            feat.append(known_cov.reshape(b, -1))
        if observed_cov is not None:
            feat.append(observed_cov.reshape(b, -1))
        x = torch.cat(feat, dim=1)
        for layer in self._linear_layer_stack_list:
            x = self._relu(layer(x))
        theta_b = self._backcast_linear_layer(x)
        theta_f = self._forecast_linear_layer(x)
        theta_f = theta_f.reshape(b, self._target_dim, -1)
        theta_b = theta_b.reshape(b, self._target_dim, -1)
        x_hat = self._backcast_g(theta_b)
        y_hat = self._forecast_g(theta_f)
        return x_hat.transpose(1, 2), y_hat.transpose(1, 2)


class _Stack(nn.Module):
    def __init__(self, num_blocks, num_layers, layer_width, expansion_coefficient_dim,
                 backcast_length, in_chunk_len, target_length, target_dim, g_type):
        super().__init__()
        self._target_length = target_length
        self._target_dim = target_dim
        if g_type == "generic":
            blocks = [_Block(num_layers, layer_width, expansion_coefficient_dim,
                             backcast_length, in_chunk_len, target_length,
                             target_dim, g_type) for _ in range(num_blocks)]
        else:
            blk = _Block(num_layers, layer_width, expansion_coefficient_dim,
                         backcast_length, in_chunk_len, target_length, target_dim, g_type)
            blocks = [blk] * num_blocks
        self._blocks = nn.ModuleList(blocks)
        self._blocks_list = blocks

    def forward(self, backcast, known_cov, observed_cov):
        stack_forecast = torch.zeros(
            backcast.shape[0], self._target_length, self._target_dim,
            dtype=backcast.dtype, device=backcast.device)
        for block in self._blocks_list:
            x_hat, y_hat = block(backcast, known_cov, observed_cov)
            stack_forecast = stack_forecast + y_hat
            backcast = backcast - x_hat
        return backcast, stack_forecast


class NBEATS(nn.Module):
    """N-BEATS (PaddleTS ``NBEATSModel`` / ``_NBEATSModule``)."""

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1,
                 known_cov_dim=0, observed_cov_dim=0, generic_architecture=True,
                 num_stacks=2, num_blocks=3, num_layers=4, layer_widths=128,
                 expansion_coefficient_dim=128, trend_polynomial_degree=4):
        super().__init__()
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        self._known_cov_dim = known_cov_dim
        self._observed_cov_dim = observed_cov_dim
        input_dim = target_dim + known_cov_dim + observed_cov_dim
        in_chunk_len_multi = in_chunk_len * input_dim + out_chunk_len * known_cov_dim
        self._target_length = out_chunk_len
        if not generic_architecture:
            num_stacks = 2
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * num_stacks
        if isinstance(layer_widths, int):
            layer_widths = [layer_widths] * num_stacks
        if generic_architecture:
            stacks = [
                _Stack(num_blocks[i], num_layers, layer_widths[i],
                       expansion_coefficient_dim, in_chunk_len, in_chunk_len_multi,
                       self._target_length, target_dim, "generic")
                for i in range(num_stacks)]
        else:
            trend = _Stack(num_blocks[0], num_layers, layer_widths[0],
                           trend_polynomial_degree + 1, in_chunk_len, in_chunk_len_multi,
                           self._target_length, target_dim, "trend")
            season = _Stack(num_blocks[1], num_layers, layer_widths[1], -1,
                            in_chunk_len, in_chunk_len_multi, self._target_length,
                            target_dim, "seasonality")
            stacks = [trend, season]
        self._stacks = nn.ModuleList(stacks)
        self._stacks_list = stacks
        last = stacks[-1]._blocks_list[-1]
        for p in list(last._backcast_linear_layer.parameters()) + \
                list(last._backcast_g.parameters()):
            p.requires_grad = False

    def forward(self, data):
        backcast = data["past_target"]
        known_cov = data.get("known_cov_numeric") if self._known_cov_dim > 0 else None
        observed_cov = (data.get("observed_cov_numeric")
                        if self._observed_cov_dim > 0 else None)
        forecast = torch.zeros(backcast.shape[0], self._target_length, self._target_dim,
                               dtype=backcast.dtype, device=backcast.device)
        for stack in self._stacks_list:
            backcast, stack_forecast = stack(backcast, known_cov, observed_cov)
            forecast = forecast + stack_forecast
        return forecast


NBEATS_TS = NBEATS


class MLP(nn.Module):
    """MLP regressor (PaddleTS ``_MLPModule``). Supports DLinear via ``kernel_size``."""

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1,
                 hidden_config=(128, 128), use_bn=False):
        super().__init__()
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        dims = [in_chunk_len] + list(hidden_config)
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if use_bn:
                layers.append(nn.BatchNorm1d(target_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], out_chunk_len))
        self._nn = nn.Sequential(*layers)

    def forward(self, data):
        out = data["past_target"].transpose(1, 2)
        out = self._nn(out)
        return out.transpose(1, 2)


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride,
                                padding=0)
        self.stride = stride

    def forward(self, x):
        # x: (B, L, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return x


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """DLinear (decomposition + per-timestep linear)."""

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1, kernel_size=25):
        super().__init__()
        self._out_chunk_len = out_chunk_len
        self.decomposition = _SeriesDecomp(kernel_size)
        self.linear_seasonal = nn.Linear(in_chunk_len, out_chunk_len)
        self.linear_trend = nn.Linear(in_chunk_len, out_chunk_len)

    def forward(self, data):
        x = data["past_target"]
        seasonal, trend = self.decomposition(x)
        seasonal = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        trend = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)
        return seasonal + trend


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
        super().__init__()
        self._conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
        self._conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation))
        self._downsample = (nn.Conv1d(in_channels, out_channels, 1)
                            if in_channels != out_channels else None)
        self._dropout = nn.Dropout(dropout_rate)
        self._padding = dilation * (kernel_size - 1)

    def forward(self, x):
        residual = self._downsample(x) if self._downsample is not None else x
        out = torch.nn.functional.pad(x, (self._padding, 0))
        out = torch.relu(self._conv1(out))
        out = self._dropout(out)
        out = torch.nn.functional.pad(out, (self._padding, 0))
        out = torch.relu(self._conv2(out))
        out = self._dropout(out)
        return out + residual


class TCN(nn.Module):
    """Temporal Convolution Net (PaddleTS ``_TCNModule``)."""

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1,
                 hidden_config=(128, 128), kernel_size=3, dropout_rate=0.0):
        super().__init__()
        self._out_chunk_len = out_chunk_len
        channels = [target_dim] + list(hidden_config) + [target_dim]
        layers = []
        for k, (ic, oc) in enumerate(zip(channels[:-1], channels[1:])):
            layers.append(_TemporalBlock(ic, oc, kernel_size, 2 ** k, dropout_rate))
        self._temporal_layers = nn.Sequential(*layers)

    def forward(self, data):
        out = data["past_target"].transpose(1, 2)
        out = self._temporal_layers(out)
        out = out.transpose(1, 2)
        return out[:, -self._out_chunk_len:, :]


NHiTS = MLP  # placeholder (replaced below)


class _NHiTSBlock(nn.Module):
    def __init__(self, in_chunk_len, out_chunk_len, target_dim, known_cov_dim,
                 observed_cov_dim, num_layers, layer_width, pooling_kernel_size,
                 n_freq_downsample, batch_norm, dropout, activation, max_pool):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        self._activation = getattr(nn, activation)()
        n_theta_b = max(in_chunk_len // n_freq_downsample * target_dim, 1)
        n_theta_f = max(out_chunk_len // n_freq_downsample * target_dim, 1)
        pool = nn.MaxPool1d if max_pool else nn.AvgPool1d
        self.pooling_layer = pool(kernel_size=pooling_kernel_size,
                                  stride=pooling_kernel_size, ceil_mode=True)
        in_len = (int(np.ceil(in_chunk_len / pooling_kernel_size))
                  * (target_dim + known_cov_dim + observed_cov_dim)
                  + int(np.ceil(out_chunk_len / pooling_kernel_size)) * known_cov_dim)
        widths = [in_len] + [layer_width] * num_layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            layers.append(self._activation)
            if batch_norm:
                layers.append(nn.BatchNorm1d(widths[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)
        self.backcast_linear_layer = nn.Linear(layer_width, n_theta_b)
        self.forecast_linear_layer = nn.Linear(layer_width, n_theta_f)

    def forward(self, backcast, known_cov, observed_cov):
        b = backcast.shape[0]
        past = [backcast]
        future = None
        if known_cov is not None:
            past.append(known_cov[:, :self._in_chunk_len, :])
            future = known_cov[:, self._in_chunk_len:, :].transpose(1, 2)
        if observed_cov is not None:
            past.append(observed_cov)
        past = torch.cat(past, dim=2).transpose(1, 2)
        x = self.pooling_layer(past).reshape(b, -1)
        if future is not None:
            x = torch.cat([x, self.pooling_layer(future).reshape(b, -1)], dim=1)
        x = self.layers(x)
        theta_b = self.backcast_linear_layer(x).reshape(b, self._target_dim, -1)
        theta_f = self.forecast_linear_layer(x).reshape(b, self._target_dim, -1)
        x_hat = torch.nn.functional.interpolate(theta_b, size=self._in_chunk_len, mode="linear")
        y_hat = torch.nn.functional.interpolate(theta_f, size=self._out_chunk_len, mode="linear")
        return x_hat.transpose(1, 2), y_hat.transpose(1, 2)


class _NHiTSStack(nn.Module):
    def __init__(self, in_chunk_len, out_chunk_len, num_blocks, num_layers,
                 layer_width, target_dim, known_cov_dim, observed_cov_dim,
                 pooling_kernel_sizes, n_freq_downsample, batch_norm, dropout,
                 activation, max_pool):
        super().__init__()
        self.out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        self._blocks_list = [
            _NHiTSBlock(in_chunk_len, out_chunk_len, target_dim, known_cov_dim,
                        observed_cov_dim, num_layers, layer_width,
                        pooling_kernel_sizes[i], n_freq_downsample[i],
                        batch_norm=(batch_norm and i == 0), dropout=dropout,
                        activation=activation, max_pool=max_pool)
            for i in range(num_blocks)]
        self._blocks = nn.ModuleList(self._blocks_list)

    def forward(self, backcast, known_cov, observed_cov):
        stack_forecast = torch.zeros(backcast.shape[0], self.out_chunk_len,
                                     self._target_dim, dtype=backcast.dtype,
                                     device=backcast.device)
        for block in self._blocks_list:
            x_hat, y_hat = block(backcast, known_cov, observed_cov)
            stack_forecast = stack_forecast + y_hat
            backcast = backcast - x_hat
        return backcast, stack_forecast


class NHiTS(nn.Module):
    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1, known_cov_dim=0,
                 observed_cov_dim=0, num_stacks=3, num_blocks=1, num_layers=2,
                 layer_widths=512, pooling_kernel_sizes=None,
                 n_freq_downsample=None, batch_norm=False, dropout=0.0,
                 activation="ReLU", MaxPool1d=True):
        super().__init__()
        if isinstance(layer_widths, int):
            layer_widths = [layer_widths] * num_stacks
        self._known_cov_dim = known_cov_dim
        self._observed_cov_dim = observed_cov_dim
        self._target_dim = target_dim
        self._target_length = out_chunk_len

        def auto(in_len, val):
            max_v = max(in_len // 2, 1)
            return tuple((max(int(v), 1),) * num_blocks
                         for v in max_v / np.geomspace(1, max_v, num_stacks))

        if pooling_kernel_sizes is None:
            pooling_kernel_sizes = auto(in_chunk_len, None)
        if n_freq_downsample is None:
            n_freq_downsample = auto(out_chunk_len, None)
        self._stacks_list = [
            _NHiTSStack(in_chunk_len, out_chunk_len, num_blocks, num_layers,
                        layer_widths[i], target_dim, known_cov_dim, observed_cov_dim,
                        pooling_kernel_sizes[i], n_freq_downsample[i],
                        batch_norm=(batch_norm and i == 0), dropout=dropout,
                        activation=activation, max_pool=MaxPool1d)
            for i in range(num_stacks)]
        self._stacks = nn.ModuleList(self._stacks_list)
        for p in self._stacks_list[-1]._blocks_list[-1].backcast_linear_layer.parameters():
            p.requires_grad = False

    def forward(self, data):
        backcast = data["past_target"]
        known_cov = data.get("known_cov_numeric") if self._known_cov_dim > 0 else None
        observed_cov = (data.get("observed_cov_numeric")
                        if self._observed_cov_dim > 0 else None)
        forecast = torch.zeros(backcast.shape[0], self._target_length, self._target_dim,
                               dtype=backcast.dtype, device=backcast.device)
        for stack in self._stacks_list:
            backcast, stack_forecast = stack(backcast, known_cov, observed_cov)
            forecast = forecast + stack_forecast
        return forecast


