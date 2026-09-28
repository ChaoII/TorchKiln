"""时序异常检测网络（移植自 paddlets ``models/anomaly/dl``）。

本文件包含「编码-解码」类异常检测模型的核心网络：
  * :class:`AnomalyMLP` / :class:`AnomalyCNN` / :class:`AnomalyLSTM`  —— 对应 paddlets ``_ed/ed.py``
  * :class:`AEBlock`        —— 对应 paddlets ``autoencoder.py::_AEBlock``
  * :class:`VAEStack` / :class:`VAEBlock` —— 对应 paddlets ``vae.py::stack/_VAEBlock``

⚠️ **命名与 Paddle 逐一对齐**（`_nn` / `_encoder` / `_decoder` / `mu` / `logvar` /
`reconstructed` / `_observed_cat_emb`），以便：
  ① 从 Paddle dump 的 ``state_dict`` 直接加载（Linear 转置、BN ``_mean/_variance`` 改名）；
  ② 逐层前向对拍（fp64 判定法）。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn

__all__ = [
    "AnomalyMLP", "AnomalyCNN", "AnomalyLSTM",
    "AEBlock", "VAEStack", "VAEBlock", "USADBlock",
]


def _act(a):
    """paddlets 传的是**类**（如 ``paddle.nn.ReLU``），这里统一成实例。"""
    if a is None:
        return nn.Identity()
    if isinstance(a, type):
        return a()
    return a


class AnomalyMLP(nn.Module):
    """paddlets ``_ed/ed.py::MLP``（Linear + BatchNorm1D 交替）。

    ⚠️ 语义说明（易错）：``input_dim`` 是**最后一维**大小；``feature_dim`` 是 **dim1**
    （Paddle ``BatchNorm1D(feature_dim)`` 按 dim1 做通道）。输入 ``(N, C, L)`` 时
    ``feature_dim=C``；输入 ``(N, L, C)`` 时 ``feature_dim=L``。
    """

    def __init__(self, input_dim, feature_dim, hidden_config,
                 activation=None, last_layer_activation=None,
                 dropout_rate: float = 0.5, use_bn: bool = True,
                 use_drop: bool = True):
        super().__init__()
        dims = [input_dim] + list(hidden_config)
        layers: List[nn.Module] = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if use_bn:
                layers.append(nn.BatchNorm1d(feature_dim))
            if i < len(dims) - 1:
                layers.append(_act(activation))
                if use_drop:
                    layers.append(nn.Dropout(dropout_rate))
            else:
                layers.append(_act(last_layer_activation))
        self._nn = nn.Sequential(*layers)

    def forward(self, x):
        return self._nn(x)


class AnomalyCNN(nn.Module):
    """paddlets ``_ed/ed.py::CNN``（Conv1d / Conv1dTranspose）。

    ``data_format``: ``'NCL'``（AE 用）或 ``'NLC'``（VAE 用）。
    torch 无 data_format，统一在 forward 里做转置以复刻 Paddle 语义：
      * ``NCL``：直接 conv（in_channels = dim1）
      * ``NLC``：Paddle 会把 ``(N,L,C)`` 视为 ``(N,C,L)`` → 先转置到 ``(N,C,L)`` 再 conv。
    """

    def __init__(self, input_dim, hidden_config, activation=None,
                 last_layer_activation=None, kernel_size: int = 3,
                 dropout_rate: float = 0.5, use_bn: bool = True,
                 is_encoder: bool = True, use_drop: bool = True,
                 data_format: str = "NCL"):
        super().__init__()
        self.data_format = data_format
        dims = [input_dim] + list(hidden_config)
        layers: List[nn.Module] = []
        for i in range(1, len(dims)):
            if is_encoder:
                layers.append(nn.Conv1d(dims[i - 1], dims[i], kernel_size))
            else:
                layers.append(nn.ConvTranspose1d(dims[i - 1], dims[i], kernel_size))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i]))
            if i < len(dims) - 1:
                layers.append(_act(activation))
                if use_drop:
                    layers.append(nn.Dropout(dropout_rate))
            else:
                layers.append(_act(last_layer_activation))
        self._nn = nn.Sequential(*layers)

    def forward(self, x):
        if self.data_format == "NLC":
            x = x.transpose(1, 2)
        y = self._nn(x)
        if self.data_format == "NLC":
            y = y.transpose(1, 2)
        return y


class AnomalyLSTM(nn.Module):
    """paddlets ``_ed/ed.py::LSTM``（逐层堆叠 ``nn.LSTM`` + 激活 + dropout）。

    ⚠️ 同 :class:`GRULayer`：Paddle RNN 默认 ``time_major=False``（batch-first）
    ⇒ torch 必须 ``batch_first=True``。
    """

    def __init__(self, input_dim, hidden_config, activation=None,
                 last_layer_activation=None, dropout_rate: float = 0.0,
                 use_drop: bool = True, num_layers: int = 1,
                 direction: str = "forward"):
        super().__init__()
        dims = [input_dim] + list(hidden_config)
        layers: List[nn.Module] = []
        for i in range(1, len(dims)):
            layers.append(nn.LSTM(dims[i - 1], dims[i], num_layers=num_layers,
                                  dropout=dropout_rate, batch_first=True,
                                  bidirectional=(direction == "bidirect")))
            if i < len(dims) - 1:
                layers.append(_act(activation))
                if use_drop:
                    layers.append(nn.Dropout(dropout_rate))
            else:
                layers.append(_act(last_layer_activation))
        self._nn = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self._nn:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        return x


class AEBlock(nn.Module):
    """paddlets ``autoencoder.py::_AEBlock``。

    forward 输入 ``X["observed_cov_numeric"]``（``(N, L, C)``），返回
    ``(recon, x)``，两者均已转置为 ``(N, L, C)`。
    """

    def __init__(self, in_chunk_len: int, ed_type: str,
                 fit_params: Dict[str, Any], hidden_config: List[int],
                 activation: Callable | None, last_layer_activation: Callable | None,
                 kernel_size: int, dropout_rate: float, use_bn: bool,
                 embedding_size: int, pooling: bool):
        super().__init__()
        assert ed_type in ("MLP", "CNN"), "`ae_type` must be 'MLP' or 'CNN'"
        self._pooling = pooling
        self._cat_size = len(fit_params["observed_cat_cols"])
        self._cat_dim = 0
        self._num_dim = fit_params["observed_num_dim"]
        self._observed_cat_emb = nn.ModuleList()
        if fit_params["observed_cat_cols"]:
            for col, col_size in fit_params["observed_cat_cols"].items():
                self._observed_cat_emb.append(nn.Embedding(col_size, embedding_size))
            if pooling:
                self._cat_dim = embedding_size
            else:
                self._cat_dim = embedding_size * len(fit_params["observed_cat_cols"])

        feature_dim = self._num_dim + self._cat_dim
        if ed_type == "MLP":
            self._encoder = AnomalyMLP(in_chunk_len, feature_dim, hidden_config,
                                       activation, last_layer_activation,
                                       dropout_rate, use_bn)
            self._decoder = AnomalyMLP(
                hidden_config[-1], feature_dim,
                list(hidden_config[::-1][1:]) + [in_chunk_len],
                activation, last_layer_activation, dropout_rate, use_bn)
        else:
            enc_in = in_chunk_len
            for _ in range(len(hidden_config)):
                enc_in = enc_in - kernel_size + 1
                if enc_in < 1:
                    raise ValueError(
                        "Conv1d output size must be >= 1; choose smaller "
                        "`kernel_size` or bigger `in_chunk_len`")
            self._encoder = AnomalyCNN(feature_dim, hidden_config,
                                       activation, last_layer_activation,
                                       kernel_size, dropout_rate, use_bn,
                                       is_encoder=True)
            self._decoder = AnomalyCNN(
                hidden_config[-1], list(hidden_config[::-1][1:]) + [feature_dim],
                activation, last_layer_activation, kernel_size, dropout_rate,
                use_bn, is_encoder=False)

    def forward(self, X):
        x = X["observed_cov_numeric"].transpose(1, 2)          # (N, C, L)
        if self._cat_size > 0:
            observed_cat = X["observed_cov_categorical"].transpose(1, 2)  # (N, cat, L)
            feature_cat = []
            for i in range(self._cat_size):
                feature_cat.append(self._observed_cat_emb[i](observed_cat[:, i]))
            if self._pooling:
                feature_cat = torch.stack(feature_cat, dim=-1).mean(dim=-1)
            else:
                feature_cat = torch.cat(feature_cat, dim=-1)
            feature_cat = feature_cat.transpose(1, 2)           # (N, emb, L)
            x = torch.cat([x, feature_cat], dim=-2)
        h = self._encoder(x)
        recon = self._decoder(h)
        return recon.transpose(1, 2), x.transpose(1, 2)


class VAEStack(nn.Module):
    """paddlets ``vae.py::stack``（按 ``base_nn`` 选 MLP/CNN/LSTM）。

    ⚠️ ``is_encoder=False`` 时**先反转** ``hidden_config``。
    """

    def __init__(self, in_chunk_dim: int, hidden_config: List[int],
                 feature_dim: int, is_encoder: bool = True, base_nn: str = "MLP",
                 use_bn: bool = True, use_drop: bool = True,
                 dropout_rate: float = 0.5, kernel_size: int = 1,
                 rnn_num_layers: int = 1, direction: str = "forward",
                 activation: Callable | None = None,
                 last_layer_activation: Callable | None = None):
        super().__init__()
        hidden_config = list(hidden_config)
        if not is_encoder:
            hidden_config = [int(i) for i in reversed(hidden_config)]
        if base_nn == "MLP":
            self._nn = AnomalyMLP(feature_dim, in_chunk_dim, hidden_config,
                                  activation, last_layer_activation,
                                  dropout_rate, use_bn, use_drop)
        elif base_nn == "LSTM":
            self._nn = AnomalyLSTM(feature_dim, hidden_config,
                                   activation, last_layer_activation,
                                   dropout_rate, use_drop,
                                   num_layers=rnn_num_layers, direction=direction)
        elif base_nn == "CNN":
            self._nn = AnomalyCNN(feature_dim, hidden_config,
                                  activation, last_layer_activation,
                                  kernel_size, dropout_rate, use_bn,
                                  is_encoder=True, use_drop=use_drop,
                                  data_format="NLC")
        else:
            raise ValueError("base_nn must be in ('MLP', 'CNN', 'LSTM')")

    def forward(self, x):
        return self._nn(x)


class VAEBlock(nn.Module):
    """paddlets ``vae.py::_VAEBlock``。forward 返回 ``[recon, mu, logvar, x]``。"""

    def __init__(self, in_chunk_dim: int, hidden_config: List[int],
                 feature_dim: int, base_en: str = "MLP", base_de: str = "MLP",
                 use_bn: bool = True, use_drop: bool = True,
                 dropout_rate: float = 0.5, kernel_size: int = 1,
                 rnn_num_layers: int = 1, direction: str = "forward",
                 activation: Callable | None = None,
                 last_layer_activation: Callable | None = None,
                 stdev: float = 0.1):
        super().__init__()
        assert base_en in ("MLP", "CNN", "LSTM"), "base_en must be in ('MLP','CNN','LSTM')"
        assert base_de in ("MLP", "CNN", "LSTM"), "base_de must be in ('MLP','CNN','LSTM')"
        self.stdev = stdev
        self.de_hidden_config = [int(i) for i in reversed(hidden_config)]
        # ⚠️ 对齐 Paddle：activation/last_layer_activation 固定为 ReLU6
        self.encoder = VAEStack(in_chunk_dim, hidden_config, feature_dim,
                                is_encoder=True, base_nn=base_en, use_bn=use_bn,
                                use_drop=use_drop, dropout_rate=dropout_rate,
                                kernel_size=kernel_size,
                                rnn_num_layers=rnn_num_layers,
                                direction=direction,
                                activation=nn.ReLU6, last_layer_activation=nn.ReLU6)
        self.decoder = VAEStack(in_chunk_dim, hidden_config, feature_dim,
                                is_encoder=False, base_nn=base_de, use_bn=use_bn,
                                use_drop=use_drop, dropout_rate=dropout_rate,
                                kernel_size=kernel_size,
                                rnn_num_layers=rnn_num_layers,
                                direction=direction,
                                activation=nn.ReLU6, last_layer_activation=nn.ReLU6)
        self.mu = nn.Linear(hidden_config[-1], feature_dim)
        self.logvar = nn.Linear(hidden_config[-1], feature_dim)
        self.reconstructed = nn.Linear(hidden_config[0], feature_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape, device=std.device, dtype=std.dtype) * self.stdev
            return eps * std + mu
        return mu

    def forward(self, X):
        x = X["observed_cov_numeric"]
        h = self.encoder(x)
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        z = self.reparameterize(mu_, logvar_)
        z = self.decoder(z)
        recon = self.reconstructed(z)
        return [recon, mu_, logvar_, x]


class USADBlock(nn.Module):
    """paddlets ``usad.py::_USAModule``（对抗式双解码器自编码器）。

    forward 返回 ``(x, w1, w2, w3)``：
      ``w1 = dec1(enc(x))``、``w2 = dec2(enc(x))``、``w3 = dec2(enc(w1))``

    ⚠️ ``flatten``（且 ``ed_type='MLP'``）时把 ``(N, C, L)`` 拉平成 ``(N, C*L)``，
    并**强制 encoder 不用 BN**（对齐 Paddle）。
    """

    def __init__(self, in_chunk_len: int, ed_type: str,
                 fit_params: Dict[str, Any], hidden_config: List[int],
                 activation: Callable | None, last_layer_activation: Callable | None,
                 kernel_size: int, dropout_rate: float, use_bn: bool,
                 embedding_size: int, pooling: bool, flatten: bool):
        super().__init__()
        assert ed_type in ("MLP", "CNN"), "`ed_type` must be 'MLP' or 'CNN'"
        self._pooling = pooling
        self._cat_size = len(fit_params["observed_cat_cols"])
        self._cat_dim = 0
        self._num_dim = fit_params["observed_num_dim"]
        self._in_chunk_len = in_chunk_len
        self._flatten = flatten if ed_type == "MLP" else False
        self._observed_cat_emb = nn.ModuleList()
        if fit_params["observed_cat_cols"]:
            for _, col_size in fit_params["observed_cat_cols"].items():
                self._observed_cat_emb.append(nn.Embedding(col_size, embedding_size))
            if pooling:
                self._cat_dim = embedding_size
            else:
                self._cat_dim = embedding_size * len(fit_params["observed_cat_cols"])

        feature_dim = self._num_dim + self._cat_dim
        self._feature_dim = feature_dim
        if ed_type == "MLP":
            if self._flatten:
                in_chunk_len = in_chunk_len * feature_dim
                use_bn = False
            self._encoder = AnomalyMLP(in_chunk_len, feature_dim, hidden_config,
                                       activation, activation, dropout_rate, use_bn)
            self._decoder1 = AnomalyMLP(
                hidden_config[-1], feature_dim,
                list(hidden_config[::-1][1:]) + [in_chunk_len],
                activation, last_layer_activation, dropout_rate, use_bn)
            self._decoder2 = AnomalyMLP(
                hidden_config[-1], feature_dim,
                list(hidden_config[::-1][1:]) + [in_chunk_len],
                activation, last_layer_activation, dropout_rate, use_bn)
        else:
            for _ in range(len(hidden_config)):
                in_chunk_len = in_chunk_len - kernel_size + 1
                if in_chunk_len < 1:
                    raise ValueError("Conv1d output size must be >= 1")
            self._encoder = AnomalyCNN(feature_dim, hidden_config,
                                       activation, last_layer_activation,
                                       kernel_size, dropout_rate, use_bn, is_encoder=True)
            self._decoder1 = AnomalyCNN(
                hidden_config[-1], list(hidden_config[::-1][1:]) + [feature_dim],
                activation, last_layer_activation, kernel_size, dropout_rate,
                use_bn, is_encoder=False)
            self._decoder2 = AnomalyCNN(
                hidden_config[-1], list(hidden_config[::-1][1:]) + [feature_dim],
                activation, last_layer_activation, kernel_size, dropout_rate,
                use_bn, is_encoder=False)

    def forward(self, X):
        x = X["observed_cov_numeric"].transpose(1, 2)
        if self._cat_size > 0:
            observed_cat = X["observed_cov_categorical"].transpose(1, 2)
            feature_cat = []
            for i in range(self._cat_size):
                feature_cat.append(self._observed_cat_emb[i](observed_cat[:, i]))
            if self._pooling:
                feature_cat = torch.stack(feature_cat, dim=-1).mean(dim=-1)
            else:
                feature_cat = torch.cat(feature_cat, dim=-1)
            feature_cat = feature_cat.transpose(1, 2)
            x = torch.cat([x, feature_cat], dim=-2)

        if self._flatten:
            x = x.reshape(x.shape[0], -1)

        z = self._encoder(x)
        w1 = self._decoder1(z)
        w2 = self._decoder2(z)
        w3 = self._decoder2(self._encoder(w1))
        return x, w1, w2, w3
