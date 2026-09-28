"""MTAD-GAT 网络（移植自 paddlets ``models/anomaly/dl/_mtad_gat``）。

对应关系：
  * :class:`ConvLayer`            —— ``layer.py::ConvLayer``
  * :class:`GRULayer`             —— ``layer.py::GRULayer``
  * :class:`Reconstruction`       —— ``model.py::Reconstruction``
  * :class:`Forecasting`          —— ``model.py::Forecasting``
  * :class:`FeatOrTempAttention`  —— ``attention.py::FeatOrTempAttention``
  * :class:`MTADGATBlock`         —— ``mtad_gat.py::_MTADGATBlock``

⚠️ 命名与 Paddle 逐一对齐（`_pad/_conv/_relu`、`_gru`、`_layers`、`_lin/_att/_bias`、
`_conv/_feature_gat/_temporal_gat/_gru/_forec_model/_recon_model`），便于权重对拍。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["ConvLayer", "GRULayer", "Reconstruction", "Forecasting",
           "FeatOrTempAttention", "MTADGATBlock"]


class ConvLayer(nn.Module):
    """paddlets ``_mtad_gat/layer.py::ConvLayer``（零填充 + Conv1d + ReLU，NCL 内部）。"""

    def __init__(self, feature_dim: int, kernel_size: int = 7):
        super().__init__()
        self._pad = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self._conv = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                               kernel_size=kernel_size)
        self._relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self._pad(x)
        x = self._relu(self._conv(x))
        return x.transpose(1, 2)


class GRULayer(nn.Module):
    """paddlets ``_mtad_gat/layer.py::GRULayer``（``num_layers==1`` 时 dropout 强制 0）。

    ⚠️ Paddle ``nn.GRU`` 默认 ``time_major=False``（= **batch-first**），
    而 torch 默认 seq-first ⇒ 必须 ``batch_first=True``（否则语义完全不同）。
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float):
        super().__init__()
        self._dropout = 0.0 if num_layers == 1 else dropout
        self._gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                           dropout=self._dropout, batch_first=True)

    def forward(self, x):
        out, h = self._gru(x)
        return out, h


class Reconstruction(nn.Module):
    """paddlets ``_mtad_gat/model.py::Reconstruction``（把末隐状态展成序列再解）。"""

    def __init__(self, in_chunk_len: int, feature_dim: int, hidden_size: int,
                 out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._decoder = GRULayer(feature_dim, hidden_size, num_layers, dropout)
        self._fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        h_end = torch.repeat_interleave(x, repeats=self._in_chunk_len, dim=1)
        h_end = h_end.reshape((x.shape[0], self._in_chunk_len, -1))
        decoder_out, _ = self._decoder(h_end)
        return self._fc(decoder_out)


class Forecasting(nn.Module):
    """paddlets ``_mtad_gat/model.py::Forecasting``（FC 堆叠，末尾不加激活）。"""

    def __init__(self, feature_dim: int, hidden_size: int, out_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        layers = [nn.Linear(feature_dim, hidden_size)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, out_dim))
        self._layers = nn.ModuleList(layers)
        self._dropout = nn.Dropout(dropout)
        self._relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self._layers) - 1):
            x = self._relu(self._layers[i](x))
            x = self._dropout(x)
        return self._layers[-1](x)


class FeatOrTempAttention(nn.Module):
    """paddlets ``_mtad_gat/attention.py::FeatOrTempAttention``（GAT / GATv2）。

    ⚠️ ``name='temporal'`` 时 **先交换** ``feature_dim``/``in_chunk_len`` 再赋给
    ``_nodes_num`` / ``_in_chunk_len``（Paddle 原样，勿"顺手改对"）。
    """

    def __init__(self, feature_dim: int, in_chunk_len: int, dropout: float,
                 alpha: float, embed_dim: Optional[int] = None,
                 use_gatv2: bool = True, use_bias: bool = True,
                 name: str = "feature"):
        super().__init__()
        if name == "temporal":
            feature_dim, in_chunk_len = in_chunk_len, feature_dim
        self._feature_dim = feature_dim
        self._in_chunk_len = in_chunk_len
        self._dropout = dropout
        self._alpha = alpha
        self._embed_dim = embed_dim if embed_dim is not None else in_chunk_len
        self._use_gatv2 = use_gatv2
        self._use_bias = use_bias
        self._name = name
        self._nodes_num = feature_dim

        if self._use_gatv2:
            self._embed_dim *= 2
            lin_input_dim = 2 * in_chunk_len
            att_input_dim = self._embed_dim
        else:
            lin_input_dim = in_chunk_len
            att_input_dim = 2 * self._embed_dim

        self._lin = nn.Linear(lin_input_dim, self._embed_dim)
        self._att = nn.Parameter(torch.empty(att_input_dim, 1))
        nn.init.xavier_uniform_(self._att)
        if self._use_bias:
            # ⚠️ Paddle 用 `Assign(paddle.empty(...))`（**未初始化内存**）；
            #    这里用 zeros 占位，对拍时以「Paddle dump 的值」为准。
            self._bias = nn.Parameter(torch.zeros(feature_dim, feature_dim))

        self._leakyrelu = nn.LeakyReLU(alpha)
        self._sigmoid = nn.Sigmoid()

    def _prepare_attention_input(self, v):
        K = self._nodes_num
        blocks_repeating = torch.repeat_interleave(v, repeats=K, dim=1)
        blocks_alternating = v.repeat((1, K, 1))
        combined = torch.cat([blocks_repeating, blocks_alternating], dim=2)
        if self._use_gatv2:
            return combined.reshape(combined.shape[0], K, K, 2 * self._in_chunk_len)
        return combined.reshape(combined.shape[0], K, K, 2 * self._embed_dim)

    def forward(self, x):
        if self._name == "feature":
            x = x.transpose(1, 2)
        if self._use_gatv2:
            att_input = self._prepare_attention_input(x)
            att_input = self._leakyrelu(self._lin(att_input))
            e = torch.matmul(att_input, self._att).squeeze(3)
        else:
            wx = self._lin(x)
            att_input = self._prepare_attention_input(wx)
            e = self._leakyrelu(torch.matmul(att_input, self._att)).squeeze(3)
        if self._use_bias:
            e = e + self._bias
        attention = torch.softmax(e, dim=2)
        attention = nn.functional.dropout(attention, p=self._dropout,
                                          training=self.training)
        h = self._sigmoid(torch.matmul(attention, x))
        if self._name == "feature":
            return h.transpose(1, 2)
        return h


class MTADGATBlock(nn.Module):
    """paddlets ``mtad_gat.py::_MTADGATBlock``。

    forward 返回 ``(preds, recons)``（形状 ``(N, 1, C)`` / ``(N, L-1, C)``）。
    """

    def __init__(self, in_chunk_len: int, fit_params: Dict[str, Any],
                 target_dims: Optional[List[int]], kernel_size: int,
                 feat_gat_embed_dim: Optional[int],
                 time_gat_embed_dim: Optional[int], use_gatv2: bool,
                 use_bias: bool, gru_n_layers: int, gru_hid_size: int,
                 forecast_n_layers: int, forecast_hid_size: int,
                 recon_n_layers: int, recon_hid_size: int, dropout: float,
                 alpha: float):
        super().__init__()
        assert in_chunk_len >= 2, "in_chunk_len must be >= 2 for mtad_gat"
        self._in_chunk_len = in_chunk_len
        self._num_dim = fit_params["observed_num_dim"]
        self._out_dim = self._num_dim
        if target_dims is not None:
            self._out_dim = len(target_dims)

        self._conv = ConvLayer(self._num_dim, kernel_size)
        self._feature_gat = FeatOrTempAttention(
            self._num_dim, in_chunk_len - 1, dropout, alpha,
            feat_gat_embed_dim, use_gatv2, use_bias, "feature")
        self._temporal_gat = FeatOrTempAttention(
            self._num_dim, in_chunk_len - 1, dropout, alpha,
            time_gat_embed_dim, use_gatv2, use_bias, "temporal")
        self._gru = GRULayer(3 * self._num_dim, gru_hid_size, gru_n_layers, dropout)
        self._forec_model = Forecasting(gru_hid_size, forecast_hid_size,
                                        self._out_dim, forecast_n_layers, dropout)
        self._recon_model = Reconstruction(in_chunk_len - 1, gru_hid_size,
                                           recon_hid_size, self._out_dim,
                                           recon_n_layers, dropout)

    def forward(self, X):
        x = X["observed_cov_numeric"][:, :self._in_chunk_len - 1, :]
        x = self._conv(x)
        h_feat = self._feature_gat(x)
        h_temp = self._temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)
        _, h_end = self._gru(h_cat)
        h_end = h_end[-1, :, :]
        h_end = h_end.reshape((x.shape[0], -1))
        preds = self._forec_model(h_end)
        recons = self._recon_model(h_end)
        return preds, recons
