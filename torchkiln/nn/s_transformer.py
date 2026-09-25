"""Transformer forecaster — PyTorch port of PaddleTS ``_TransformerModule``.

Input is a dict ``{"past_target"}`` and output is ``(B, out_chunk_len, target_dim)``.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

__all__ = ["Transformer", "PositionalEncoding"]


class PositionalEncoding(nn.Module):
    """PaddleTS ``_PositionalEncoding``."""

    def __init__(self, d_model, max_len, dropout_rate):
        super().__init__()
        self._dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-1. * np.log2(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("_pe", pe)

    def forward(self, x):
        out = x + self._pe[: x.shape[1], :]
        return self._dropout(out)


class Transformer(nn.Module):
    """Transformer model (PaddleTS ``_TransformerModule``)."""

    COVS = ["observed_cov_numeric", "known_cov_numeric"]

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1, input_dim=None,
                 d_model=8, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=64, activation="relu", dropout_rate=0.1,
                 custom_encoder=None, custom_decoder=None):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        self._input_dim = input_dim if input_dim is not None else target_dim
        self._encoder = nn.Linear(self._input_dim, d_model)
        self._positional_encoding = PositionalEncoding(d_model, in_chunk_len, dropout_rate)
        self._transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout_rate,
            activation=activation,
            custom_encoder=custom_encoder, custom_decoder=custom_decoder,
            batch_first=True)
        self._decoder = nn.Linear(d_model, out_chunk_len * target_dim)

    def _create_transformer_inputs(self, x):
        covs = [x[cov][:, :self._in_chunk_len, :] for cov in self.COVS if cov in x]
        feats = [x["past_target"]] + covs
        src = torch.cat(feats, dim=-1)
        tgt = src[:, -1:, :]
        return src, tgt

    def forward(self, x):
        src, tgt = self._create_transformer_inputs(x)
        src = self._encoder(src) * np.sqrt(self._input_dim)
        src = self._positional_encoding(src)
        tgt = self._encoder(tgt) * np.sqrt(self._input_dim)
        tgt = self._positional_encoding(tgt)
        out = self._transformer(src, tgt)
        out = self._decoder(out)
        out = out[:, 0, :].reshape(-1, self._out_chunk_len, self._target_dim)
        return out
