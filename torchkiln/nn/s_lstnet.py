"""LSTNet forecaster — PyTorch port of PaddleTS ``_LSTNetModule``.

Input is a dict ``{"past_target"}`` and output is ``(B, out_chunk_len, target_dim)``.
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LSTNet"]


class LSTNet(nn.Module):
    """LSTNet (PaddleTS ``_LSTNetModule``)."""

    _RNN = {"LSTM": nn.LSTM, "GRU": nn.GRU}

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1,
                 skip_size=1, channels=1, kernel_size=3,
                 rnn_cell_type="GRU", rnn_num_cells=10,
                 skip_rnn_cell_type="GRU", skip_rnn_num_cells=10,
                 dropout_rate=0.2, output_activation=None):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._channels = channels
        self._skip_size = skip_size
        self._skip_rnn_num_cells = skip_rnn_num_cells
        self._output_activation = output_activation
        conv_out = in_chunk_len - kernel_size
        self._conv_skip = conv_out // skip_size
        self._cnn = nn.Conv1d(target_dim, channels, kernel_size)
        self._dropout = nn.Dropout(dropout_rate)
        self._rnn = self._RNN[rnn_cell_type](channels, rnn_num_cells, batch_first=True)
        self._skip_rnn = self._RNN[skip_rnn_cell_type](
            channels, skip_rnn_num_cells, batch_first=True)
        self._fc = nn.Linear(rnn_num_cells + skip_size * skip_rnn_num_cells, target_dim)
        self._ar_fc = nn.Linear(in_chunk_len, out_chunk_len)

    def forward(self, data):
        x = data["past_target"]
        cnn_out = self._cnn(x.transpose(1, 2)).transpose(1, 2)  # (B,T,C)
        cnn_out = F.relu(cnn_out)
        cnn_out = self._dropout(cnn_out)

        _, rnn_out = self._rnn(cnn_out)
        rnn_out = rnn_out[0] if isinstance(rnn_out, tuple) else rnn_out
        rnn_out = self._dropout(rnn_out)
        rnn_out = rnn_out.squeeze(0)  # (B, C)

        skip_out = cnn_out[:, -self._conv_skip * self._skip_size:, :]
        skip_out = skip_out.reshape(-1, self._conv_skip, self._skip_size, self._channels)
        skip_out = skip_out.transpose(1, 2)
        skip_out = skip_out.reshape(-1, self._conv_skip, self._channels)
        _, skip_out = self._skip_rnn(skip_out)
        skip_out = skip_out[0] if isinstance(skip_out, tuple) else skip_out
        skip_out = skip_out.reshape(-1, self._skip_size * self._skip_rnn_num_cells)
        skip_out = self._dropout(skip_out)

        res = self._fc(torch.cat([rnn_out, skip_out], dim=1))
        res = res.unsqueeze(1)  # (B, 1, target)

        ar_in = x[:, -self._in_chunk_len:, :].transpose(1, 2)
        ar_out = self._ar_fc(ar_in).transpose(1, 2)
        out = ar_out + res
        if self._output_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self._output_activation == "tanh":
            out = torch.tanh(out)
        return out
