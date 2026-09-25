"""RNN block forecaster — PyTorch port of PaddleTS ``_RNNBlock``.

Input is a dict ``{"past_target"}`` and output is ``(B, out_chunk_len, target_dim)``.
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

__all__ = ["RNNBlock"]


class RNNBlock(nn.Module):
    """RNN block regressor (PaddleTS ``_RNNBlock``)."""

    _RNN = {"SimpleRNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1,
                 rnn_type="LSTM", hidden_dim=128, num_layers_recurrent=1,
                 out_fcn_config=None, dropout=0.0,
                 known_cov_dim=0, observed_cov_dim=0, static_cov_dim=0):
        super().__init__()
        self.in_chunk_len = in_chunk_len
        self.out_chunk_len = out_chunk_len
        self._rnn_type = rnn_type
        self._target_dim = target_dim
        self._num_size = target_dim + known_cov_dim + observed_cov_dim + static_cov_dim
        self._input_size = self._num_size
        self._rnn = self._RNN[rnn_type](
            self._input_size, hidden_dim, num_layers_recurrent, dropout=dropout,
            batch_first=True)
        last = hidden_dim
        feats = []
        for feature in list(out_fcn_config or []) + [out_chunk_len * target_dim]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)

    def forward(self, data):
        feature = data["past_target"]  # (B, T, C)
        out, hidden = self._rnn(feature)
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]
        predictions = hidden[-1, :, :]
        predictions = self.fc(predictions)
        predictions = predictions.reshape(
            predictions.shape[0], self.out_chunk_len, self._target_dim)
        return predictions
