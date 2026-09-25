"""DeepAR forecaster — PyTorch port of PaddleTS ``_DeepAR``.

Probabilistic forecasting with an RNN.  Training/validation path outputs
distribution params (``GaussianLikelihood``); prediction path decodes
regressively by sampling / mean.
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

__all__ = ["DeepAR", "GaussianLikelihood"]


class GaussianLikelihood(object):
    """PaddleTS ``GaussianLikelihood`` (2-arg Gaussian)."""

    num_params = 2

    def __init__(self):
        self.rescale = nn.Softplus()

    def output_to_params(self, model_output):
        mu = model_output[..., 0].unsqueeze(-1)
        sigma = self.rescale(model_output[..., 1]).unsqueeze(-1)
        return torch.cat([mu, sigma], dim=-1)

    def params_to_distr(self, distr_params):
        mu, sigma = distr_params[..., 0], distr_params[..., 1]
        return torch.distributions.Normal(mu, sigma)

    def get_mean(self, distr_params):
        return distr_params[..., 0]

    def sample(self, model_output, num_samples=1):
        distr_params = self.output_to_params(model_output)
        distr = self.params_to_distr(distr_params)
        return distr.sample((num_samples,))

    def loss(self, model_output, target):
        distr_params = self.output_to_params(model_output)
        distr = self.params_to_distr(distr_params)
        losses = -distr.log_prob(target)
        return losses.mean()


class DeepAR(nn.Module):
    """DeepAR (PaddleTS ``_DeepAR``)."""

    OUTPUT_QUANTILE_NUM = 101

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1, known_cov_dim=0,
                 rnn_type="LSTM", hidden_dim=128, num_layers_recurrent=1,
                 drop_out=0.0, likelihood_model=None, num_samples=10,
                 regression_mode="mean", output_mode="quantiles"):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._input_size = target_dim + known_cov_dim
        self._target_dim = target_dim
        self._rnn = getattr(nn, rnn_type)(self._input_size, hidden_dim,
                                          num_layers_recurrent, dropout=drop_out,
                                          batch_first=True)
        self._likelihood_model = likelihood_model or GaussianLikelihood()
        self._num_samples = num_samples
        self.output_projector = nn.Linear(hidden_dim,
                                          self._likelihood_model.num_params * target_dim)
        self._regression_mode = regression_mode
        self._output_mode = output_mode
        self.predicting = False

    def _build_input(self, target, known_cov, first_target_replace=None):
        target_roll = torch.roll(target, shifts=1, dims=1)
        if first_target_replace is not None:
            target_roll[:, 0] = first_target_replace
        else:
            target_roll = target_roll[:, 1:]
            if known_cov is not None:
                known_cov = known_cov[:, 1:]
        if known_cov is not None:
            return torch.cat([target_roll, known_cov], dim=-1)
        return target_roll

    def _decode_direct(self, input_tensor, hidden_state):
        decoder_output, hidden_state = self._rnn(input_tensor, hidden_state)
        output = self.output_projector(decoder_output)
        output = output.reshape(output.shape[0], output.shape[1], self._target_dim, -1)
        return output, hidden_state

    def _encoder(self, data):
        past_target = data["past_target"]
        past_known_cov = data.get("known_cov_numeric")
        if past_known_cov is not None:
            past_known_cov = past_known_cov[:, :past_target.shape[1]]
        input_tensor = self._build_input(past_target, past_known_cov)
        _, hidden_state = self._rnn(input_tensor)
        return hidden_state

    def _decoder(self, data, hidden_state):
        future_target = data["future_target"]
        future_known_cov = data.get("known_cov_numeric")
        if future_known_cov is not None:
            future_known_cov = future_known_cov[:, -future_target.shape[1]:, :]
        first_target_replace = data["past_target"][:, -1]
        input_tensor = self._build_input(future_target, future_known_cov, first_target_replace)
        if not self.predicting:
            model_output, _ = self._decode_direct(input_tensor, hidden_state)
            if self.training:
                output = self._likelihood_model.output_to_params(model_output)
            else:
                output = self._likelihood_model.sample(model_output, self._num_samples)
                output = output.permute(1, 2, 3, 0)
            return output
        # prediction phase (mean regression)
        return self._decode_regressive_by_mean(input_tensor, hidden_state)

    def _decode_regressive_by_mean(self, input_tensor, hidden_state):
        num_steps = input_tensor.shape[1]
        quantiles_output = []
        prediction_output = []
        last_pred_target = input_tensor[:, 0, :self._target_dim]
        for step in range(num_steps):
            x = input_tensor[:, step].unsqueeze(1)
            x[:, 0, :self._target_dim] = last_pred_target
            current_target_params, hidden_state = self._decode_direct(x, hidden_state)
            distr_params = self._likelihood_model.output_to_params(current_target_params)
            distr = self._likelihood_model.params_to_distr(distr_params)
            current_pred_sample = distr.sample((self._num_samples,))
            quantiles_output.append(current_pred_sample.squeeze(2))
            current_target_mean = self._likelihood_model.get_mean(distr_params)
            last_pred_target = current_target_mean.squeeze(1)
            prediction_output.append(last_pred_target)
        quantiles_output = torch.stack(quantiles_output, 2)
        quantiles_output = torch.quantile(
            quantiles_output, torch.linspace(0, 1, self.OUTPUT_QUANTILE_NUM).to(quantiles_output.device),
            dim=0).permute(1, 2, 3, 0)
        prediction_output = torch.stack(prediction_output, 1)
        return quantiles_output, prediction_output

    def forward(self, x):
        hidden_state = self._encoder(x)
        if not self.training:
            x["future_target"] = torch.rand(
                [x["past_target"].shape[0], self._out_chunk_len, self._target_dim],
                device=x["past_target"].device)
        output = self._decoder(x, hidden_state)
        return output
