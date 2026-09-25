"""TemporalFusionTransformer (TFT) — PyTorch port of PaddleTS ``_tft``.

Input is a dict with covariates (past_target / known_cov_numeric /
observed_cov_numeric / static_cov_numeric + categorical variants), output is
``(B, out_chunk_len, target_dim, num_quantiles)``.

Module/param naming matches PaddleTS exactly so paddle state_dict weights can
be loaded and the forward compared.
"""
from __future__ import absolute_import

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TemporalFusionTransformer", "TimeDistributed", "NullTransform",
    "GatedLinearUnit", "GatedResidualNetwork", "GateAddNorm",
    "VariableSelectionNetwork", "InputChannelEmbedding",
    "NumericInputTransformation", "CategoricalInputTransformation",
    "InterpretableMultiHeadAttention",
]


class TimeDistributed(nn.Module):
    """Wrap a module and stack the time dim with the batch dim before applying it."""

    def __init__(self, module, batch_first=True, return_reshaped=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first
        self.return_reshaped = return_reshaped

    def forward(self, x):
        if x.dim() <= 2:
            return self.module(x)
        x_reshape = x.reshape(-1, x.shape[-1])
        y = self.module(x_reshape)
        if self.return_reshaped:
            if self.batch_first:
                y = y.reshape(x.shape[0], -1, y.shape[-1])
            else:
                y = y.reshape(-1, x.shape[1], y.shape[-1])
        return y


class NullTransform(nn.Module):
    @staticmethod
    def forward(empty_input):
        return []


class GatedLinearUnit(nn.Module):
    """GLU: output = sigmoid(fc1(x)) * fc2(x)."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return sig * x


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.05,
                 context_dim=None, batch_first=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.project_residual = input_dim != output_dim
        if self.project_residual:
            self.skip_layer = TimeDistributed(nn.Linear(input_dim, output_dim))
        self.fc1 = TimeDistributed(nn.Linear(input_dim, hidden_dim), batch_first=batch_first)
        if context_dim is not None:
            self.context_projection = TimeDistributed(
                nn.Linear(context_dim, hidden_dim, bias=False), batch_first=batch_first)
        self.elu1 = nn.ELU()
        self.fc2 = TimeDistributed(nn.Linear(hidden_dim, output_dim), batch_first=batch_first)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.gate = TimeDistributed(GatedLinearUnit(output_dim), batch_first=batch_first)
        self.layernorm = TimeDistributed(nn.LayerNorm(output_dim), batch_first=batch_first)

    def forward(self, x, context=None):
        if self.project_residual:
            residual = self.skip_layer(x)
        else:
            residual = x
        x = self.fc1(x)
        if context is not None:
            context = self.context_projection(context)
            x = x + context
        x = self.elu1(x)
        x = self.fc2(x)
        x = self.dropout_layer(x)
        x = self.gate(x)
        x = x + residual
        x = self.layernorm(x)
        return x


class GateAddNorm(nn.Module):
    def __init__(self, input_dim, dropout=None):
        super().__init__()
        self.dropout_rate = dropout
        if dropout:
            self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.gate = TimeDistributed(GatedLinearUnit(input_dim), batch_first=True)
        self.layernorm = TimeDistributed(nn.LayerNorm(input_dim), batch_first=True)

    def forward(self, x, residual=None):
        if self.dropout_rate:
            x = self.dropout_layer(x)
        x = self.gate(x)
        if residual is not None:
            x = x + residual
        x = self.layernorm(x)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, num_inputs, hidden_dim, dropout,
                 context_dim=None, batch_first=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context_dim = context_dim
        self.flattened_grn = GatedResidualNetwork(
            input_dim=num_inputs * input_dim, hidden_dim=hidden_dim,
            output_dim=num_inputs, dropout=dropout, context_dim=context_dim,
            batch_first=batch_first)
        self.softmax = nn.Softmax(dim=1)
        self.single_variable_grns = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(input_dim=input_dim, hidden_dim=hidden_dim,
                                     output_dim=hidden_dim, dropout=dropout,
                                     batch_first=batch_first))

    def forward(self, flattened_embedding, context=None):
        sparse_weights = self.flattened_grn(flattened_embedding, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        processed_inputs = []
        for i in range(self.num_inputs):
            processed_inputs.append(
                self.single_variable_grns[i](
                    flattened_embedding[..., (i * self.input_dim): (i + 1) * self.input_dim]))
        processed_inputs = torch.stack(processed_inputs, dim=-1)
        outputs = processed_inputs * sparse_weights.transpose(1, 2)
        outputs = outputs.sum(dim=-1)
        return outputs, sparse_weights


class NumericInputTransformation(nn.Module):
    def __init__(self, num_inputs, state_size):
        super().__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size
        self.numeric_projection_layers = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.numeric_projection_layers.append(nn.Linear(1, self.state_size))

    def forward(self, x):
        projections = []
        for i in range(self.num_inputs):
            projections.append(self.numeric_projection_layers[i](x[:, i].unsqueeze(1)))
        return projections


class CategoricalInputTransformation(nn.Module):
    def __init__(self, num_inputs, state_size, cardinalities):
        super().__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size
        self.cardinalities = cardinalities
        self.categorical_embedding_layers = nn.ModuleList()
        for cardinality in self.cardinalities:
            self.categorical_embedding_layers.append(nn.Embedding(cardinality, self.state_size))

    def forward(self, x):
        embeddings = []
        for i in range(self.num_inputs):
            embeddings.append(self.categorical_embedding_layers[i](x[:, i]))
        return embeddings


class InputChannelEmbedding(nn.Module):
    def __init__(self, state_size, num_numeric, num_categorical,
                 categorical_cardinalities, time_distribute=False):
        super().__init__()
        self.state_size = state_size
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities
        self.time_distribute = time_distribute
        if num_numeric == 0:
            self.numeric_transform = NullTransform()
        elif self.time_distribute:
            self.numeric_transform = TimeDistributed(
                NumericInputTransformation(num_numeric, state_size), return_reshaped=False)
        else:
            self.numeric_transform = NumericInputTransformation(num_numeric, state_size)
        if num_categorical == 0:
            self.categorical_transform = NullTransform()
        elif self.time_distribute:
            self.categorical_transform = TimeDistributed(
                CategoricalInputTransformation(num_categorical, state_size,
                                               categorical_cardinalities),
                return_reshaped=False)
        else:
            self.categorical_transform = CategoricalInputTransformation(
                num_categorical, state_size, categorical_cardinalities)

    def forward(self, x_numeric, x_categorical):
        if x_numeric.numel() > 0:
            batch_shape = x_numeric.shape
        else:
            batch_shape = x_categorical.shape
        processed_numeric = self.numeric_transform(x_numeric)
        processed_categorical = self.categorical_transform(x_categorical)
        if not processed_numeric + processed_categorical:
            return None
        merged = torch.cat(processed_numeric + processed_categorical, dim=1)
        if self.time_distribute:
            merged = merged.reshape(batch_shape[0], batch_shape[1], -1)
        return merged


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.d_model = embed_dim
        self.num_heads = num_heads
        self.all_heads_dim = embed_dim * num_heads
        self.w_q = nn.Linear(embed_dim, self.all_heads_dim)
        self.w_k = nn.Linear(embed_dim, self.all_heads_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(self.d_model, self.d_model)

    def attention(self, q, k, v, mask=None):
        attention_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_model)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e9)
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_outputs = torch.matmul(attention_scores, v)
        return attention_outputs, attention_scores

    def forward(self, q, k, v, mask=None):
        num_samples = q.shape[0]
        q_proj = self.w_q(q).reshape(num_samples, -1, self.num_heads, self.d_model)
        k_proj = self.w_k(k).reshape(num_samples, -1, self.num_heads, self.d_model)
        v_proj = self.w_v(v).repeat(1, 1, self.num_heads).reshape(
            num_samples, -1, self.num_heads, self.d_model)
        q_proj = q_proj.permute(0, 2, 1, 3)
        k_proj = k_proj.permute(0, 2, 1, 3)
        v_proj = v_proj.permute(0, 2, 1, 3)
        attn_outputs_all, attn_scores_all = self.attention(q_proj, k_proj, v_proj, mask)
        attention_scores = attn_scores_all.mean(dim=1)
        attention_outputs = attn_outputs_all.mean(dim=1)
        output = self.out(attention_outputs)
        return output, attention_outputs, attention_scores


class TemporalFusionTransformer(nn.Module):
    def __init__(self, in_chunk_len, out_chunk_len, fit_params,
                 hidden_dim=64, lstm_layers_num=1, attention_heads_num=2,
                 output_quantiles=(0.1, 0.5, 0.9), dropout=0.0):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._target_dim = fit_params["target_dim"]
        self._known_num_dim = fit_params["known_num_dim"]
        self._known_cat_dim = fit_params["known_cat_dim"]
        self._observed_num_dim = fit_params["observed_num_dim"]
        self._observed_cat_dim = fit_params["observed_cat_dim"]
        self._known_cat_size = fit_params["known_cat_size"]
        self._observed_cat_size = fit_params["observed_cat_size"]
        self._num_historical_numeric = (
            self._target_dim + self._known_num_dim + self._observed_num_dim)
        self._num_historical_categorical = self._known_cat_dim + self._observed_cat_dim
        self._historical_categorical_cardinalities = (
            self._known_cat_size + self._observed_cat_size)
        self._num_future_numeric = self._known_num_dim
        self._num_future_categorical = self._known_cat_dim
        self._future_categorical_cardinalities = self._known_cat_size
        self._static_num_dim = fit_params["static_num_dim"]
        self._static_cat_dim = fit_params["static_cat_dim"]
        self._static_cat_size = fit_params["static_cat_size"]
        self._interpretable_output = False

        self._output_quantiles = output_quantiles
        self._num_outputs = len(self._output_quantiles) * self._target_dim
        self._attention_heads_num = attention_heads_num
        self._lstm_layers = lstm_layers_num
        self._state_size = hidden_dim
        self._dropout = dropout

        self._static_transform = InputChannelEmbedding(
            state_size=hidden_dim, num_numeric=self._static_num_dim,
            num_categorical=self._static_cat_dim,
            categorical_cardinalities=self._static_cat_size, time_distribute=False)
        self._historical_ts_transform = InputChannelEmbedding(
            state_size=hidden_dim, num_numeric=self._num_historical_numeric,
            num_categorical=self._num_historical_categorical,
            categorical_cardinalities=self._historical_categorical_cardinalities,
            time_distribute=True)
        self._future_ts_transform = InputChannelEmbedding(
            state_size=hidden_dim, num_numeric=self._num_future_numeric,
            num_categorical=self._num_future_categorical,
            categorical_cardinalities=self._future_categorical_cardinalities,
            time_distribute=True)

        self._static_selection = VariableSelectionNetwork(
            input_dim=hidden_dim, num_inputs=self._static_num_dim + self._static_cat_dim,
            hidden_dim=hidden_dim, dropout=dropout) \
            if self._static_num_dim + self._static_cat_dim else None
        self._historical_ts_selection = VariableSelectionNetwork(
            input_dim=hidden_dim,
            num_inputs=self._num_historical_numeric + self._num_historical_categorical,
            hidden_dim=hidden_dim, dropout=dropout, context_dim=hidden_dim)
        self._future_ts_selection = VariableSelectionNetwork(
            input_dim=hidden_dim,
            num_inputs=self._num_future_numeric + self._num_future_categorical,
            hidden_dim=hidden_dim, dropout=dropout, context_dim=hidden_dim)

        static_covariate_encoder = GatedResidualNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
            dropout=dropout)
        self._static_encoder_selection = copy.deepcopy(static_covariate_encoder)
        self._static_encoder_enrichment = copy.deepcopy(static_covariate_encoder)
        self._static_encoder_sequential_cell_init = copy.deepcopy(static_covariate_encoder)
        self._static_encoder_sequential_state_init = copy.deepcopy(static_covariate_encoder)

        self._past_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                                  num_layers=lstm_layers_num, dropout=dropout,
                                  batch_first=True)
        self._future_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                                    num_layers=lstm_layers_num, dropout=dropout,
                                    batch_first=True)
        self._post_lstm_gating = GateAddNorm(input_dim=hidden_dim, dropout=dropout)

        self._static_enrichment_grn = GatedResidualNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
            context_dim=hidden_dim, dropout=dropout)

        self._multihead_attn = InterpretableMultiHeadAttention(
            embed_dim=hidden_dim, num_heads=attention_heads_num)
        self._post_attention_gating = GateAddNorm(input_dim=hidden_dim, dropout=dropout)

        self._pos_wise_ff_grn = GatedResidualNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
            dropout=dropout)
        self._pos_wise_ff_gating = GateAddNorm(input_dim=hidden_dim, dropout=None)

        self._output_layer = nn.Linear(hidden_dim, self._num_outputs)

    @staticmethod
    def replicate_along_time(static_signal, time_steps):
        return static_signal.unsqueeze(1).repeat(1, time_steps, 1)

    @staticmethod
    def stack_time_steps_along_batch(temporal_signal):
        return temporal_signal.reshape(-1, temporal_signal.shape[-1])

    def apply_temporal_selection(self, temporal_representation,
                                 static_selection_signal, temporal_selection_module):
        num_samples, num_temporal_steps, _ = temporal_representation.shape
        if static_selection_signal is not None:
            time_distributed_context = self.replicate_along_time(
                static_selection_signal, num_temporal_steps)
            time_distributed_context = self.stack_time_steps_along_batch(
                time_distributed_context)
        else:
            time_distributed_context = None
        temporal_flattened_embedding = self.stack_time_steps_along_batch(
            temporal_representation)
        temporal_selection_output, temporal_selection_weights = \
            temporal_selection_module(temporal_flattened_embedding,
                                      context=time_distributed_context)
        temporal_selection_output = temporal_selection_output.reshape(
            num_samples, num_temporal_steps, -1)
        temporal_selection_weights = temporal_selection_weights.squeeze(-1).reshape(
            num_samples, num_temporal_steps, -1)
        return temporal_selection_output, temporal_selection_weights

    def transform_inputs(self, batch):
        empty = torch.zeros((0, 0, 0), device=batch.get("past_target", torch.zeros(1)).device
                            if isinstance(batch.get("past_target"), torch.Tensor)
                            else next(iter(batch.values())).device)
        static_rep = self._static_transform(
            x_numeric=batch.get('static_cov_numeric', empty).squeeze(1),
            x_categorical=batch.get('static_cov_categorical', empty).squeeze(1))
        if self._num_historical_numeric > 0:
            historical_ts_numeric = torch.cat([
                batch.get("past_target", empty),
                batch.get("known_cov_numeric", empty)[:, :self._in_chunk_len, :],
                batch.get("observed_cov_numeric", empty)], dim=-1)
        else:
            historical_ts_numeric = empty
        if self._num_historical_categorical > 0:
            historical_ts_categorical = torch.cat([
                batch.get("known_cov_categorical", empty)[:, :self._in_chunk_len, :],
                batch.get("observed_cov_categorical", empty)], dim=-1)
        else:
            historical_ts_categorical = empty
        historical_ts_rep = self._historical_ts_transform(
            x_numeric=historical_ts_numeric, x_categorical=historical_ts_categorical)
        future_ts_rep = self._future_ts_transform(
            x_numeric=batch.get('known_cov_numeric', empty)[:, self._in_chunk_len:, :],
            x_categorical=batch.get('known_cov_categorical', empty)[:, self._in_chunk_len:, :])
        return future_ts_rep, historical_ts_rep, static_rep

    def get_static_encoders(self, selected_static):
        c_selection = self._static_encoder_selection(selected_static)
        c_enrichment = self._static_encoder_enrichment(selected_static)
        c_seq_hidden = self._static_encoder_sequential_state_init(selected_static)
        c_seq_cell = self._static_encoder_sequential_cell_init(selected_static)
        return c_enrichment, c_selection, c_seq_cell, c_seq_hidden

    def apply_sequential_processing(self, selected_historical, selected_future,
                                    c_seq_hidden, c_seq_cell):
        lstm_input = torch.cat([selected_historical, selected_future], dim=1)
        if c_seq_hidden is not None:
            hidden = (c_seq_hidden.unsqueeze(0).repeat(self._lstm_layers, 1, 1),
                      c_seq_cell.unsqueeze(0).repeat(self._lstm_layers, 1, 1))
            past_lstm_output, hidden = self._past_lstm(selected_historical, hidden)
        else:
            past_lstm_output, hidden = self._past_lstm(selected_historical)
        future_lstm_output, _ = self._future_lstm(selected_future, hidden)
        lstm_output = torch.cat([past_lstm_output, future_lstm_output], dim=1)
        gated_lstm_output = self._post_lstm_gating(lstm_output, residual=lstm_input)
        return gated_lstm_output

    def apply_static_enrichment(self, gated_lstm_output, static_enrichment_signal):
        num_samples, num_temporal_steps, _ = gated_lstm_output.shape
        if static_enrichment_signal is not None:
            time_distributed_context = self.replicate_along_time(
                static_enrichment_signal, num_temporal_steps)
            time_distributed_context = self.stack_time_steps_along_batch(
                time_distributed_context)
        else:
            time_distributed_context = None
        flattened_gated_lstm_output = self.stack_time_steps_along_batch(gated_lstm_output)
        enriched_sequence = self._static_enrichment_grn(
            flattened_gated_lstm_output, context=time_distributed_context)
        enriched_sequence = enriched_sequence.reshape(
            num_samples, -1, self._state_size)
        return enriched_sequence

    def apply_self_attention(self, enriched_sequence, num_historical_steps,
                             num_future_steps):
        output_sequence_length = num_future_steps
        mask = torch.cat([
            torch.zeros((output_sequence_length, num_historical_steps),
                        device=enriched_sequence.device),
            torch.triu(torch.ones((output_sequence_length, output_sequence_length),
                                  device=enriched_sequence.device), diagonal=1)],
            dim=1)
        post_attention, attention_outputs, attention_scores = self._multihead_attn(
            q=enriched_sequence[:, num_historical_steps:, :],
            k=enriched_sequence, v=enriched_sequence, mask=mask.bool())
        gated_post_attention = self._post_attention_gating(
            x=post_attention,
            residual=enriched_sequence[:, num_historical_steps:, :])
        return gated_post_attention, attention_scores

    def forward(self, batch):
        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)
        if static_rep is not None:
            selected_static, static_weights = self._static_selection(static_rep)
            c_enrichment, c_selection, c_seq_cell, c_seq_hidden = \
                self.get_static_encoders(selected_static)
        else:
            static_weights = None
            c_enrichment = c_selection = c_seq_cell = c_seq_hidden = None
        selected_historical, historical_selection_weights = \
            self.apply_temporal_selection(
                temporal_representation=historical_ts_rep,
                static_selection_signal=c_selection,
                temporal_selection_module=self._historical_ts_selection)
        selected_future, future_selection_weights = self.apply_temporal_selection(
            temporal_representation=future_ts_rep,
            static_selection_signal=c_selection,
            temporal_selection_module=self._future_ts_selection)
        gated_lstm_output = self.apply_sequential_processing(
            selected_historical=selected_historical, selected_future=selected_future,
            c_seq_hidden=c_seq_hidden, c_seq_cell=c_seq_cell)
        enriched_sequence = self.apply_static_enrichment(
            gated_lstm_output=gated_lstm_output,
            static_enrichment_signal=c_enrichment)
        gated_post_attention, attention_scores = self.apply_self_attention(
            enriched_sequence=enriched_sequence,
            num_historical_steps=self._in_chunk_len,
            num_future_steps=self._out_chunk_len)
        post_poswise_ff_grn = self._pos_wise_ff_grn(gated_post_attention)
        gated_poswise_ff = self._pos_wise_ff_gating(
            post_poswise_ff_grn,
            residual=gated_lstm_output[:, self._in_chunk_len:, :])
        predicted_quantiles = self._output_layer(gated_poswise_ff).reshape(
            gated_poswise_ff.shape[0], self._out_chunk_len, self._target_dim, -1)
        if not self._interpretable_output:
            return predicted_quantiles
        return {
            'static_weights': static_weights.squeeze(-1) if static_weights is not None else None,
            'historical_selection_weights': historical_selection_weights,
            'future_selection_weights': future_selection_weights,
            'attention_scores': attention_scores,
        }
