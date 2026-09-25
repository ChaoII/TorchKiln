"""Informer forecaster — PyTorch port of PaddleTS ``_InformerModule``.

Input is a dict ``{"past_target"}`` and output is ``(B, out_chunk_len, target_dim)``.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Informer"]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position_embedding = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-1. * np.log2(1e4) / d_model))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("_position_embedding", position_embedding)

    def forward(self, x):
        return self._position_embedding[: x.shape[1], :]


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_stamp, d_model):
        super().__init__()
        self._timefeat_embedding = nn.Linear(d_stamp, d_model)

    def forward(self, x):
        return self._timefeat_embedding(x)


class TokenEmbedding(nn.Module):
    def __init__(self, target_dim, d_model):
        super().__init__()
        self._token_embedding = nn.Conv1d(
            target_dim, d_model, kernel_size=3, padding=1, padding_mode="circular")

    def forward(self, x):
        out = self._token_embedding(x.transpose(1, 2)).transpose(1, 2)
        return out


class MixedEmbedding(nn.Module):
    def __init__(self, target_dim, d_model, max_len, dropout_rate=0.1):
        super().__init__()
        self._token_embedding = TokenEmbedding(target_dim, d_model)
        self._position_embedding = PositionalEmbedding(d_model, max_len)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self._token_embedding(x) + self._position_embedding(x)
        return self._dropout(out)


class _CrossAttention(nn.Module):
    def __init__(self, embed_dim, kdim, vdim, num_heads, dropout_rate=0.1):
        super().__init__()
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._head_dim = embed_dim // num_heads
        self._dropout = nn.Dropout(dropout_rate)
        self._norm = nn.LayerNorm(embed_dim)
        self._q_proj = nn.Linear(embed_dim, embed_dim)
        self._k_proj = nn.Linear(kdim, embed_dim)
        self._v_proj = nn.Linear(vdim, embed_dim)
        self._out_proj = nn.Linear(embed_dim, embed_dim)

    def _prepare_qkv(self, query, key, value):
        q = self._q_proj(query)
        k = self._k_proj(key)
        v = self._v_proj(value)
        q = q.reshape(q.shape[0], q.shape[1], self._num_heads, self._head_dim).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self._num_heads, self._head_dim).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self._num_heads, self._head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, query, key, value, attn_mask):
        batch_size = query.shape[0]
        q, k, v = self._prepare_qkv(query, key, value)
        scores = torch.matmul(q * np.sqrt(self._head_dim), k.transpose(2, 3))
        if attn_mask is not None:
            m = attn_mask[None, None, :, :].repeat(batch_size, self._num_heads, 1, 1)
            scores = scores + m
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self._num_heads * self._head_dim)
        out = self._out_proj(context)
        out = self._dropout(out)
        return self._norm(out + query)


class _ProbSparseAttention(nn.Module):
    def __init__(self, embed_dim, kdim, vdim, num_heads, dropout_rate=0.1):
        super().__init__()
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._head_dim = embed_dim // num_heads
        self._dropout = nn.Dropout(dropout_rate)
        self._norm = nn.LayerNorm(embed_dim)
        self._q_proj = nn.Linear(embed_dim, embed_dim)
        self._k_proj = nn.Linear(kdim, embed_dim)
        self._v_proj = nn.Linear(vdim, embed_dim)
        self._out_proj = nn.Linear(embed_dim, embed_dim)

    def _prepare_qkv(self, query, key, value):
        q = self._q_proj(query)
        k = self._k_proj(key)
        v = self._v_proj(value)
        q = q.reshape(q.shape[0], q.shape[1], self._num_heads, self._head_dim).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self._num_heads, self._head_dim).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self._num_heads, self._head_dim).transpose(1, 2)
        return q, k, v

    def _compute_prob_qk(self, query, key, sample_k, n_top):
        B, H, L_K, _ = key.shape
        _, _, L_Q, _ = query.shape
        K_expand = key[:, :, None, :, :].repeat(1, 1, L_Q, 1, 1)
        sampled_idx = torch.randint(0, L_K, (L_Q, sample_k), device=query.device)[
            None, None, :, :, None].repeat(B, H, 1, 1, 1)
        K_sample = torch.gather(K_expand, -2, sampled_idx.expand(-1, -1, -1, -1, key.shape[-1]))
        Q_K_sample = torch.matmul(query[:, :, :, None, :],
                                  K_sample.transpose(-1, -2))
        Q_K_sample = Q_K_sample.squeeze(-2)
        M = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)
        M_top_index = M.topk(n_top, dim=-1, sorted=False).indices
        Q_sample = torch.gather(query, -2, M_top_index[:, :, :, None].expand(-1, -1, -1, query.shape[-1]))
        return Q_sample, M_top_index

    def _get_initial_context(self, value, L_Q, attn_mask):
        if attn_mask is not None:
            return torch.cumsum(value, dim=-2)
        context = value.mean(dim=-2)[:, :, None, :].repeat(1, 1, L_Q, 1)
        return context

    def forward(self, query, key, value, attn_mask):
        L_K = key.shape[1]
        L_Q = query.shape[1]
        batch_size = query.shape[0]
        u_k = max(min(int(5 * np.log(L_K)), L_Q), 1)
        u_q = max(min(int(5 * np.log(L_Q)), L_Q), 1)
        q, k, v = self._prepare_qkv(query, key, value)
        Q_sample, index = self._compute_prob_qk(q, k, sample_k=u_k, n_top=u_q)
        scores = torch.matmul(Q_sample * np.sqrt(self._head_dim), k.transpose(2, 3))
        if attn_mask is not None:
            m = attn_mask[None, None, :, :].repeat(batch_size, self._num_heads, 1, 1)
            m = torch.gather(m, -2, index[:, :, :, None].expand(-1, -1, -1, m.shape[-1]))
            scores = scores + m
        attn = F.softmax(scores, dim=-1)
        values = torch.matmul(attn, v)
        context = self._get_initial_context(v, L_Q, attn_mask)
        context = context.scatter(-2, index[:, :, :, None].expand(-1, -1, -1, context.shape[-1]), values)
        context = context.transpose(1, 2).reshape(batch_size, -1, self._num_heads * self._head_dim)
        out = self._out_proj(context)
        out = self._dropout(out)
        return self._norm(out + query)


class _ConvLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self._maxpool = nn.MaxPool1d(3, 2, 1)
        self._norm = nn.BatchNorm1d(d_model)
        self._activation = nn.ELU()
        self._downconv = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="circular")

    def forward(self, src):
        out = src.transpose(1, 2)
        out = self._downconv(out)
        out = self._norm(out)
        out = self._activation(out)
        out = self._maxpool(out)
        return out.transpose(1, 2)


class _InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_channels, activation, dropout_rate):
        super().__init__()
        self._attn = _ProbSparseAttention(d_model, d_model, d_model, num_heads, dropout_rate)
        self._conv1 = nn.Conv1d(d_model, ffn_channels, 1)
        self._conv2 = nn.Conv1d(ffn_channels, d_model, 1)
        self._norm = nn.LayerNorm(d_model)
        self._dropout = nn.Dropout(dropout_rate)
        self._activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, src, src_mask=None):
        out = residual = self._attn(src, src, src, src_mask)
        out = out.transpose(1, 2)
        out = self._conv1(out)
        out = self._activation(out)
        out = self._dropout(out)
        out = self._conv2(out)
        out = out.transpose(1, 2)
        out = self._dropout(out)
        return self._norm(out + residual)


class _InformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_channels, activation, dropout_rate, num_layers):
        super().__init__()
        self._encoder_layers = nn.ModuleList([
            _InformerEncoderLayer(d_model, num_heads, ffn_channels, activation, dropout_rate)
            for _ in range(num_layers)])
        self._conv_layers = nn.ModuleList([
            _ConvLayer(d_model) for _ in range(num_layers - 1)])

    def forward(self, src, src_mask=None):
        for enc_layer, conv_layer in zip(self._encoder_layers[:-1], self._conv_layers):
            src = enc_layer(src, src_mask)
            src = conv_layer(src)
        src = self._encoder_layers[-1](src, src_mask)
        return src


class _InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_channels, dropout_rate):
        super().__init__()
        self._selfAttn = _ProbSparseAttention(d_model, d_model, d_model, num_heads, dropout_rate)
        self._crossAttn = _CrossAttention(d_model, d_model, d_model, num_heads, dropout_rate)
        self._conv1 = nn.Conv1d(d_model, ffn_channels, 1)
        self._conv2 = nn.Conv1d(ffn_channels, d_model, 1)
        self._norm = nn.LayerNorm(d_model)
        self._dropout = nn.Dropout(dropout_rate)
        self._activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        out = self._selfAttn(tgt, tgt, tgt, tgt_mask)
        out = residual = self._crossAttn(out, memory, memory, memory_mask)
        out = out.transpose(1, 2)
        out = self._conv1(out)
        out = self._activation(out)
        out = self._dropout(out)
        out = self._conv2(out)
        out = out.transpose(1, 2)
        out = self._dropout(out)
        return self._norm(residual + out)


class _InformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_channels, dropout_rate, num_layers):
        super().__init__()
        self._decoder_layers = nn.ModuleList([
            _InformerDecoderLayer(d_model, num_heads, ffn_channels, dropout_rate)
            for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for dec_layer in self._decoder_layers:
            tgt = dec_layer(tgt, memory, tgt_mask, memory_mask)
        return tgt


class _Informer(nn.Module):
    def __init__(self, d_model, nhead, ffn_channels, num_encoder_layers,
                 num_decoder_layers, activation, dropout_rate=0.1):
        super().__init__()
        self._encoder = _InformerEncoder(d_model, nhead, ffn_channels, activation,
                                         dropout_rate, num_encoder_layers)
        self._decoder = _InformerDecoder(d_model, nhead, ffn_channels, dropout_rate,
                                         num_decoder_layers)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self._encoder(src, src_mask)
        out = self._decoder(tgt, memory, tgt_mask, memory_mask)
        return out


class Informer(nn.Module):
    """Informer (PaddleTS ``_InformerModule``)."""

    def __init__(self, in_chunk_len, out_chunk_len, target_dim=1, start_token_len=0,
                 d_model=512, nhead=8, ffn_channels=2048, num_encoder_layers=2,
                 num_decoder_layers=1, activation="relu", dropout_rate=0.1):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._start_token_len = start_token_len
        self._target_dim = target_dim
        self._src_embedding = MixedEmbedding(target_dim, d_model, in_chunk_len, dropout_rate)
        self._tgt_embedding = MixedEmbedding(target_dim, d_model,
                                             start_token_len + out_chunk_len, dropout_rate)
        self._informer = _Informer(
            d_model, nhead, ffn_channels, num_encoder_layers, num_decoder_layers,
            activation, dropout_rate)
        self._out_proj = nn.Linear(d_model, target_dim)

    def _create_informer_inputs(self, x):
        src = x["past_target"]
        batch_size, _, d_model = src.shape
        tgt = src[:, self._in_chunk_len - self._start_token_len:, :]
        padding = torch.zeros([batch_size, self._out_chunk_len, d_model], dtype=src.dtype)
        tgt = torch.cat([tgt, padding], dim=1)
        return src, tgt

    def forward(self, x):
        src, tgt = self._create_informer_inputs(x)
        src = self._src_embedding(src)
        tgt = self._tgt_embedding(tgt)
        out = self._informer(src, tgt)
        out = self._out_proj(out)
        return out[:, -self._out_chunk_len:, :]
