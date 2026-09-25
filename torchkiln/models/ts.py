"""Time-series model builder (PaddleTS-compatible models)."""
from __future__ import absolute_import

__all__ = ["build_ts_model"]


def build_ts_model(arch):
    from torchkiln.nn import ts_models as T

    head = arch.get("Head") or {}
    model = str(head.get("model", arch.get("algorithm", "nbeats"))).lower()
    in_len = int(head.get("in_chunk_len", arch.get("in_chunk_len", 96)))
    out_len = int(head.get("out_chunk_len", arch.get("out_chunk_len", 24)))
    dim = int(head.get("target_dim", arch.get("target_dim", 1)))
    kw = {k: head[k] for k in (
        "num_stacks", "num_blocks", "num_layers", "layer_widths",
        "expansion_coefficient_dim", "trend_polynomial_degree",
        "hidden_config", "use_bn", "kernel_size", "dropout_rate",
        "pooling_kernel_sizes", "n_freq_downsample", "batch_norm",
        "dropout", "activation", "MaxPool1d") if k in head}
    if model in ("nbeats", "n-beats"):
        return T.NBEATS(in_chunk_len=in_len, out_chunk_len=out_len, target_dim=dim,
                        generic_architecture=bool(head.get("generic_architecture", True)),
                        **kw)
    if model in ("nhits", "n-hits"):
        return T.NHiTS(in_chunk_len=in_len, out_chunk_len=out_len, target_dim=dim, **kw)
    if model in ("mlp", "dlinear"):
        cls = T.MLP if model == "mlp" else T.DLinear
        base = dict(in_chunk_len=in_len, out_chunk_len=out_len, target_dim=dim)
        if model == "mlp":
            base["hidden_config"] = kw.pop("hidden_config", (128, 128))
            base["use_bn"] = kw.pop("use_bn", False)
        else:
            base["kernel_size"] = kw.pop("kernel_size", 25)
        return cls(**base)
    if model == "tcn":
        base = dict(in_chunk_len=in_len, out_chunk_len=out_len, target_dim=dim,
                    hidden_config=kw.pop("hidden_config", (128, 128)),
                    kernel_size=kw.pop("kernel_size", 3),
                    dropout_rate=kw.pop("dropout_rate", 0.0))
        return T.TCN(**base)
    # ------- PaddleTS (paddlets) SOTA models ------- #
    if model in ("rnn", "rnnblock"):
        from torchkiln.nn.s_rnn import RNNBlock
        return RNNBlock(
            in_len, out_len, target_dim=dim,
            rnn_type=head.get("rnn_type", "LSTM"),
            hidden_dim=head.get("hidden_dim", 128),
            num_layers_recurrent=head.get("num_layers_recurrent", 1),
            out_fcn_config=head.get("out_fcn_config"),
            dropout=head.get("dropout", 0.0))
    if model == "lstnet":
        from torchkiln.nn.s_lstnet import LSTNet
        return LSTNet(
            in_len, out_len, target_dim=dim,
            skip_size=head.get("skip_size", 1),
            channels=head.get("channels", 1),
            kernel_size=head.get("kernel_size", 3),
            rnn_cell_type=head.get("rnn_cell_type", "GRU"),
            rnn_num_cells=head.get("rnn_num_cells", 10),
            skip_rnn_cell_type=head.get("skip_rnn_cell_type", "GRU"),
            skip_rnn_num_cells=head.get("skip_rnn_num_cells", 10),
            dropout_rate=head.get("dropout_rate", 0.2),
            output_activation=head.get("output_activation"))
    if model == "transformer":
        from torchkiln.nn.s_transformer import Transformer
        return Transformer(
            in_len, out_len, target_dim=dim,
            d_model=head.get("d_model", 8),
            nhead=head.get("nhead", 4),
            num_encoder_layers=head.get("num_encoder_layers", 1),
            num_decoder_layers=head.get("num_decoder_layers", 1),
            dim_feedforward=head.get("dim_feedforward", 64),
            activation=head.get("activation", "relu"),
            dropout_rate=head.get("dropout_rate", 0.1))
    if model == "scinet":
        from torchkiln.nn.s_scinet import SCINet
        return SCINet(
            in_len, out_len, target_dim=dim,
            num_stack=head.get("num_stack", 1),
            num_level=head.get("num_level", 3),
            num_decoder_layer=head.get("num_decoder_layer", 1),
            concat_len=head.get("concat_len", 0),
            kernel_size=head.get("kernel_size", 5),
            dropout_rate=head.get("dropout_rate", 0.5),
            num_group=head.get("num_group", 1),
            hidden_size=head.get("hidden_size", 1))
    if model == "informer":
        from torchkiln.nn.s_informer import Informer
        return Informer(
            in_len, out_len, target_dim=dim,
            start_token_len=head.get("start_token_len", 0),
            d_model=head.get("d_model", 512),
            nhead=head.get("nhead", 8),
            ffn_channels=head.get("ffn_channels", 2048),
            num_encoder_layers=head.get("num_encoder_layers", 2),
            num_decoder_layers=head.get("num_decoder_layers", 1),
            activation=head.get("activation", "relu"),
            dropout_rate=head.get("dropout_rate", 0.1))
    if model == "deepar":
        from torchkiln.nn.s_deepar import DeepAR
        return DeepAR(
            in_len, out_len, target_dim=dim,
            known_cov_dim=head.get("known_cov_dim", 0),
            rnn_type=head.get("rnn_type", "LSTM"),
            hidden_dim=head.get("hidden_dim", 128),
            num_layers_recurrent=head.get("num_layers_recurrent", 1),
            drop_out=head.get("dropout", 0.0),
            num_samples=head.get("num_samples", 10),
            regression_mode=head.get("regression_mode", "mean"),
            output_mode=head.get("output_mode", "quantiles"))
    if model == "tft":
        from torchkiln.nn.s_tft import TemporalFusionTransformer
        return TemporalFusionTransformer(
            in_len, out_len, {
                "target_dim": dim,
                "known_num_dim": head.get("known_num_dim", 0),
                "known_cat_dim": head.get("known_cat_dim", 0),
                "observed_num_dim": head.get("observed_num_dim", 0),
                "observed_cat_dim": head.get("observed_cat_dim", 0),
                "known_cat_size": head.get("known_cat_size", []),
                "observed_cat_size": head.get("observed_cat_size", []),
                "static_num_dim": head.get("static_num_dim", 0),
                "static_cat_dim": head.get("static_cat_dim", 0),
                "static_cat_size": head.get("static_cat_size", []),
            },
            hidden_dim=head.get("hidden_dim", 64),
            lstm_layers_num=head.get("lstm_layers_num", 1),
            attention_heads_num=head.get("attention_heads_num", 2),
            output_quantiles=head.get("output_quantiles", [0.1, 0.5, 0.9]),
            dropout=head.get("dropout", 0.0))
    raise ValueError("unknown ts model: {}".format(model))
