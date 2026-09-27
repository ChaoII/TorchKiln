"""kokoro TTS 任务适配器（训练/评估跑通；损失为自定，见 kokoro_loss.py 说明）。

注册方式与仓库其它任务一致：
  * ``ptcore/trainers/__init__.py`` 的 ``TRAINER_REGISTRY``
  * ``torchkiln/tasks/__init__.py`` 的 TASK_MAP
  * ``torchkiln/cli.py`` 的 ``TASK_ALIASES`` / ``FAMILY_OF``
"""
from __future__ import annotations

import json
import os

import torch

from ptcore.task import TaskAdapter
from torchkiln.audio.kokoro_dataset import KokoroDataset, kokoro_collate
from torchkiln.audio.kokoro_loss import KokoroLoss

__all__ = ["KokoroTtsTask"]


class KokoroTtsTask(TaskAdapter):
    """kokoro-82M 文本转语音（训练 + 评估）。

    ⚠️ 训练时必须**按模块分段反传**（整链一次反传会触发原生段错误 0xC0000005，见 AGENTS）。
       本适配器在 ``forward_train`` 里返回各中间量，loss 由 ``KokoroLoss`` 计算；
       trainer 侧若走整链 backward 崩溃，可改用 ``segment_backward=True`` 的最小实现。
    """

    def __init__(self, config=None):
        cfg = (config or {}).get("Architecture", {}) if config else {}
        self.name = cfg.get("task", "kokoro_tts")

    # ------------------------------------------------------------------ build
    def build_post_process(self, config):
        return None

    def build_model(self, config, post_process):
        from torchkiln.audio.kokoro.model import KModel
        arch = config["Architecture"]
        cfg_path = arch.get("config_path")
        w_path = arch.get("weights_path")
        assert cfg_path and os.path.isfile(cfg_path), \
            "Architecture.config_path 需指向 kokoro 的 config.json"
        kcfg = json.load(open(cfg_path, encoding="utf-8"))
        km = KModel.__new__(KModel)
        torch.nn.Module.__init__(km)
        km.repo_id = "local"
        km.vocab = kcfg["vocab"]

        import torch.nn as nn
        from torchkiln.audio.kokoro.modules import (
            CustomAlbert, ProsodyPredictor, TextEncoder)
        from torchkiln.audio.kokoro.istftnet import Decoder
        from transformers import AlbertConfig
        km.bert = CustomAlbert(AlbertConfig(vocab_size=kcfg["n_token"],
                                            **kcfg["plbert"]))
        km.bert_encoder = nn.Linear(km.bert.config.hidden_size,
                                    kcfg["hidden_dim"])
        km.context_length = km.bert.config.max_position_embeddings
        km.predictor = ProsodyPredictor(
            style_dim=kcfg["style_dim"], d_hid=kcfg["hidden_dim"],
            nlayers=kcfg["n_layer"], max_dur=kcfg["max_dur"],
            dropout=kcfg["dropout"])
        km.text_encoder = TextEncoder(
            channels=kcfg["hidden_dim"],
            kernel_size=kcfg["text_encoder_kernel_size"],
            depth=kcfg["n_layer"], n_symbols=kcfg["n_token"])
        km.decoder = Decoder(dim_in=kcfg["hidden_dim"],
                             style_dim=kcfg["style_dim"],
                             dim_out=kcfg["n_mels"], disable_complex=False,
                             **kcfg["istftnet"])
        if w_path and os.path.isfile(w_path):
            sd = torch.load(w_path, map_location="cpu", weights_only=True)
            for key in ["bert", "bert_encoder", "predictor", "text_encoder",
                        "decoder"]:
                getattr(km, key).load_state_dict(
                    {k[7:]: v for k, v in sd[key].items()}, strict=False)
        km.kokoro_cfg = kcfg
        return km

    def build_loss(self, config, model):
        lc = config.get("Loss", {}) or {}
        return KokoroLoss(
            sr=lc.get("sr", 24000), n_fft=lc.get("n_fft", 1024),
            hop_length=lc.get("hop_length", 256), n_mels=lc.get("n_mels", 80),
            w_mel=lc.get("w_mel", 1.0), w_f0=lc.get("w_f0", 1.0),
            w_energy=lc.get("w_energy", 1.0),
            w_duration=lc.get("w_duration", 1.0))

    def build_metric(self, config):
        return _TtsMetric()

    def build_datasets(self, config, logger):
        d = config.get("Train", {}).get("dataset", {})
        e = config.get("Eval", {}).get("dataset", {}) or {}
        arch = config.get("Architecture", {})
        tr = KokoroDataset(
            d["label_file_list"][0], d.get("data_dir", ""),
            arch.get("config_path"), sr=d.get("sr", 24000),
            max_len=d.get("max_len", 24000 * 6))
        ev = None
        if e.get("label_file_list"):
            ev = KokoroDataset(
                e["label_file_list"][0], e.get("data_dir", ""),
                arch.get("config_path"), sr=e.get("sr", 24000),
                max_len=e.get("max_len", 24000 * 6))
        return tr, ev

    # -------------------------------------------------------------- dataloader
    def train_collate(self, batch):
        return kokoro_collate(batch)

    def eval_collate(self, batch):
        return kokoro_collate(batch)

    # ----------------------------------------------------------------- forward
    def forward_train(self, model, images, batch):
        """可微前向；``images`` = ``input_ids``（trainer 传 batch[0]）。"""
        input_ids = images
        ref_s = batch[1] if len(batch) > 1 else None
        speed = 1.0
        km = model
        il = torch.full((input_ids.shape[0],), input_ids.shape[-1],
                        dtype=torch.long, device=input_ids.device)
        tm = torch.arange(il.max(), device=input_ids.device).unsqueeze(0).expand(
            input_ids.shape[0], -1).type_as(il)
        tm = torch.gt(tm + 1, il.unsqueeze(1))

        bd = km.bert(input_ids, attention_mask=(~tm).int())
        d_en = km.bert_encoder(bd).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = km.predictor.text_encoder(d_en, s, il, tm)
        x, _ = km.predictor.lstm(d)
        dur_logits = km.predictor.duration_proj(x)              # (B,T,max_dur)
        duration = torch.sigmoid(dur_logits).sum(axis=-1) / speed   # (B,T)
        # 注意: 不能 squeeze() —— batch>1 时会变多维，repeat_interleave 要求 0/1 维
        pred_dur = torch.round(duration).clamp(min=1).long()    # (B,T)

        # 逐样本构造对齐矩阵；长度对齐（batch 内取同一 L'，避免 stack 失败）
        B, T = pred_dur.shape
        Ls = [(int(pred_dur[bi].sum()), bi) for bi in range(B)]
        Lmax = max(x[0] for x in Ls)
        pats = []
        for bi in range(B):
            n = int(pred_dur[bi].sum())
            idx = torch.repeat_interleave(
                torch.arange(T, device=input_ids.device), pred_dur[bi])
            p_ = torch.zeros((T, Lmax), device=input_ids.device)
            p_[:T, : idx.shape[0]][idx, torch.arange(idx.shape[0],
                                                     device=input_ids.device)] = 1
            pats.append(p_)
        pat = torch.stack(pats)                                  # (B,T,Lmax)
        en = d.transpose(-1, -2) @ pat
        F0, N = km.predictor.F0Ntrain(en, s)
        te = km.text_encoder(input_ids, il, tm)
        asr = te @ pat
        audio = km.decoder(asr, F0, N, ref_s[:, :128])
        audio = audio.squeeze(1) if audio.dim() == 3 else audio
        return {"audio": audio, "F0": F0, "N": N,
                "duration_logits": dur_logits, "pred_dur": pred_dur}

    def eval_step(self, model, batch, post_process, metric, device):
        ids = batch[0].to(device)
        ref_s = batch[1].to(device)
        wav = batch[2].to(device)
        model.eval()
        with torch.no_grad():
            preds = self.forward_train(model, ids, [ids, ref_s, wav])
        metric(preds, {"wav": wav})
    def summary_lines(self, config, global_config, post_process):
        a = config.get("Architecture", {})
        return ["kokoro TTS: config_path=%s weights_path=%s"
                % (a.get("config_path"), a.get("weights_path")),
                "⚠️ 损失为自定（官方未公开训练代码）"]


class _TtsMetric(object):
    """极简 TTS 指标：波形 L1 + 余弦（用于 smoke 冒烟；正式评测需 mel/F0 指标）。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.wav_l1, self.cos, self.n = 0.0, 0.0, 0

    def __call__(self, preds, ref):
        a = preds["audio"].detach().float()
        b = ref["wav"].detach().float()
        L = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :L].reshape(a.shape[0], -1), b[..., :L].reshape(b.shape[0], -1)
        self.wav_l1 += float((a - b).abs().mean())
        self.cos += float(torch.nn.functional.cosine_similarity(
            a, b, dim=-1).mean())
        self.n += 1

    def get_metric(self):
        n = max(self.n, 1)
        return {"wav_l1": self.wav_l1 / n, "wav_cos": self.cos / n}
