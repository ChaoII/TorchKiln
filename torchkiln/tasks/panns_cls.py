"""PANNs 音频分类任务（torchkiln 侧）—— 用于与 PaddleSpeech 做**训练对齐**。

复刻 PaddleSpeech ``cls/exps/panns/train.py`` 的流水线：
  feats = LogMelSpectrogram(sr=32000, n_fft=1024, hop=320, win=1024, n_mels=64,
                            f_min=50, f_max=14000)   # 然后 transpose -> (N, T, 64)
  x     = feats.unsqueeze(1)                          # -> (N, 1, T, 64)  ← CNN14 需要 4D
  logits= SoundClassifier(CNN14(extract_embedding=True), num_class)(x)
  loss  = CrossEntropyLoss(logits, label)
  optim = Adam(lr)

⚠️ 官方 `panns.yaml` 未随 wheel 发布 ⇒ 超参为**本项目自定**（在配置里注明）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ptcore.task import TaskAdapter
from torchkiln.audio.esc50_dataset import ESC50Dataset, esc50_collate

__all__ = ["PannsClsTask", "PannsFEATURE"]


def mag2db(x):
    return 10.0 * torch.log10(x.clamp(min=1e-10))


class PannsFEATURE(nn.Module):
    """PaddleSpeech `LogMelSpectrogram(**feat_conf)` 的 torch 复刻（可微、按设备）。"""

    def __init__(self, sr=32000, n_fft=1024, hop_length=320, win_length=1024,
                 window="hann", f_min=50.0, f_max=14000.0, n_mels=64):
        super().__init__()
        assert window == "hann", window
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window_fn=torch.hann_window,
            center=True, pad_mode="reflect", power=2.0, n_mels=n_mels,
            f_min=f_min, f_max=f_max, norm="slaney", mel_scale="slaney")

    def forward(self, wav):
        """wav: (N, L) -> (N, n_mels, T) 与 paddle LogMelSpectrogram 同布局。"""
        m = self.spec(wav)
        return mag2db(m)


class PannsClassifier(nn.Module):
    """PaddleSpeech `SoundClassifier`：backbone + Dropout + Linear(emb_size, n)。

    ``feature`` 是 PaddleSpeech `train.py` 里独立于模型的 `LogMelSpectrogram`
    （官方把它放在训练循环外），这里挂到模型上以便 trainer 统一处理。
    """

    def __init__(self, backbone, num_class, dropout=0.1, feat_conf=None):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.emb_size, num_class)
        self.feature = PannsFEATURE(**(feat_conf or {}))

    def forward(self, x):
        # x: (N, 1, T, n_mels)
        x = self.backbone(x)          # CNN14(extract_embedding=True) -> (N, emb)
        return self.fc(self.dropout(x))


class PannsClsTask(TaskAdapter):
    def __init__(self, config=None):
        arch = (config or {}).get("Architecture", {}) if config else {}
        self.name = arch.get("task", "panns_cls")

    # ---- build ----
    def build_post_process(self, config):
        return None

    def build_model(self, config, post_process):
        arch = config["Architecture"]
        from torchkiln.audio.panns import CNN14
        bb = CNN14(extract_embedding=True)
        m = PannsClassifier(bb, arch.get("num_class", 50),
                            dropout=arch.get("dropout", 0.1),
                            feat_conf=arch.get("feature"))
        w = arch.get("weights_path")
        if w:
            import os
            import numpy as np
            if w.endswith(".npz") and os.path.isfile(w):     # 与 Paddle 同起点
                sd = dict(np.load(w))
                # 载入**整个模型**（backbone + 可选分类头 fc）：
                #   * `init_cnn14.npz`  只有 backbone -> fc 保持随机（微调场景）
                #   * `ctrl_init_torch.npz` 含 fc -> 严格同起点（确定性对照场景）
                # 键名前缀兼容：npz 里可能是裸 `bn0.*`（需补 `backbone.`）或已带前缀。
                msd = m.state_dict()
                sub = {}
                for k, v in sd.items():
                    if k in msd:
                        sub[k] = torch.as_tensor(v)
                    elif ("backbone." + k) in msd:
                        sub["backbone." + k] = torch.as_tensor(v)
                r = m.load_state_dict(sub, strict=False)
                miss = [k for k in r.missing_keys
                        if "num_batches_tracked" not in k
                        and not k.endswith((".window", ".fb"))]  # 非训练 buffer
                print("[panns_cls] 同起点加载: 用 %d 键 (含头=%s), missing=%d, unexpected=%d"
                      % (len(sub), any(k.startswith("fc.") for k in sub),
                         len(miss), len(r.unexpected_keys)))
            elif os.path.isfile(w):
                ck = torch.load(w, map_location="cpu", weights_only=False)
                m.load_state_dict(ck.get("state_dict", ck), strict=False)
        return m

    def build_loss(self, config, model):
        return _PannsCE()

    def build_metric(self, config):
        return _ClsMetric()

    def build_datasets(self, config, logger):
        d = config.get("Train", {}).get("dataset", {})
        e = config.get("Eval", {}).get("dataset", {}) or {}
        tr = ESC50Dataset(d["data_dir"], mode="train",
                          split=d.get("split", 1),
                          sample_rate=d.get("sample_rate", 32000),
                          order_seed=d.get("order_seed"))
        ev = None
        if e.get("data_dir"):
            ev = ESC50Dataset(e["data_dir"], mode="dev",
                              split=e.get("split", 1),
                              sample_rate=e.get("sample_rate", 32000),
                              order_seed=e.get("order_seed"))
        return tr, ev

    # ---- data ----
    def train_collate(self, batch):
        return esc50_collate(batch)

    def eval_collate(self, batch):
        return esc50_collate(batch)

    # ---- forward ----
    def forward_train(self, model, images, batch):
        """images: (N, L) 波形；返回 logits。"""
        feats = model.feature(images)                 # (N, n_mels, T)
        feats = feats.transpose(1, 2).unsqueeze(1)    # -> (N, 1, T, n_mels)
        return model(feats)

    def eval_step(self, model, batch, post_process, metric, device):
        wav = batch[0].to(device)
        lab = batch[1].to(device)
        with torch.no_grad():
            logits = self.forward_train(model, wav, batch)
        metric(logits, lab)

    def summary_lines(self, config, global_config, post_process):
        a = config.get("Architecture", {})
        return ["PANNs 分类: backbone=CNN14 num_class=%s weights=%s"
                % (a.get("num_class"), a.get("weights_path")),
                "⚠️ 官方 panns.yaml 未发布 -> 超参为自定（见配置注释）"]


class _PannsCE(nn.CrossEntropyLoss):
    """trainer 传的 ``labels`` 是 **list** ``[wav, label]``（且可能被 ``_to_fp32`` 转了精度）。

    ⚠️ 返回值必须是 **dict**（trainer 取 ``loss_dict["loss"]``），
    这与 kokoro 的 `KokoroLoss` 是同一个约定。
    """

    def forward(self, logits, labels):
        tgt = labels
        if isinstance(labels, (list, tuple)):
            tgt = labels[-1]
        if not torch.is_tensor(tgt):
            tgt = torch.as_tensor(tgt)
        tgt = tgt.long().reshape(-1)
        if tgt.numel() != logits.shape[0]:
            for x in (labels if isinstance(labels, (list, tuple)) else [labels]):
                if torch.is_tensor(x) and x.numel() == logits.shape[0]:
                    tgt = x.long().reshape(-1)
                    break
        loss = super().forward(logits, tgt)
        return {"loss": loss, "cls_loss": loss.detach()}

    # ---- data ----
    def train_collate(self, batch):
        return esc50_collate(batch)

    def eval_collate(self, batch):
        return esc50_collate(batch)

    # ---- forward ----
    def forward_train(self, model, images, batch):
        """images: (N, L) 波形；返回 logits。"""
        fc = model.feature
        feats = fc(images)                       # (N, n_mels, T)
        feats = feats.transpose(1, 2).unsqueeze(1)   # -> (N, 1, T, n_mels)
        return model(feats)

    def eval_step(self, model, batch, post_process, metric, device):
        wav = batch[0].to(device)
        lab = batch[1].to(device)
        with torch.no_grad():
            logits = self.forward_train(model, wav, batch)
        metric(logits, lab)

    def summary_lines(self, config, global_config, post_process):
        a = config.get("Architecture", {})
        return ["PANNs 分类: backbone=CNN14 num_class=%s weights=%s"
                % (a.get("num_class"), a.get("weights_path")),
                "⚠️ 官方 panns.yaml 未发布 -> 超参为自定（见配置注释）"]


class _ClsMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.c = self.n = 0
        self.loss_sum = 0.0

    def __call__(self, logits, lab):
        if logits is None:
            return
        self.loss_sum += float(F.cross_entropy(logits, lab)) * lab.numel()
        self.c += int((logits.argmax(1) == lab).sum())
        self.n += int(lab.numel())

    def get_metric(self):
        n = max(self.n, 1)
        return {"acc": self.c / n, "cls_loss": self.loss_sum / n, "num": self.n}
