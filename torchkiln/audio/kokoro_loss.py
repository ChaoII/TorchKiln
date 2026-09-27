"""kokoro 训练损失（自写）—— 多任务 TTS 重建。

⚠️ **官方未公开训练代码**（hexgrad/kokoro 只有推理），故本文件的 loss 组合为**自定**，
   参考 StyleTTS2/Kokoro 的常规做法与第三方 `jonirajala/kokoro_training` 的思路，
   **不能声称与官方对齐**（③ 已改用「模块级梯度通路 vs 官方推理包」来验收）。

四个分量：
  * ``mel``      : 生成波形 vs 目标波形的 **mel 重建 L1**（用 torchaudio 的 mel，可微）
  * ``f0``       : ``F0_pred`` vs 由目标 F0 曲线下采样得到的目标，L1
  * ``energy``   : ``N_pred`` vs 目标能量（目标波形 RMS 包络），L1
  * ``duration`` : 预测时长分布 vs 目标时长（交叉熵/Dice 风格，用 reals 的 ``max_dur`` 类）

注意：
  * ``pred_dur = round(...)`` **不可微** ⇒ 时长监督要作用在 ``duration`` 的 **sigmoid 分布**上
    （SDP/对齐思想），这里用「目标时长 one-hot 的 soft 交叉熵」。
  * **整链一次反传会触发原生段错误**（见 AGENTS）：训练时**必须按模块分段反传**
    （``mel/f0/energy`` 走 predictor+decoder，``duration`` 只走 predictor/bert）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KokoroLoss", "mel_spectrogram"]


def mel_spectrogram(wav: torch.Tensor, sr: int = 24000, n_fft: int = 1024,
                    hop_length: int = 256, win_length: int = 1024,
                    n_mels: int = 80, f_min: float = 0.0,
                    f_max: float = None) -> torch.Tensor:
    """可微 mel（log 域）。wav: (B, L) 或 (L,)。返回 (B, n_mels, T)。"""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if f_max is None:
        f_max = sr / 2.0
    mel = torchaudio_mel(wav, sr, n_fft, hop_length, win_length, n_mels,
                         f_min, f_max)
    return torch.log(mel.clamp(min=1e-5))


def torchaudio_mel(wav, sr, n_fft, hop_length, win_length, n_mels, f_min, f_max):
    """惰性构造 MelSpectrogram（**按设备缓存**，避免 CPU/CUDA 混用报
    ``stft input and window must be on the same device``）。"""
    import torchaudio
    if not hasattr(torchaudio_mel, "_cache"):
        torchaudio_mel._cache = {}
    cache = torchaudio_mel._cache
    dev = wav.device
    key = (sr, n_fft, hop_length, win_length, n_mels, f_min, f_max, str(dev))
    if key not in cache:
        m = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels, f_min=f_min, f_max=f_max,
            power=2.0)
        m = m.to(dev)          # ← 关键: window buffer 随设备
        m.eval()
        cache[key] = m
    return cache[key](wav).to(wav.dtype)


def _f0_from_wav(wav: torch.Tensor, sr: int = 24000, hop: int = 300,
                 fmin: float = 80.0, fmax: float = 400.0) -> torch.Tensor:
    """从目标波形估 F0（自相关法，逐帧）。仅作监督目标，不需可微。"""
    B, L = wav.shape
    n = max(1, L // hop)
    out = torch.zeros(B, n, device=wav.device, dtype=torch.float32)
    lo = max(2, int(sr / fmax))
    hi = max(lo + 2, int(sr / fmin))
    for b in range(B):
        for i in range(n):
            s = i * hop
            seg = wav[b, s:s + hi * 2]
            if seg.numel() < hi * 2:
                seg = F.pad(seg, (0, hi * 2 - seg.numel()))
            seg = seg - seg.mean()
            e = (seg * seg).sum()
            if e <= 1e-8:
                continue
            ac = torch.fft.irfft(torch.fft.rfft(seg, 2 * hi * 2) *
                                 torch.fft.rfft(seg, 2 * hi * 2).conj())[:hi]
            if lo >= ac.numel():
                continue
            seg_ac = ac[lo:hi]
            k = int(seg_ac.argmax().item())
            lag = lo + k
            if lag > 0:
                out[b, i] = sr / lag
    return out


class KokoroLoss(nn.Module):
    """kokoro 多任务训练损失（自定，见文件头说明）。"""

    def __init__(self, sr: int = 24000, n_fft: int = 1024,
                 hop_length: int = 256, n_mels: int = 80,
                 w_mel: float = 1.0, w_f0: float = 1.0, w_energy: float = 1.0,
                 w_duration: float = 1.0, max_dur: int = 50):
        super().__init__()
        self.sr, self.n_fft, self.hop_length = sr, n_fft, hop_length
        self.n_mels = n_mels
        self.w_mel, self.w_f0 = w_mel, w_f0
        self.w_energy, self.w_duration = w_energy, w_duration
        self.max_dur = max_dur

    def mel_loss(self, wav_pred: torch.Tensor, wav_tgt: torch.Tensor):
        a = mel_spectrogram(wav_pred, self.sr, self.n_fft, self.hop_length,
                            self.n_fft, self.n_mels)
        b = mel_spectrogram(wav_tgt, self.sr, self.n_fft, self.hop_length,
                            self.n_fft, self.n_mels)
        T = min(a.shape[-1], b.shape[-1])
        return F.l1_loss(a[..., :T], b[..., :T])

    def f0_energy_loss(self, wav_tgt: torch.Tensor, F0_pred, N_pred):
        """F0/能量监督。

        注意长度口径：``F0_pred`` 来自 decoder 前的对齐结果（上采样后），
        与从原始波形估的 F0 帧数**不一致**（约为 1:10）。
        这里用 ``interpolate`` 把预测下采样到目标帧数（自定口径，见文件头）。
        """
        tgt = _f0_from_wav(wav_tgt, self.sr, hop=300)          # (B, n_tgt)
        n = tgt.shape[-1]
        f0 = F.interpolate(F0_pred.unsqueeze(1), size=n,
                           mode="linear", align_corners=False).squeeze(1)
        l_f0 = F.l1_loss(f0, tgt)
        # 能量目标：波形 RMS 包络，同样插值到同一帧数
        hop = max(1, wav_tgt.shape[-1] // n)
        rms = wav_tgt.unfold(1, hop * 2, hop)[:, :n].pow(2).mean(-1).sqrt()
        m = min(rms.shape[-1], n)
        n_pred = F.interpolate(N_pred.unsqueeze(1), size=m,
                               mode="linear", align_corners=False).squeeze(1)
        l_n = F.l1_loss(n_pred, rms[..., :m])
        return l_f0, l_n

    def duration_loss(self, duration_logits: torch.Tensor,
                      pred_dur: torch.Tensor):
        """时长监督：把 ``pred_dur`` 当目标，对 ``sigmoid(duration_logits)`` 的和做 L1。

        说明：官方 ``duration_proj`` 输出 (B,T,max_dur) 的**逐帧时长分布**，其 sigmoid 和对
        时间求和得到总时长。这里用「预测总时长 vs 目标总时长」的 L1（自定），
        避开了 ``round`` 不可微的问题。
        """
        s = torch.sigmoid(duration_logits).sum(dim=-1)      # (B,T)
        pred_total = s.sum(dim=-1)                          # (B,)
        tgt_total = pred_dur.float().sum().expand_as(pred_total)
        return F.l1_loss(pred_total, tgt_total)

    def forward(self, out: dict, ref):
        """out: 模型输出; ref: trainer 传的 ``labels``。

        ⚠️ trainer 传的是 **list** ``[input_ids, ref_s, wav]``（不是 dict）——
        故这里兼容两种：list/tuple 时取 ``ref[2]`` 作目标波形。
        """
        wav_pred = out["audio"]
        if isinstance(ref, (list, tuple)):
            wav_tgt = ref[2]
        else:
            wav_tgt = ref["wav"]
        l_mel = self.mel_loss(wav_pred, wav_tgt)
        l_f0, l_n = self.f0_energy_loss(wav_tgt, out["F0"], out["N"])
        l_dur = self.duration_loss(out["duration_logits"], out["pred_dur"])
        total = (self.w_mel * l_mel + self.w_f0 * l_f0
                 + self.w_energy * l_n + self.w_duration * l_dur)
        return {
            "loss": total,
            "mel": l_mel.detach(),
            "f0": l_f0.detach(),
            "energy": l_n.detach(),
            "duration": l_dur.detach(),
        }
