"""时序异常检测：损失、异常打分、后处理与指标（移植 paddlets ``models/anomaly``）。

包含：
  * :func:`smooth_l1_loss_vae`     —— paddlets ``utils.py::smooth_l1_loss_vae``（VAE 损失）
  * :func:`my_kl_loss` / :func:`series_prior_loss` / :func:`series_prior_energy`
                                    —— Anomaly Transformer 的关联差异（KL）与打分
  * :func:`result_adjust`          —— **点调整(point-adjust)约定**（TS 异常检测文献通用）
  * :class:`TSAnomalyLoss`         —— 按 ``kind`` 分派的训练损失（AE/VAE/USAD/MTAD-GAT/AT）
  * :class:`TSAnomalyMetric`       —— 点级 P/R/F1（含 point-adjust）、AUC-ROC、AUC-PR（AP）

⚠️ Anomaly Transformer 的打分在 paddlets 里对**批内**做 softmax（``axis=-1``），
分数会随 batch 组成变化；这里保留该口径（``at_score_mode='paddlets'``），
并提供 ``'pointwise'``（不做批内 softmax）供需要批间可比时使用。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["smooth_l1_loss_vae", "my_kl_loss", "series_prior_loss",
           "series_prior_energy", "result_adjust", "TSAnomalyLoss",
           "TSAnomalyMetric"]


# --------------------------------------------------------------------------- #
# paddlets utils 的忠实移植
# --------------------------------------------------------------------------- #
def my_kl_loss(p, q):
    """KL 散度（p、q 形状相同，返回 shape ``(H,)``：先对最后一维求和再对 dim=1 求均值）。"""
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def smooth_l1_loss_vae(output_tensor_list, kld_beta: float = 0.2):
    """paddlets ``utils.smooth_l1_loss_vae``：``smooth_l1(recon, obs, sum) + beta * KL``。"""
    recon, mu, logvar, obs = output_tensor_list
    recon_loss = F.smooth_l1_loss(recon, obs, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return recon_loss + kld_beta * kld


def series_prior_loss(output_list, x, criterion=None, win_size: int = 100, k: int = 3):
    """paddlets ``utils.series_prior_loss``（Anomaly Transformer 训练用）。

    返回 ``(loss1, loss2, for_loss_one)``：
      ``loss1 = rec - k*series``、``loss2 = rec + k*prior``
    """
    output, series, prior, _ = output_list
    criterion = criterion or nn.MSELoss()
    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        series_kl = prior[u] / torch.tile(torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1),
                                          (1, 1, 1, win_size))
        series_loss = series_loss + (torch.mean(my_kl_loss(series[u], series_kl.detach()))
                                     + torch.mean(my_kl_loss(series_kl.detach(), series[u])))
        prior_loss = prior_loss + (torch.mean(my_kl_loss(series_kl, series[u].detach()))
                                   + torch.mean(my_kl_loss(series[u].detach(), series_kl)))
    series_loss = series_loss / len(prior)
    prior_loss = prior_loss / len(prior)
    rec_loss = criterion(output, x)
    return rec_loss - k * series_loss, rec_loss + k * prior_loss, (rec_loss - k * series_loss).item()


def series_prior_energy(output_list, rec_loss, temperature: float = 50,
                        win_size: int = 100, mode: str = "paddlets"):
    """paddlets ``utils.series_prios_energy`` 的打分（Anomaly Transformer 推理用）。"""
    output, series, prior, _ = output_list
    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        series_kl = prior[u] / torch.tile(torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1),
                                          (1, 1, 1, win_size))
        if u == 0:
            series_loss = my_kl_loss(series[u], series_kl.detach()) * temperature
            prior_loss = my_kl_loss(series_kl, series[u].detach()) * temperature
        else:
            series_loss = series_loss + my_kl_loss(series[u], series_kl.detach()) * temperature
            prior_loss = prior_loss + my_kl_loss(series_kl, series[u].detach()) * temperature
    if mode == "paddlets":
        metric = F.softmax(-series_loss - prior_loss, dim=-1)   # ⚠️ 批内 softmax（paddlets 原样）
    else:
        metric = torch.sigmoid(-series_loss - prior_loss)        # 批间可比的替代口径
    return metric * rec_loss


def result_adjust(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    """点调整(point-adjust)：命中的真实异常段整体判为异常（TS 异常检测通用约定）。

    与 paddlets ``utils.result_adjust`` 逐行一致（原地修改并返回 ``pred``）。
    """
    pred = np.asarray(pred).copy()
    real = np.asarray(real)
    anomaly_state = False
    for i in range(len(real)):
        if real[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if real[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
            for j in range(i, len(real)):
                if real[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
        elif real[i] == 0:
            anomaly_state = False
        if not anomaly_state:
            pred[i] = 0
    return pred


# --------------------------------------------------------------------------- #
# 训练损失
# --------------------------------------------------------------------------- #
class TSAnomalyLoss(nn.Module):
    """按 ``kind`` 分派的异常检测训练损失。

    * ``ae``        : ``MSE(recon, x)``（paddlets 基类 ``_compute_loss``）
    * ``vae``       : ``smooth_l1_loss_vae``（recon+KL，``kld_beta=0.2``）
    * ``usad``      : 两阶段对抗（**需要 trainer 的 ``train_step`` 钩子**）
    * ``mtad_gat``  : ``sqrt(MSE(y,pred)) + sqrt(MSE(x,recon))``
    * ``anomaly_transformer`` : ``series_prior_loss``（两阶段，**需要钩子**）
    """

    TWO_PHASE = ("usad", "anomaly_transformer")

    def __init__(self, kind: str = "ae", in_chunk_len: int = 100,
                 k: int = 3, kld_beta: float = 0.2, alpha: float = 0.5,
                 beta: float = 0.5, target_dims=None, **kw):
        super().__init__()
        self.kind = kind
        self.in_chunk_len = in_chunk_len
        self.k = k
        self.kld_beta = kld_beta
        self.alpha = alpha
        self.beta = beta
        self.target_dims = target_dims
        self._lr = None
        self._opt1 = None
        self._opt2 = None
        # ⚠️ 只有两阶段模型才**挂载** train_step；否则 BaseTrainer 会走钩子路径（单阶段模型不该走）
        if kind in self.TWO_PHASE:
            self.train_step = self._train_step_impl

    # ---- helpers ----
    @staticmethod
    def _mtad_split(batch):
        obs = batch[0]
        L = obs.shape[1]
        return obs[:, :L - 1, :], obs[:, L - 1:, :]

    def forward(self, preds, batch):
        k = self.kind
        # 两阶段模型（USAD / AT）的真实更新在 `train_step` 里；这里只回一个占位 loss
        if k in self.TWO_PHASE and preds is None:
            dev = batch[0].device if torch.is_tensor(batch[0]) else "cpu"
            z = torch.zeros((), device=dev)
            return {"loss": z, "rec_loss": z}
        if k == "ae":
            recon, x = preds
            loss = F.mse_loss(recon, x)
            return {"loss": loss, "rec_loss": loss.detach()}
        if k == "vae":
            loss = smooth_l1_loss_vae(preds, self.kld_beta)
            return {"loss": loss, "rec_loss": loss.detach()}
        if k == "usad":
            x, w1, _, _ = preds
            with torch.no_grad():
                l1 = F.mse_loss(w1, x)
            return {"loss": l1, "rec_loss": l1}
        if k == "mtad_gat":
            preds_, recons = preds
            x, y = self._mtad_split(batch)
            p = preds_.squeeze(1) if preds_.ndim == 3 else preds_
            yy = y.squeeze(1) if y.ndim == 3 else y
            fl = torch.sqrt(F.mse_loss(yy, p))
            rl = torch.sqrt(F.mse_loss(x, recons))
            return {"loss": fl + rl, "fore_loss": fl.detach(), "rec_loss": rl.detach()}
        if k == "anomaly_transformer":
            out = preds[0]
            x = batch[0]
            with torch.no_grad():
                l1, _, _ = series_prior_loss([out, preds[1], preds[2], None], x,
                                             nn.MSELoss(), self.in_chunk_len, self.k)
            return {"loss": l1, "rec_loss": l1}
        raise ValueError("unknown anomaly kind: %s" % k)

    # ---- USAD / AT 的两阶段更新（由 BaseTrainer 的钩子调用）----
    def _train_step_impl(self, model, loss_dict, batch, step_idx=1, **kw):
        if self.kind == "usad":
            return self._usad_step(model, batch, step_idx)
        if self.kind == "anomaly_transformer":
            return self._at_step(model, loss_dict, batch)
        raise RuntimeError("train_step 只用于两阶段模型")

    def _usad_step(self, model, batch, batch_idx):
        net = model
        X = {"observed_cov_numeric": batch[0]}
        opt1, opt2 = self._ensure_usad_opts(net)

        x, w1, _, w3 = net(X)
        loss1 = (1.0 / batch_idx) * F.mse_loss(w1, x) + (1 - 1.0 / batch_idx) * F.mse_loss(w3, x)
        loss1.backward()
        opt1.step()
        net.zero_grad(set_to_none=True)

        x, _, w2, w3 = net(X)
        loss2 = (1.0 / batch_idx) * F.mse_loss(w2, x) - (1 - 1.0 / batch_idx) * F.mse_loss(w3, x)
        loss2.backward()
        opt2.step()
        net.zero_grad(set_to_none=True)
        return {"loss": torch.tensor(float((loss1 + loss2).item()),
                                     device=batch[0].device)}

    def _ensure_usad_opts(self, net):
        """opt1 = encoder + decoder1；opt2 = encoder + decoder2（与 paddlets 一致）。"""
        if self._opt1 is None:
            p1 = [p for n, p in net.named_parameters()
                  if n.startswith("_encoder.") or n.startswith("_decoder1.")]
            self._opt1 = torch.optim.Adam(p1, lr=self._lr or 1e-3, betas=(0.9, 0.999))
        if self._opt2 is None:
            p2 = [p for n, p in net.named_parameters()
                  if n.startswith("_encoder.") or n.startswith("_decoder2.")]
            self._opt2 = torch.optim.Adam(p2, lr=self._lr or 1e-3, betas=(0.9, 0.999))
        return self._opt1, self._opt2

    def _at_step(self, model, loss_dict, batch):
        net = model
        x = batch[0]
        if self._opt1 is None:
            self._opt1 = torch.optim.Adam(net.parameters(), lr=self._lr or 1e-4,
                                          betas=(0.9, 0.999))
        out, series, prior, _ = net({"observed_cov_numeric": x})
        l1, l2, _ = series_prior_loss([out, series, prior, None], x, nn.MSELoss(),
                                      self.in_chunk_len, self.k)
        l1.backward()
        self._opt1.step()
        net.zero_grad(set_to_none=True)

        out, series, prior, _ = net({"observed_cov_numeric": x})
        l1, l2, _ = series_prior_loss([out, series, prior, None], x, nn.MSELoss(),
                                      self.in_chunk_len, self.k)
        l2.backward()
        self._opt1.step()
        net.zero_grad(set_to_none=True)
        return {"loss": torch.tensor(float((l1 + l2).item()), device=x.device)}

    def set_lr(self, lr):
        self._lr = lr


# --------------------------------------------------------------------------- #
# 异常打分（与 paddlets 各模型 `_predict` 口径一致）
# --------------------------------------------------------------------------- #
def anomaly_score(kind: str, preds, batch, in_chunk_len: int,
                  alpha: float = 0.5, beta: float = 0.5, flatten: bool = False,
                  at_temperature: float = 50.0, at_score_mode: str = "paddlets",
                  target_dims=None):
    """返回**每窗口一个**异常分数（``np.ndarray``, shape ``(B,)``）。"""
    with torch.no_grad():
        if kind in ("ae", "vae"):
            recon, x = preds[0], preds[-1]
            # ⚠️ 统一降到 **(B,)**：`dim=tuple(range(1, ndim))`
            dims = tuple(range(1, recon.ndim))
            return torch.mean((recon - x) ** 2, dim=dims).cpu().numpy()
        if kind == "usad":
            x, w1, _, w3 = preds
            dims = tuple(range(1, x.ndim))
            l1 = torch.mean((w1 - x) ** 2, dim=dims)
            l3 = torch.mean((w3 - x) ** 2, dim=dims)
            return (alpha * l1 + beta * l3).cpu().numpy()
        if kind == "mtad_gat":
            pred_, recon = preds
            obs = batch[0]
            L = obs.shape[1]
            y = obs[:, L - 1:, :]
            p = pred_.squeeze(1) if pred_.ndim == 3 else pred_      # (B, C)
            yy = y.squeeze(1) if y.ndim == 3 else y                 # (B, C)
            r = recon[:, -1, :]                                     # (B, C)
            if target_dims is not None:
                p = p[:, target_dims]
                yy = yy[:, target_dims]
                r = r[:, target_dims]
            s = torch.mean(torch.sqrt((p - yy) ** 2) + torch.sqrt((r - yy) ** 2), dim=-1)
            return s.cpu().numpy()
        if kind == "anomaly_transformer":
            out, series, prior, _ = preds
            x = batch[0]
            rec = F.mse_loss(out, x, reduction="none").mean()
            s = series_prior_energy([out, series, prior, None], rec,
                                    at_temperature, in_chunk_len, at_score_mode)
            return s.cpu().numpy()
    raise ValueError("unknown anomaly kind: %s" % kind)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _roc_auc(y, s):
    """ROC-AUC（Mann-Whitney U，含并列处理）。"""
    y = np.asarray(y).astype(np.float64)
    s = np.asarray(s).astype(np.float64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y, s):
    """PR-AUC（average precision，阶梯式）。"""
    y = np.asarray(y).astype(np.float64)
    s = np.asarray(s).astype(np.float64)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / y.sum()
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def _f1_at(y, pred):
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


class TSAnomalyMetric:
    """点级异常检测指标。

    * ``threshold_percentile``：按**分数分位数**定阈值（label-free，默认 99.0 = 取 top 1%）
    * ``point_adjust``：是否套用 point-adjust（TS 异常检测文献通用，默认 True）
    * 另报 ``best_f1``（遍历阈值取最优，**属"oracle"参考值**，不作为主指标）
    """

    def __init__(self, main_indicator="f1", threshold_percentile: float = 99.0,
                 point_adjust: bool = True, **kw):
        self.main_indicator = main_indicator
        self.q = float(threshold_percentile)
        self.point_adjust = point_adjust
        self.reset()

    def reset(self):
        self.scores = []
        self.labels = []

    def __call__(self, scores, labels):
        if scores is None:
            return
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        self.scores.append(np.asarray(scores).reshape(-1))
        self.labels.append(np.asarray(labels).reshape(-1))

    def get_metric(self):
        out = {"num": 0}
        if not self.scores:
            return out
        s = np.concatenate(self.scores).reshape(-1)
        y = np.concatenate(self.labels).reshape(-1).astype(np.int64)
        if len(s) != len(y):          # 防御：长度不一致时按较短截断（不应发生）
            n = min(len(s), len(y))
            s, y = s[:n], y[:n]
        out["num"] = int(len(s))
        out["auc_roc"] = _roc_auc(y, s)
        out["auc_pr"] = _average_precision(y, s)
        out["anomaly_ratio"] = float(y.mean()) if len(y) else 0.0

        thr = float(np.percentile(s, self.q))
        pred = (s > thr).astype(np.int64)
        yy = y
        if self.point_adjust:
            pred = result_adjust(pred, y)
        p, r, f1 = _f1_at(yy, pred)
        out.update({"precision": p, "recall": r, "f1": f1, "threshold": thr})

        # oracle 参考：遍历候选阈值取 best-F1（**不作为主指标**）
        best_f1 = 0.0
        for t in np.percentile(s, np.arange(50, 100.0001, 0.5)):
            pr = (s > float(t)).astype(np.int64)
            if self.point_adjust:
                pr = result_adjust(pr, y)
            _, _, f = _f1_at(y, pr)
            if f > best_f1:
                best_f1 = f
        out["best_f1"] = best_f1
        out[self.main_indicator] = out.get(self.main_indicator, out["f1"])
        return out
