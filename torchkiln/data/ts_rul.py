"""RUL（剩余寿命）数据集：run-to-failure 多轨迹切分 + 分段线性 RUL 标签。

支持两种格式：
  1. **C-MAPSS**（NASA 涡扇发动机退化，空格分隔、无表头）：
     列 = ``unit, cycle, op1..op3, s1..s21``
     * ``train_*.txt``：完整 run-to-failure 轨迹 → **在每条轨迹内**滑窗，
       RUL 标签 = ``min(MAX_RUL, 该轨迹末尾 - 当前 cycle)``（**分段线性**，工业界标准做法）
     * ``test_*.txt`` + ``RUL_*.txt``：测试轨迹被**截断**，只在**每条轨迹的最后一步**
       取一个样本，RUL 真值来自 ``RUL_*.txt``（每行一台）
  2. **通用长格式 CSV**：``unit, cycle, f1..fd, [rul]``
     * 有 ``rul`` 列则直接用；无则按 ``unit`` 内 ``max(cycle) - cycle`` 推（并 ``MAX_RUL`` 截断）

⚠️ **划分按 unit 分**（不是按时间切），否则同一台设备的相邻窗口会同时出现在
train/test → **数据泄漏**。``split_ratio`` 只作用于 **unit 列表**。
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["RULDataset", "rul_collate"]


def _read_cmapss(path):
    """返回 ``(unit, cycle, feats)``；feats 形状 ``(N, 24)``（3 操作 + 21 传感器）。"""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    a = np.asarray(rows, dtype=np.float64)
    return a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2:]


def _read_long_csv(path, rul_col=None):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    body = np.asarray([[float(x) if x != "" else np.nan for x in r] for r in rows[1:]],
                      dtype=np.float64)
    cols = {h: i for i, h in enumerate(header)}
    unit = body[:, cols["unit"]].astype(np.int64)
    cycle = body[:, cols["cycle"]].astype(np.int64)
    if rul_col and rul_col in cols:
        rul = body[:, cols[rul_col]]
        drop = {cols["unit"], cols["cycle"], cols[rul_col]}
    else:
        rul = None
        drop = {cols["unit"], cols["cycle"]}
    feats = body[:, [i for i in range(len(header)) if i not in drop]]
    return unit, cycle, feats, rul


class RULDataset(Dataset):
    """RUL 数据集（``mode='Train'`` / ``'Eval'``）。

    样本 = ``[feat(L, D), rul(标量)]``。
    """

    def __init__(self, config, mode="Train", logger=None):
        ds = (config.get(mode) or {}).get("dataset") or {}
        self.window = int(ds.get("window", ds.get("in_chunk_len", 30)))
        self.stride = int(ds.get("stride", 1))
        self.max_rul = float(ds.get("max_rul", 125))
        kind = str(ds.get("format", "cmapss")).lower()
        sensor_cols = ds.get("sensor_cols")

        if kind == "cmapss":
            data_dir = ds.get("data_dir", "datasets/cmapss")
            if mode == "Train":
                unit, cycle, feats = _read_cmapss(os.path.join(data_dir, ds["train_file"]))
                rul = None
            else:
                unit, cycle, feats = _read_cmapss(os.path.join(data_dir, ds["test_file"]))
                rul = np.loadtxt(os.path.join(data_dir, ds["rul_file"]), dtype=np.float64)
        else:
            p = ds["csv_path"]
            if not os.path.isabs(p) and ds.get("data_dir"):
                p = os.path.join(ds["data_dir"], p)
            unit, cycle, feats, rul = _read_long_csv(p, ds.get("rul_col", "rul"))

        if sensor_cols:
            idx = [int(i) for i in sensor_cols]
            feats = feats[:, idx]
        elif ds.get("auto_select", True):
            # ⚠️ C-MAPSS 的 24 列里很多是**常量 / 无信息**传感器（std≈0）——直接全喂会
            #    引入噪声、RUL 学不好。按「训练段 unit 内的标准差」自动筛掉常量列
            #    （业界标准做法：只保留有退化趋势的传感器）。
            units_all = sorted(set(unit.tolist()))
            n_tr = max(1, int(len(units_all) * float(
                (ds.get("split_ratio") or [0.7, 0.15, 0.15])[0])))
            m_tr = np.isin(unit, units_all[:n_tr])
            sd_col = np.nanstd(feats[m_tr], axis=0)
            keep = np.where(sd_col > float(ds.get("min_std", 1e-6)))[0]
            if len(keep) == 0:
                keep = np.arange(feats.shape[1])
            feats = feats[:, keep]
            if logger is not None:
                logger.info("RUL auto_select: %d/%d 列保留（std>%s）",
                            len(keep), feats.shape[1] + (feats.shape[1] - len(keep)),
                            ds.get("min_std", 1e-6))

        # ⭐ **特征工程**（业界提精度的关键；默认关闭以保持与 paddlets 口径一致）
        #    feat_eng 可选:
        #      * `diff`     : 一阶差分（退化速率）
        #      * `roll_mean`: 滚动均值（平滑噪声，窗口 = roll_window）
        #      * `roll_std` : 滚动标准差（波动增大常预示故障）
        #      * `slope`    : 窗口线性斜率（退化趋势强度）
        #      * `delta0`   : 相对**轨迹起点**的偏离（累积退化量）
        #    ⚠️ 全部**逐 unit** 计算，避免跨设备串扰。
        fe = ds.get("feat_eng") or []
        if fe:
            rw = int(ds.get("roll_window", 10))
            parts = [feats]
            # ⚠️ 循环变量**不能叫 kind** —— 外层 `kind` 是数据格式（cmapss/csv），
            #    遮蔽后导致后面 `kind == "cmapss"` 恒假（曾把 test 的 100 台变成 1428 样本）
            for fk in fe:
                extra = np.zeros_like(feats)
                for uu in sorted(set(unit.tolist())):
                    m = unit == uu
                    sub = feats[m]
                    if fk == "diff":
                        d = np.diff(sub, axis=0)
                        extra[m] = np.vstack([np.zeros((1, sub.shape[1])), d])
                    elif fk == "roll_mean":
                        cs = np.cumsum(np.vstack([np.zeros((1, sub.shape[1])), sub]), axis=0)
                        for i in range(len(sub)):
                            a = max(0, i - rw + 1)
                            extra[m][i] = (cs[i + 1] - cs[a]) / (i + 1 - a)
                    elif fk == "roll_std":
                        for i in range(len(sub)):
                            a = max(0, i - rw + 1)
                            extra[m][i] = np.std(sub[a:i + 1], axis=0)
                    elif fk == "slope":
                        for i in range(len(sub)):
                            a = max(0, i - rw + 1)
                            seg = sub[a:i + 1]
                            if len(seg) < 2:
                                continue
                            ys = seg[-rw:] if len(seg) >= rw else seg
                            xx = np.arange(len(ys), dtype=np.float64)
                            extra[m][i] = ((xx - xx.mean())[:, None] * (ys - ys.mean())).sum(0) / (
                                ((xx - xx.mean()) ** 2).sum() or 1.0)
                    elif fk == "delta0":
                        extra[m] = sub - sub[:1]
                parts.append(extra)
            feats = np.concatenate(parts, axis=1)
            if logger is not None:
                logger.info("RUL feat_eng=%s: 特征维 %d -> %d", fe, parts[0].shape[1], feats.shape[1])

        self.mu, self.sd = None, None
        self.op_centroids = None
        if ds.get("standardize", True):
            # ⚠️ 统计量只用**训练段 unit**（避免测试信息泄漏）
            units_all = sorted(set(unit.tolist()))
            n_tr = max(1, int(len(units_all) * float(
                (ds.get("split_ratio") or [0.7, 0.15, 0.15])[0])))
            tr_units = set(units_all[:n_tr])
            mask = np.isin(unit, list(tr_units))
            if not mask.any():
                mask = np.ones(len(unit), bool)

            # ⭐ **按工况（operating condition）分组归一化** —— FD002/004 有 6 种工况，
            #    同一传感器在不同工况下均值可差数十倍；若全局 z-score，工况差异会被
            #    当成"退化信号"污染 RUL 学习（实测全局归一化时 RMSE 明显更差）。
            #    做法：对操作条件列（前 `n_op_cols` 列）做**取整聚类** -> 每组独立 z-score。
            n_op = int(ds.get("n_op_cols", 3))
            op_group = None
            if n_op > 0 and feats.shape[1] >= n_op:
                op = np.round(feats[:, :n_op], int(ds.get("op_round", 2)))
                uniq, inv = np.unique(op[mask], axis=0, return_inverse=True)
                # 只把出现频率 >= min_op_frac 的组合当独立工况，其余归到最近组
                cnt = np.bincount(inv)
                keep_g = np.where(cnt >= float(ds.get("min_op_frac", 0.02)) * len(inv))[0]
                if len(keep_g) > 1:
                    cent = uniq[keep_g]
                    self.op_centroids = cent
                    # 全部点按最近质心分组
                    d2 = ((op[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
                    op_group = d2.argmin(axis=1)
                    if logger is not None:
                        logger.info("RUL 工况聚类: %d 种（op 列 %d）", len(cent), n_op)

            if op_group is not None:
                mu = np.zeros(feats.shape[1])
                sd = np.ones(feats.shape[1])
                for g in np.unique(op_group):
                    gm = (op_group == g) & mask
                    if gm.sum() < 2:
                        gm = op_group == g
                    mu_g = np.nanmean(feats[gm], axis=0)
                    sd_g = np.nanstd(feats[gm], axis=0)
                    sd_g = np.where(sd_g < 1e-8, 1.0, sd_g)
                    sel = op_group == g
                    feats[sel] = (feats[sel] - mu_g) / sd_g
                self.mu, self.sd = mu, sd
            else:
                self.mu = np.nanmean(feats[mask], axis=0)
                self.sd = np.nanstd(feats[mask], axis=0)
                self.sd = np.where(self.sd < 1e-8, 1.0, self.sd)
                feats = (feats - self.mu) / self.sd

        # 按 unit 划分（**禁止按时间切**，否则泄漏）
        units = sorted(set(unit.tolist()))
        r = list(ds.get("split_ratio", [0.7, 0.15, 0.15]))
        a = int(len(units) * r[0])
        b = int(len(units) * (r[0] + (r[1] if len(r) > 1 else 0)))
        if kind == "cmapss" and mode != "Train":
            keep = units                       # C-MAPSS 的 test 文件本身就是另一批 unit
        else:
            keep = units[:a] if mode == "Train" else units[a:b]
        keep = set(keep)

        self.feat, self.target = [], []
        # ⚠️ C-MAPSS 经典陷阱：训练若用**全部窗口**，早期健康段（RUL 恒为 125）占大多数，
        #    模型退化成"输出均值"（常数基线 RMSE≈40）。
        #    标准做法：`train_last_frac` 只保留每条轨迹**后半段**的窗口，
        #    让训练分布贴近评估口径（test 只取轨迹末尾）。
        last_frac = float(ds.get("train_last_frac", 1.0))
        for u in sorted(keep):
            m = unit == u
            f, c = feats[m], cycle[m]
            o = np.argsort(c)
            f, c = f[o], c[o]
            if kind == "cmapss" and mode != "Train":
                # 测试：只在轨迹**最后一步**取一个样本（对齐 C-MAPSS 官方评测口径）
                if len(f) < self.window:
                    pad = np.repeat(f[:1], self.window - len(f), axis=0)
                    f = np.concatenate([pad, f], axis=0)
                y = float(rul[u - 1]) if rul is not None else 0.0
                self.feat.append(np.nan_to_num(f[-self.window:], nan=0.0).astype(np.float32))
                self.target.append(np.float32(min(y, self.max_rul)))
                continue
            # ⚠️ `rul` 在 C-MPASS test 上是**每台一个值**（长度 = unit 数），
            #    train 上为 None（用 cycle 推算）；通用 CSV 则是逐行长度。
            #    `rul_per_unit`: 长度等于 unit 数 -> test 的每台真值
            rul_per_unit = (rul is not None and len(rul) == len(np.unique(unit)))
            if rul_per_unit or rul is None:
                rul_u = None                              # 用 cycle 差推算
            else:
                rul_u = rul[m][o]                         # 逐行真值（通用 CSV）
            start = self.window - 1
            if mode == "Train" and 0 < last_frac < 1.0:
                n_keep = max(1, int(len(f) * last_frac))
                start = max(start, len(f) - n_keep)
            for i in range(start, len(f), self.stride):
                # ⚠️ rul_u 可能是 None（C-MAPSS test：每台一个值）-> 用 cycle 差推算
                y = (c[-1] - c[i]) if rul_u is None else rul_u[i]
                self.feat.append(np.nan_to_num(f[i - self.window + 1:i + 1], nan=0.0).astype(np.float32))
                self.target.append(np.float32(min(float(y), self.max_rul)))
        self.dim = feats.shape[1]
        if logger is not None:
            logger.info("%s RUL dataset: %d samples, units=%d, L=%d D=%d max_rul=%s",
                        mode, len(self.feat), len(keep), self.window, self.dim, self.max_rul)

    def __len__(self):
        return len(self.feat)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        return [self.feat[index], np.float32(self.target[index])]


def rul_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return []
    return [torch.from_numpy(np.stack([b[0] for b in batch], 0)),
            torch.from_numpy(np.stack([b[1] for b in batch], 0))]
