"""CenterPoint-Pillars (SOTA LiDAR 3D detector) — PyTorch port of Paddle3D.

Mirrors ``PaddlePaddle/Paddle3D`` module names exactly so the official
``.pdparams`` weights can be loaded 1:1 (missing=0 / unexpected=0)::

    voxel_encoder (PillarFeatureNet)  pfn_layers.{i}.linear/.norm
    backbone (SecondBackbone)         blocks.{i}.{conv,bn}
    neck (SecondFPN)                  deblocks.{i}.{conv, bn}
    bbox_head (CenterHead)            shared_conv.{conv,bn}, tasks.{k}.{reg,height,dim,rot,hm}

Input : raw points (N, 4) [x, y, z, intensity] in the LiDAR frame
Output: per-task dict with ``hm/reg/height/dim/rot`` at 1/down_ratio resolution.
"""
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PillarFeatureNet",
    "PointPillarsScatter",
    "SecondBackbone",
    "SecondFPN",
    "CenterHead",
    "CenterPointPillars",
    "hard_voxelize",
    "load_paddle_centerpoint",
]


# --------------------------------------------------------------------------- #
# voxelization
# --------------------------------------------------------------------------- #
def hard_voxelize(points, point_cloud_range, voxel_size, max_points=100,
                  max_voxels=40000):
    """Hard-voxelize one sample -> (voxels, coords, num_points).

    ``voxels``: (P, max_points, C) float32 (zero padded),
    ``coords``: (P, 4) int32 [batch=0, z, y, x], ``num_points``: (P,) int32.
    """
    pc_range = np.asarray(point_cloud_range, dtype=np.float32)
    vs = np.asarray(voxel_size, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    c = pts.shape[1]
    keep = (
        (pts[:, 0] >= pc_range[0]) & (pts[:, 0] < pc_range[3])
        & (pts[:, 1] >= pc_range[1]) & (pts[:, 1] < pc_range[4])
        & (pts[:, 2] >= pc_range[2]) & (pts[:, 2] < pc_range[5])
    )
    pts = pts[keep]
    if pts.shape[0] == 0:
        return (np.zeros((0, max_points, c), np.float32),
                np.zeros((0, 4), np.int32), np.zeros((0,), np.int32))

    gx, gy, gz = np.round((pc_range[3:] - pc_range[:3]) / vs).astype(np.int64)
    xyz = np.floor((pts[:, :3] - pc_range[:3]) / vs).astype(np.int32)
    xyz = np.clip(xyz, 0, np.array([gx - 1, gy - 1, gz - 1], np.int32))

    lin = (xyz[:, 0].astype(np.int64) * gy + xyz[:, 1]) * gz + xyz[:, 2]
    uniq, inv = np.unique(lin, return_inverse=True)
    nvox = uniq.shape[0]
    if nvox > max_voxels:  # keep the densest pillars (most points)
        order_v = np.argsort(-np.bincount(inv, minlength=nvox))[:max_voxels]
        remap = -np.ones(nvox, np.int64)
        remap[order_v] = np.arange(max_voxels)
        inv = remap[inv]
        sel = inv >= 0
        pts, inv = pts[sel], inv[sel]
        nvox = max_voxels

    counts = np.bincount(inv, minlength=nvox)
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    pts_s = pts[order]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    voxels = np.zeros((nvox, max_points, c), np.float32)
    num_points = np.minimum(counts, max_points).astype(np.int32)
    offsets = np.arange(pts.shape[0]) - np.repeat(starts, counts)
    sel = offsets < max_points
    voxels[inv_s[sel], offsets[sel]] = pts_s[sel]

    coords = np.zeros((nvox, 4), np.int32)
    first_xyz = xyz[order[starts]]  # (nvox,3) [x,y,z]
    coords[:, 0] = 0
    coords[:, 1] = first_xyz[:, 2]  # z
    coords[:, 2] = first_xyz[:, 1]  # y
    coords[:, 3] = first_xyz[:, 0]  # x
    return voxels, coords, num_points


# --------------------------------------------------------------------------- #
# PillarFeatureNet
# --------------------------------------------------------------------------- #
def _pad_indicator(num_points, max_points):
    rng = torch.arange(max_points, device=num_points.device)
    return (rng[None, :] < num_points[:, None]).float()


class _PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, max_points=100, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        if not last_layer:
            out_channels = out_channels // 2
        self.units = out_channels
        self.max_num_points_in_voxel = max_points
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.linear(x)  # (N, M, C)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x_max = x.max(dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        return torch.cat([x, x_max.expand_as(x)], dim=2)


class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=4, feat_channels=(64, 64), with_distance=False,
                 max_num_points_in_voxel=100, voxel_size=(0.16, 0.16, 4.0),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1), legacy=False):
        super().__init__()
        self.legacy = legacy
        self.with_distance = with_distance
        inc = in_channels + 3 + 2 + (1 if with_distance else 0)
        channels = [inc] + list(feat_channels)
        layers = []
        for i in range(len(channels) - 1):
            layers.append(_PFNLayer(
                channels[i], channels[i + 1],
                max_points=max_num_points_in_voxel,
                last_layer=(i == len(channels) - 2)))
        self.pfn_layers = nn.ModuleList(layers)
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range
        self.max_num_points_in_voxel = max_num_points_in_voxel

    def forward(self, features, num_points_per_voxel, coors):
        features_ls = [features]
        features_sum = features[:, :, :3].sum(dim=1, keepdim=True)
        points_mean = features_sum / num_points_per_voxel.to(features.dtype).reshape(-1, 1, 1)
        features_ls.append(features[:, :, :3] - points_mean)
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(features.dtype).reshape(-1, 1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(features.dtype).reshape(-1, 1) * self.vy + self.y_offset)
        features_ls.append(f_center)
        if self.with_distance:
            features_ls.append(features[:, :, :3].norm(2, dim=2, keepdim=True))
        features = torch.cat(features_ls, dim=-1)
        mask = _pad_indicator(num_points_per_voxel, self.max_num_points_in_voxel)
        features = features * mask.reshape(-1, self.max_num_points_in_voxel, 1)
        for pfn in self.pfn_layers:
            features = pfn(features)
        return features.squeeze(1)


class PointPillarsScatter(nn.Module):
    def __init__(self, in_channels, voxel_size, point_cloud_range):
        super().__init__()
        self.in_channels = int(in_channels)
        rng = np.asarray(point_cloud_range, np.float32)
        vs = np.asarray(voxel_size, np.float32)
        g = np.round((rng[3:] - rng[:3]) / vs).astype(np.int64)
        self.nx, self.ny = int(g[0]), int(g[1])

    def forward(self, voxel_features, coords, batch_size):
        canvases = []
        for b in range(batch_size):
            canvas = voxel_features.new_zeros(self.nx * self.ny, self.in_channels)
            m = coords[:, 0] == b
            this = coords[m]
            idx = (this[:, 2] * self.nx + this[:, 3]).long()
            canvas[idx] = voxel_features[m]
            canvases.append(canvas.transpose(0, 1))
        out = torch.cat(canvases, 0)
        return out.reshape(batch_size, self.in_channels, self.ny, self.nx)


# --------------------------------------------------------------------------- #
# SecondBackbone / SecondFPN
# --------------------------------------------------------------------------- #
class SecondBackbone(nn.Module):
    def __init__(self, in_channels=64, out_channels=(64, 128, 256),
                 layer_nums=(3, 5, 5), downsample_strides=(1, 2, 2)):
        super().__init__()
        in_filters = [in_channels] + list(out_channels[:-1])
        blocks = []
        for i, n in enumerate(layer_nums):
            block = [nn.Conv2d(in_filters[i], out_channels[i], 3,
                               stride=downsample_strides[i], padding=1, bias=False),
                     nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                     nn.ReLU()]
            for _ in range(n):
                block += [nn.Conv2d(out_channels[i], out_channels[i], 3, padding=1, bias=False),
                          nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                          nn.ReLU()]
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return tuple(outs)


class SecondFPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256), out_channels=(128, 128, 128),
                 upsample_strides=(0.5, 1, 2), use_conv_for_no_stride=True,
                 use_spatial_attn_before_concat=False):
        super().__init__()
        deblocks = []
        for i, out_ch in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                layer = nn.ConvTranspose2d(in_channels[i], out_ch,
                                           kernel_size=upsample_strides[i],
                                           stride=upsample_strides[i], bias=False)
            else:
                s = round(1 / stride)
                layer = nn.Conv2d(in_channels[i], out_ch, kernel_size=s,
                                  stride=s, bias=False)
            deblocks.append(nn.Sequential(
                layer, nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01), nn.ReLU()))
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        return ups[0] if len(ups) == 1 else torch.cat(ups, dim=1)


# --------------------------------------------------------------------------- #
# CenterHead
# --------------------------------------------------------------------------- #
class _ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _SeparateHead(nn.Module):
    def __init__(self, in_channels, heads, head_conv=64, final_kernel=3,
                 init_bias=-2.19, eps=1e-5, momentum=0.1):
        super().__init__()
        self.heads = heads
        for head, (classes, num_conv) in heads.items():
            layers = []
            c_in = in_channels
            for _ in range(num_conv - 1):
                layers.append(_ConvModule(c_in, head_conv, final_kernel, 1,
                                          final_kernel // 2, eps, momentum))
                c_in = head_conv
            layers.append(nn.Conv2d(head_conv, classes, final_kernel, 1,
                                    final_kernel // 2, bias=True))
            setattr(self, head, nn.Sequential(*layers))
        with torch.no_grad():
            self.hm[-1].bias.fill_(init_bias)

    def forward(self, x):
        return {h: getattr(self, h)(x) for h in self.heads}


class CenterHead(nn.Module):
    def __init__(self, in_channels=384, tasks=None, common_heads=None,
                 share_conv_channel=64, init_bias=-2.19, num_hm_conv=2,
                 eps=1e-5, momentum=0.1):
        super().__init__()
        if common_heads is None:
            common_heads = {"reg": (2, 2), "height": (1, 2), "dim": (3, 2), "rot": (2, 2)}
        if tasks is None:
            tasks = [{"num_class": 1}, {"num_class": 2}]
        self.num_classes = [t["num_class"] for t in tasks]
        self.shared_conv = _ConvModule(in_channels, share_conv_channel, 3, 1, 1,
                                       eps, momentum)
        self.tasks = nn.ModuleList()
        for nc in self.num_classes:
            heads = dict(common_heads)
            heads.update({"hm": (nc, num_hm_conv)})
            self.tasks.append(_SeparateHead(share_conv_channel, heads,
                                            final_kernel=3, init_bias=init_bias,
                                            eps=eps, momentum=momentum))

    def forward(self, x):
        x = self.shared_conv(x)
        return [task(x) for task in self.tasks]


# --------------------------------------------------------------------------- #
# Whole model
# --------------------------------------------------------------------------- #
class CenterPointPillars(nn.Module):
    """Paddle3D CenterPoint-Pillars (2D-pillar, KITTI config)."""

    def __init__(self, num_class=(1, 2),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 voxel_size=(0.16, 0.16, 4.0), max_points=100,
                 feat_channels=(64, 64), in_channels=4):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.voxel_encoder = PillarFeatureNet(
            in_channels=in_channels, feat_channels=feat_channels,
            max_num_points_in_voxel=max_points, voxel_size=voxel_size,
            point_cloud_range=point_cloud_range, legacy=False)
        self.middle_encoder = PointPillarsScatter(
            64, voxel_size, point_cloud_range)
        self.backbone = SecondBackbone(64, (64, 128, 256), (3, 5, 5), (1, 2, 2))
        self.neck = SecondFPN((64, 128, 256), (128, 128, 128), (0.5, 1, 2), True)
        self.bbox_head = CenterHead(
            in_channels=384,
            tasks=[{"num_class": n} for n in num_class],
            common_heads={"reg": (2, 2), "height": (1, 2), "dim": (3, 2), "rot": (2, 2)})

    @torch.no_grad()
    def voxelize(self, points):
        """points: list of (Ni,4) numpy arrays -> stacked voxel tensors."""
        vs, coords_all, np_all = [], [], []
        for b, p in enumerate(points):
            v, c, n = hard_voxelize(p, self.point_cloud_range, self.voxel_size,
                                    self.max_points)
            c = c.copy()
            c[:, 0] = b
            vs.append(v)
            coords_all.append(c)
            np_all.append(n)
        voxels = torch.as_tensor(np.concatenate(vs, 0), dtype=torch.float32)
        coords = torch.as_tensor(np.concatenate(coords_all, 0), dtype=torch.int64)
        num = torch.as_tensor(np.concatenate(np_all, 0), dtype=torch.int64)
        return voxels, coords, num

    def forward(self, points):
        """points: list of (Ni,4) array (numpy) or a (B, N, 4) tensor."""
        if isinstance(points, torch.Tensor):
            points = [p.cpu().numpy() for p in points]
        batch_size = len(points)
        device = next(self.parameters()).device
        voxels, coords, num = self.voxelize(points)
        voxels = voxels.to(device)
        coords = coords.to(device)
        num = num.to(device)
        voxel_features = self.voxel_encoder(voxels, num, coords)
        x = self.middle_encoder(voxel_features, coords, batch_size)
        feats = self.backbone(x)
        x = self.neck(feats)
        return self.bbox_head(x)


# --------------------------------------------------------------------------- #
# Paddle -> torch weight mapping
# --------------------------------------------------------------------------- #
def load_paddle_centerpoint(model, pkl_path, verbose=True):
    """Load a Paddle3D ``.pdparams`` (dumped to numpy pickle) into the model."""
    import pickle

    sd = pickle.load(open(pkl_path, "rb"))
    mapped = {}
    for k, v in sd.items():
        t = torch.as_tensor(np.asarray(v))
        if k.endswith("linear.weight"):
            # Paddle Linear weight is [in, out]; PyTorch expects [out, in]
            t = t.t().contiguous()
        nk = k
        if k.endswith("._mean"):
            nk = k[: -len("._mean")] + ".running_mean"
        elif k.endswith("._variance"):
            nk = k[: -len("._variance")] + ".running_var"
        mapped[nk] = t
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if verbose:
        print("loaded tensors :", len(mapped))
        print("missing        :", len(missing), missing[:10])
        print("unexpected     :", len(unexpected), unexpected[:10])
    return missing, unexpected
