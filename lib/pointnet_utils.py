import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points as pytorch3d_knn_points
import numpy as np


# reference https://github.com/yanx27/Pointnet_Pointnet2_pytorch, modified by Yang You


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
        new_points[i, j, k, l] == old_points[i, idx[i, j, k], l]
    """
    raw_size = idx.size()
    idx = idx.view(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.view(*raw_size, -1)


def knn_points(p1, p2, k, return_sorted=False, version='pytorch3d', pytorch3d_version=-1, **kwargs):
    if version == 'pytorch3d':  # square distance
        return pytorch3d_knn_points(p1, p2, K=k,
                                    return_sorted=return_sorted,
                                    version=pytorch3d_version, **kwargs)

    elif version == 'pytorch':  # distance
        if not pytorch3d_version == -1 and kwargs == {}: raise NotImplementedError
        torch.cdist(p1, p2, compute_mode='donot_use_mm_for_euclid_dist').\
            topk(k, dim=2, largest=False, sorted=return_sorted)

    else:
        raise NotImplementedError


def index_points_dists(dists, idx1, idx2):
    """
    Input:
        dists: [B, N, M]
        idx1: [B, *idx_shape] (value < N)
        idx2: [B, *idx_shape] (value < M)
    Return:
        dists: [B, *idx_shape]
        new_dists[i, j, k, ...] == old_dists[i, idx1[i, j, k, ...], idx2[i, j, k, ...]]
    """
    batch_size, n, m = dists.shape
    idx_shape = idx1.shape
    new_dists = torch.gather(dists, dim=1, index=idx1.reshape(batch_size, -1, 1).expand(-1, -1, m))
    new_dists = torch.gather(new_dists, dim=2, index=idx2.reshape(batch_size, -1, 1))
    new_dists = new_dists.reshape(idx_shape)
    return new_dists


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = index_points(xyz, farthest).unsqueeze(1)
        dist = torch.cdist(xyz, centroid).squeeze(2)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    # indices could be different with original version if there are more samples than nsample in one ball
    values, indices, _ = knn_points(new_xyz, xyz, nsample, return_sorted=False)
    # noinspection PyTypeChecker
    group_idx = torch.where(values < radius ** 2, indices, indices[:, :, :1])

    # # ori:
    # new_version_group_idx = group_idx.clone()
    # def square_distance(src, dst):
    #     return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)
    #
    # device = xyz.device
    # B, N, C = xyz.shape
    # _, S, _ = new_xyz.shape
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists = square_distance(new_xyz, xyz)
    # group_idx[sqrdists > radius ** 2] = N
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]
    #
    # # test:
    # for i in range(B):
    #     for j in range(S):
    #         try:
    #             assert torch.sum(new_version_group_idx[i, j].unique() != group_idx[i, j].unique()) == 0
    #         except AssertionError:
    #             assert torch.sum(sqrdists[i, j, new_version_group_idx[i, j]] >= radius ** 2) == 0

    return group_idx


def sample_and_group(nsample, group_radius, ngroup, xyz, points_fea, returnfps=False, knn=False):
    """
    sample points and aggregate nearby points feature
    Input:
        nsample: number of sampled points
        radius: used in query_ball_point if knn == False
        ngroup: number of grouped points for each sampled point
        xyz: input points position data, [B, N, 3]
        points_fea: input points data, [B, N, D]
    Return:
        sampled_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    fps_idx = farthest_point_sample(xyz, nsample)  # [B, npoint]
    sampled_xyz = index_points(xyz, fps_idx)

    if knn:
        dists, grouped_idx, _ = knn_points(sampled_xyz, xyz, k=ngroup, return_sorted=False)
    else:
        grouped_idx = query_ball_point(group_radius, ngroup, xyz, sampled_xyz)

    grouped_xyz = index_points(xyz, grouped_idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - sampled_xyz[:, :, None, :]

    if points_fea is not None:
        grouped_points = index_points(points_fea, grouped_idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return sampled_xyz, new_points, grouped_xyz, fps_idx
    else:
        return sampled_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    assert xyz.shape[2] == 3
    B, N, _ = xyz.shape
    new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, 3)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, nsample, group_radius, ngroup, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.nsample = nsample
        self.group_radius = group_radius
        self.ngroup = ngroup
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points_fea):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            sampled_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            sampled_xyz, sampled_points = sample_and_group_all(xyz, points_fea)
        else:
            sampled_xyz, sampled_points = sample_and_group(self.nsample,
                                                           self.group_radius,
                                                           self.ngroup,
                                                           xyz, points_fea,
                                                           knn=self.knn)
        # sampled_xyz: sampled points position data, [B, npoint, C]
        # sampled_points: sampled points data, [B, nsample, ngroup, C+D]
        sampled_points = sampled_points.permute(0, 3, 2, 1)  # [B, C+D, ngroup, nsample]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            sampled_points = F.relu(bn(conv(sampled_points)))

        sampled_points = torch.max(sampled_points, 2)[0].transpose(1, 2)  # [B, nsample, last_channel]
        # maximum for each channel for each points group of sampled point
        return sampled_xyz, sampled_points

