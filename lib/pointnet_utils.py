import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
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
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


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
    new_dists = new_dists.reshape(*idx_shape)
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
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
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
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(nsample, sample_method, group_radius, ngroup, xyz, points_fea, returnfps=False, knn=False):
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
        new_points_fea: sampled points data, [B, npoint, nsample, 3+D]
    """
    if sample_method == 'fps':
        fps_idx = farthest_point_sample(xyz, nsample) # [B, npoint]
        sampled_xyz = index_points(xyz, fps_idx)
    elif sample_method == 'uniform':
        sampled_xyz = xyz[:, :nsample, :]
    elif sample_method is None and nsample is None:
        sampled_xyz = xyz
    else:
        raise NotImplementedError

    if knn:
        dists = torch.cdist(sampled_xyz, xyz, compute_mode='donot_use_mm_for_euclid_dist')  # B x npoint x N
        grouped_idx = dists.topk(ngroup, dim=-1, largest=False, sorted=True)[1]  # argsort()[:, :, :ngroup]  # B x npoint x K
    else:
        grouped_idx = query_ball_point(group_radius, ngroup, xyz, sampled_xyz)

    grouped_xyz = index_points(xyz, grouped_idx) # [B, npoint, nsample, C]
    grouped_xyz_relative = grouped_xyz - sampled_xyz[:, :, None, :]

    if points_fea is not None:
        grouped_points = index_points(points_fea, grouped_idx)
        new_points_fea = torch.cat([grouped_xyz_relative, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points_fea = grouped_xyz_relative

    if returnfps:
        return sampled_xyz, new_points_fea, grouped_xyz, fps_idx
    else:
        return sampled_xyz, new_points_fea


class PointNetSetAbstraction(nn.Module):
    def __init__(self, nsample, sample_method, group_radius, ngroup, in_channels, mlp_channels, knn=False, attn_xyz=False):
        super(PointNetSetAbstraction, self).__init__()
        self.nsample = nsample
        self.sample_method = sample_method
        self.group_radius = group_radius
        self.ngroup = ngroup
        self.knn = knn
        self.attn_xyz = attn_xyz
        self.mlps = []
        last_channel = in_channels
        for channels in mlp_channels:
            self.mlps.append(nn.Linear(last_channel, channels))
            self.mlps.append(nn.ReLU(inplace=True))
            last_channel = channels
        self.mlps = nn.Sequential(*self.mlps)

    def forward(self, xyz, points_fea):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            sampled_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        sampled_xyz, sampled_points = sample_and_group(self.nsample, self.sample_method, self.group_radius, self.ngroup,
                                                       xyz, points_fea, knn=self.knn)
        # sampled_xyz: sampled points position data, [B, npoint, C]
        # sampled_points: sampled points feature, [B, nsample, ngroup, C+D]
        if self.attn_xyz:
            group_xyz_norm = sampled_points.detach()[:, :, :, :3]

        sampled_points = self.mlps(sampled_points)

        # TODO: AttentivePooling?
        if self.attn_xyz:
            sampled_points, neighbor_index = torch.max(sampled_points, 2)

            neighbor_index = neighbor_index[:, :, :, None].expand(-1, -1, -1, 3)
            attn_xyz = torch.gather(group_xyz_norm, 2, neighbor_index)
            attn_xyz = torch.mean(attn_xyz, dim=2)
            attn_xyz = attn_xyz + sampled_xyz

            return attn_xyz, sampled_points

        else:
            # maxiunm for channels of each points group of each sampled point
            # [B, nsample, mlps_last_channel]
            return sampled_xyz, torch.max(sampled_points, 2)[0]

