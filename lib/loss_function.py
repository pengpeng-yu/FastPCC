import torch
from lib.pointnet_utils import square_distance


def chamfer_loss(points1: torch.Tensor, points2: torch.Tensor):
    """
    Input:
        points1: [B, N, C]
        points2: [B, M, C]
    """
    nchannels = points1.shape[-1]
    assert nchannels == points2.shape[-1]
    with torch.no_grad():
        dist = square_distance(points1, points2)  # B, N, M
        nearest1_idx = dist.argmin(dim=2, keepdim=True)  # B, N, 1
        nearest2_idx = dist.argmin(dim=1, keepdim=True).permute(0, 2, 1)  # B, M, 1
    dist1 = points1 - torch.gather(points2, 1, nearest1_idx.expand(-1, -1, nchannels))  # B, N, C
    dist2 = points2 - torch.gather(points1, 1, nearest2_idx.expand(-1, -1, nchannels))  # B, M, C
    dist1, dist2 = dist1.square().sum(-1), dist2.square().sum(-1)  # B, N/M

    loss = dist1.sum() / (points1.shape[0] * points1.shape[1]) + dist2.sum() / (points2.shape[0] * points2.shape[1])
    return loss

