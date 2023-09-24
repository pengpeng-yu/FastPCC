import torch
try:
    from pytorch3d.ops.knn import knn_points
except ImportError: knn_points = None


def chamfer_loss(points1: torch.Tensor, points2: torch.Tensor,
                 version='pytorch3d', loss_factor=1.0, p=2.0):
    """
    Input:
        points1: [B, N, C]
        points2: [B, M, C]
    """
    nchnls = points1.shape[-1]
    assert nchnls == points2.shape[-1]

    if version == 'pytorch':
        with torch.no_grad():
            # matrix multiplication has lower precision
            # https://github.com/pytorch/pytorch/issues/37734
            # compute_mode='donot_use_mm_for_euclid_dist'
            dist = torch.cdist(points1, points2, )  # B, N, M
            nearest1_idx = dist.argmin(dim=2, keepdim=True)  # B, N, 1
            nearest2_idx = dist.argmin(dim=1, keepdim=True).permute(0, 2, 1)  # B, M, 1
        dist1 = points1 - torch.gather(points2, 1, nearest1_idx.expand(-1, -1, nchnls))  # B, N, C
        dist2 = points2 - torch.gather(points1, 1, nearest2_idx.expand(-1, -1, nchnls))  # B, M, C
        dist1.pow_(p)
        dist2.pow_(p)

    elif version == 'pytorch3d':
        if not p == 2: raise NotImplementedError
        dist1 = knn_points(points1, points2, K=1, return_sorted=False).dists
        dist2 = knn_points(points2, points1, K=1, return_sorted=False).dists

    else:
        raise NotImplementedError

    assert 0 < loss_factor < 2
    loss = dist1.sum() * ((2 - loss_factor) / (points1.shape[0] * points1.shape[1])) + \
           dist2.sum() * (loss_factor / (points2.shape[0] * points2.shape[1]))
    return loss


# TODO: emd
def emd_loss():
    pass
