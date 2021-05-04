import torch


def chamfer_loss(points1: torch.Tensor, points2: torch.Tensor, loss_factor=1.0):
    """
    Input:
        points1: [B, N, C]
        points2: [B, M, C]
    """
    nchnls = points1.shape[-1]
    assert nchnls == points2.shape[-1]
    with torch.no_grad():
        # matrix multiplication has lower percision
        # https://github.com/pytorch/pytorch/issues/37734
        dist = torch.cdist(points1, points2, compute_mode='donot_use_mm_for_euclid_dist')  # B, N, M
        nearest1_idx = dist.argmin(dim=2, keepdim=True)  # B, N, 1
        nearest2_idx = dist.argmin(dim=1, keepdim=True).permute(0, 2, 1)  # B, M, 1
    dist1 = points1 - torch.gather(points2, 1, nearest1_idx.expand(-1, -1, nchnls))  # B, N, C
    dist2 = points2 - torch.gather(points1, 1, nearest2_idx.expand(-1, -1, nchnls))  # B, M, C
    dist1, dist2 = dist1.square(), dist2.square()  # B, N/M, C

    assert 0 < loss_factor < 2
    loss = dist1.sum() * ((2 - loss_factor) / (points1.shape[0] * points1.shape[1])) + \
           dist2.sum() * (loss_factor / (points2.shape[0] * points2.shape[1]))
    return loss


def chamfer_loss_t(points1: torch.Tensor, points2: torch.Tensor, loss_factor=1.0, max_match_num=None):
    """
    Input:
        points1: [B, N, C]
        points2: [B, M, C]
    """
    nchnls = points1.shape[-1]
    assert nchnls == points2.shape[-1]
    with torch.no_grad():
        dist = torch.cdist(points1, points2, compute_mode='donot_use_mm_for_euclid_dist')  # B, N, M
        if max_match_num is None:
            nearest1_idx = dist.argmin(dim=2, keepdim=True)  # B, N, 1
            nearest2_idx = dist.argmin(dim=1, keepdim=True).permute(0, 2, 1)  # B, M, 1
        else:
            assert 0 < max_match_num < min(points1.shape[1], points2.shape[1])
            min_dist1, nearest1_idx = dist.min(dim=2)  # B, N
            min_dist2, nearest2_idx = dist.min(dim=1)  # B, M
            min_dist_top1 = min_dist1.topk(max_match_num, dim=1, sorted=True)[1]  # B, K
            min_dist_top2 = min_dist2.topk(max_match_num, dim=1, sorted=True)[1]  # B, K
            nearest1_idx = nearest1_idx.gather(1, min_dist_top1)
            nearest2_idx = nearest2_idx.gather(1, min_dist_top2)

    if max_match_num is None:
        dist1 = points1 - torch.gather(points2, 1, nearest1_idx.expand(-1, -1, nchnls))  # B, N, C
        dist2 = points2 - torch.gather(points1, 1, nearest2_idx.expand(-1, -1, nchnls))  # B, M, C
        dist1, dist2 = dist1.square(), dist2.square()  # B, N/M, C

        assert 0 < loss_factor < 2
        loss = dist1.sum() * ((2 - loss_factor) / (points1.shape[0] * points1.shape[1])) + \
               dist2.sum() * (loss_factor / (points2.shape[0] * points2.shape[1]))
        return loss

    else:
        dist1 = points1.gather(1, min_dist_top1[:, :, None].expand(-1, -1, nchnls)) - torch.gather(points2, 1, nearest1_idx[:, :, None].expand(-1, -1, nchnls))  # B, N, C
        dist2 = points2.gather(1, min_dist_top2[:, :, None].expand(-1, -1, nchnls)) - torch.gather(points1, 1, nearest2_idx[:, :, None].expand(-1, -1, nchnls))  # B, M, C
        dist1, dist2 = dist1.square(), dist2.square()  # B, N/M, C

        assert 0 < loss_factor < 2
        loss = dist1.sum() * ((2 - loss_factor) / (points1.shape[0] * max_match_num)) + \
               dist2.sum() * (loss_factor / (points2.shape[0] * max_match_num))
        return loss


# TODO: emd
def emd_loss():
    pass
