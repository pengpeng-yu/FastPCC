import torch
import torch.nn.functional as F


def precision_recall(pred, tgt):
    true_pos = (pred & tgt).sum().item()
    false_pos = (pred & ~tgt).sum().item()
    false_neg = (~pred & tgt).sum().item()
    return {'Precision': true_pos / (true_pos + false_pos),
            'Recall': true_pos / (true_pos + false_neg),
            'TP': true_pos,
            'FP': false_pos,
            'FN': false_neg,
            'total': len(tgt)}


def batch_image_psnr(a: torch.Tensor, b: torch.Tensor, max_val):
    assert a.shape == b.shape
    return 10 * (torch.log10(torch.tensor([max_val ** 2], device=a.device, dtype=torch.float)) -
                 torch.log10(F.mse_loss(a, b, reduction='none').mean(dim=[*range(1, a.ndim)])))
