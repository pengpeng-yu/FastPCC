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


def rgb_to_yuvbt709(rgb: torch.Tensor):
    assert rgb.dtype == torch.float32
    assert rgb.ndim == 2 and rgb.shape[1] == 3
    y = (0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]) / 255.0
    u = (-0.1146 * rgb[:, 0] - 0.3854 * rgb[:, 1] + 0.5000 * rgb[:, 2]) / 255.0 + 0.5
    v = (0.5000 * rgb[:, 0] - 0.4542 * rgb[:, 1] - 0.0458 * rgb[:, 2]) / 255.0 + 0.5
    return torch.cat([y[:, None], u[:, None], v[:, None]], dim=1)


def gen_rgb_to_yuvbt709_param():
    weight = torch.tensor(
        [[0.2126, 0.7152, 0.0722],
         [-0.1146,  -0.3854, 0.5000],
         [0.5000, -0.4542, -0.0458]],
        dtype=torch.float
    ) / 255
    bias = torch.tensor([0, 0.5, 0.5], dtype=torch.float)
    return weight, bias
