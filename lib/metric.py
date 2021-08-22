import subprocess

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


def mpeg_pc_error(infile1, infile2, resolution, normal=False, command='pc_error_d', threads=1):
    # https://github.com/NJUVISION/PCGCv2
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)",
                "h.       1(p2point)", "h.,PSNR  1(p2point)"]

    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)",
                "h.       2(p2point)", "h.,PSNR  2(p2point)"]

    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)",
                "h.        (p2point)", "h.,PSNR   (p2point)"]

    headers_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                      "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                      "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headersF

    command = f'{command}' \
              f' -a {infile1}' \
              f' -b {infile2}' \
              f' --hausdorff=1' \
              f' --resolution={resolution - 1}' \
              f' --nbThreads={threads}'

    if normal:
        headers += headers_p2plane
        command = command + ' -n ' + infile1

    results = {}
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')
        for key in headers:
            if key in line:
                line = line.strip()
                value = float(line[line.find(': ') + 2:])
                results[key] = value
        c = subp.stdout.readline()

    return results


def batch_image_psnr(a: torch.Tensor, b: torch.Tensor, max_val):
    assert a.shape == b.shape
    return 10 * (torch.log10(torch.tensor([max_val ** 2], device=a.device, dtype=torch.float)) -
                 torch.log10(F.mse_loss(a, b, reduction='none').mean(dim=[*range(1, a.ndim)])))
