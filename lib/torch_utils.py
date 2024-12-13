import platform
from typing import List, Tuple, Dict, Callable

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        torch.manual_seed(seed)
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def select_device(logger, local_rank, device='', batch_size=None) -> Tuple[torch.device, List[int]]:
    # device = 'cpu' or 'Cuda:0,' or '0,1,2,3'
    s = ''
    device = str(device).strip().lower().replace('cuda:', '')
    cuda = device.lower() != 'cpu'
    if cuda:
        devices = [int(_) for _ in device.split(',') if _] if device else '0'
        n = len(devices)
        assert torch.cuda.is_available() and torch.cuda.device_count() >= n, \
            f'CUDA unavailable, invalid device {device} requested'
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        for d in devices:
            p = torch.cuda.get_device_properties(int(d))
            s += f" CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"  # bytes to MB
        if local_rank == -1:
            cuda_ids = devices
            torch_device = torch.device('cuda', cuda_ids[0])
        else:
            assert 0 <= local_rank < n
            cuda_ids = [devices[local_rank]]
            torch_device = torch.device('cuda', cuda_ids[0])
        torch.cuda.set_device(torch_device)
    else:
        s += 'CPU'
        cuda_ids = [-1]
        torch_device = torch.device('cpu')

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch_device, cuda_ids


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def concat_loss_dicts(loss_dict_a: Dict[str, torch.Tensor],
                      loss_dict_b: Dict[str, torch.Tensor],
                      b_key_to_a_key_f: Callable[[str], str] = lambda x: x,
                      b_value_transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
    for b_key in loss_dict_b:
        a_key = b_key_to_a_key_f(b_key)
        if a_key in loss_dict_a:
            loss_dict_a[a_key] = loss_dict_a[a_key] + b_value_transform(loss_dict_b[b_key])
        else:
            loss_dict_a[a_key] = b_value_transform(loss_dict_b[b_key])
    return loss_dict_a


class TorchCudaMaxMemoryAllocated:
    def __enter__(self, device=None):
        torch.cuda.reset_peak_memory_stats(device=device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.max_memory_allocated_kb = torch.cuda.max_memory_allocated(device=None) / 1024
        return False


if __name__ == '__main__':
    pass
