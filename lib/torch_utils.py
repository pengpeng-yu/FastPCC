import os
import platform
import math
from functools import wraps
from typing import List, Tuple, Union, Dict, Optional, Callable, Any

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

try:
    import MinkowskiEngine as ME
except ImportError: ME = None


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


class MLPBlock(nn.Module):
    """
    if version == 'linear':
        input: (N, L_1, ..., L_n, C_in)
        output: (N, L_1, ..., L_n, C_out)
    elif version == 'conv':
        input: (N, C_in, L_1, ..., L_n,)
        output: (N, C_out, L_1, ..., L_n)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bn: bool = False,
                 act: Optional[str] = 'leaky_relu(0.2)',
                 version: str = 'linear',
                 skip_connection: Optional[str] = None):
        super(MLPBlock, self).__init__()
        assert version in ['linear', 'conv']
        assert act is None or act.split('(', 1)[0] in ['relu', 'leaky_relu', 'prelu']
        assert skip_connection in ['sum', 'concat', None]

        self.bn = nn.BatchNorm1d(out_channels) if bn is True else None
        if version == 'linear':
            self.mlp = nn.Linear(in_channels, out_channels, bias=self.bn is None)
        elif version == 'conv':
            self.mlp = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=self.bn is None)

        if act is None:
            self.act = None
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act.startswith('leaky_relu'):
            self.act = nn.LeakyReLU(
                negative_slope=float(act.split('(', 1)[1].split(')', 1)[0]),
                inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU()
        else: raise NotImplementedError(act)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.version = version
        self.skip_connection = skip_connection

    def forward(self, x):
        ori_x = x
        ori_shape = x.shape
        if len(ori_shape) < 3:
            raise NotImplementedError

        if self.version == 'linear':
            assert ori_shape[-1] == self.mlp.in_features
            if len(ori_shape) > 3:
                x = x.contiguous().view(ori_shape[0], -1, ori_shape[-1])
            x = self.mlp(x)
            if isinstance(self.bn, nn.BatchNorm1d):
                x = x.permute(0, 2, 1)
                x = self.bn(x)
                x = x.permute(0, 2, 1)
            if self.act is not None: x = self.act(x)
            if len(ori_shape) != 3:
                x = x.view(*ori_shape[:-1], self.out_channels)
            if self.skip_connection is None:
                pass
            elif self.skip_connection == 'sum':
                x = x + ori_x
            elif self.skip_connection == 'concat':
                x = torch.cat([ori_x, x], dim=-1)
            return x

        elif self.version == 'conv':
            if len(ori_shape) > 3:
                x = x.view(ori_shape[0], ori_shape[1], -1)
            x = self.mlp(x)
            if self.bn is not None: x = self.bn(x)
            if self.act is not None: x = self.act(x)
            if len(ori_shape) != 3:
                x = x.view(ori_shape[0], self.out_channels, *ori_shape[2:])
            if self.skip_connection is None:
                pass
            elif self.skip_connection == 'sum':
                x = x + ori_x
            elif self.skip_connection == 'concat':
                x = torch.cat([ori_x, x], dim=1)
            return x

    def __repr__(self):
        info = f'MLPBlock(in_ch={self.in_channels}, out_ch={self.out_channels}, ' \
               f'act={repr(self.act).replace("inplace=True", "")}'
        if self.bn is not None:
            info += f', bn={self.bn}'
        if self.version != 'linear':
            info += f', version={self.version}'
        if self.skip_connection is not None:
            info += f', skip={self.skip_connection}'
        info += ')'
        return info


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


def minkowski_tensor_wrapped_op(
        x: Union[torch.Tensor, ME.SparseTensor],
        operation: Callable[[torch.Tensor], Any],
        needs_recover: bool = True,
        add_batch_dim: bool = False):
    if needs_recover is True:
        assert add_batch_dim is False
    if ME is None or isinstance(x, torch.Tensor):
        return operation(x)

    ret = operation(x.F)
    ret = list(ret) if isinstance(ret, Tuple) else [ret]

    if needs_recover is True:
        for idx in range(len(ret)):
            if isinstance(ret[idx], torch.Tensor):
                ret[idx] = ME.SparseTensor(
                    features=ret[idx],
                    coordinate_map_key=x.coordinate_map_key,
                    coordinate_manager=x.coordinate_manager
                )
    elif add_batch_dim is True:
        for idx in range(len(ret)):
            if isinstance(ret[idx], torch.Tensor):
                ret[idx] = ret[idx][None]
    ret = tuple(ret)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def get_minkowski_tensor_coords_tuple(x):
    try:
        sparse_tensor_coords_tuple = x.coordinate_map_key, \
                                     x.coordinate_manager
    except AttributeError:
        sparse_tensor_coords_tuple = None
    return sparse_tensor_coords_tuple


def minkowski_tensor_wrapped_fn(
        inout_mapping_dict: Dict[Union[int, str], Union[int, List[int], None]] = None,
        add_batch_dim: bool = True):
    inout_mapping_dict = inout_mapping_dict or {}

    def func_decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            ret_coords: Dict[int, Union[Tuple, List]] = {}
            args = list(args)

            if ME is None or inout_mapping_dict == {}:
                needs_recover = False
            else:
                needs_recover = True
                for in_key, out_idx in inout_mapping_dict.items():
                    if isinstance(in_key, str) and in_key[0] == '<':
                        flag_end = in_key.find('>')
                        assert flag_end != -1
                        flag = in_key[1: flag_end]
                        in_key = in_key[flag_end + 1:]
                    else:
                        flag = None
                    try:
                        in_key = int(in_key)
                    except ValueError: pass
                    if isinstance(in_key, int):
                        collection = args
                    elif isinstance(in_key, str):
                        collection = kwargs
                    else:
                        raise NotImplementedError
                    try:
                        obj = collection[in_key]
                    except (IndexError, KeyError):
                        # If designated var is not provided, ignore it.
                        continue
                    if isinstance(obj, ME.SparseTensor):
                        collection[in_key] = obj.F
                        if add_batch_dim:
                            collection[in_key] = collection[in_key][None]
                        if out_idx is not None:
                            if not isinstance(out_idx, list):
                                out_idx = [out_idx]
                            for i in out_idx:
                                ret_coords[i] = (obj.coordinate_map_key, obj.coordinate_manager)
                    elif isinstance(obj, tuple) or isinstance(obj, list):
                        assert len(obj) == 2
                        assert isinstance(obj[0], ME.CoordinateMapKey)
                        assert isinstance(obj[1], ME.CoordinateManager)
                        if out_idx is not None:
                            if not isinstance(out_idx, list):
                                out_idx = [out_idx]
                            for i in out_idx:
                                ret_coords[i] = obj
                    else:
                        # If designated var is not a ME.SparseTensor
                        # or a tuple of a CoordinateMapKey and a CoordinateManager,
                        # ignore it.
                        pass
                    if flag == 'del':
                        del collection[in_key]
                    elif flag is None: pass
                    else: raise NotImplementedError
            if ret_coords == {}: needs_recover = False

            ret = func(*args, **kwargs)

            if needs_recover:
                ret = list(ret) if isinstance(ret, tuple) else [ret]
                for out_idx, (coords_key, coords_mg) in ret_coords.items():
                    assert isinstance(ret[out_idx], torch.Tensor)
                    ret[out_idx] = ME.SparseTensor(
                        features=ret[out_idx][0] if add_batch_dim else ret[out_idx],
                        coordinate_map_key=coords_key,
                        coordinate_manager=coords_mg
                    )
                if len(ret) == 1:
                    ret = ret[0]
                else:
                    ret = tuple(ret)
            return ret
        return wrapped_func
    return func_decorator


def minkowski_tensor_split(x, split_size: Union[int, List[int]]) -> List:
    ret = []
    if isinstance(split_size, List):
        points = torch.empty(len(split_size), 2, dtype=torch.long)
        points[:, 1] = torch.cumsum(torch.tensor(split_size), dim=0)
        points[1:, 0] = points[:-1, 1]
        points[0, 0] = 0
    elif isinstance(split_size, int):
        block_num = math.ceil(x.F.shape[1] / split_size)
        assert block_num > 1
        print(block_num)
        points = torch.empty(block_num, 2, dtype=torch.long)
        points[:, 0] = torch.arange(0, block_num, dtype=torch.long) * split_size
        points[:-1, 1] = points[1:, 0]
        points[-1, 1] = x.F.shape[1]
    else: raise NotImplementedError

    for start, end in points:
        ret.append(ME.SparseTensor(
            features=x.F[:, start: end],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager))
    return ret


def minkowski_expand_coord_2x(coord: torch.Tensor, current_tensor_stride: int):
    assert coord.ndim == 2 and coord.shape[1] == 4
    strides = torch.tensor(((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0),
                            (0, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1)),
                           dtype=coord.dtype, device=coord.device) * (current_tensor_stride // 2)
    return coord.unsqueeze(1) + strides.unsqueeze(0)


class TorchCudaMaxMemoryAllocated:
    def __enter__(self, device=None):
        torch.cuda.reset_peak_memory_stats(device=device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.max_memory_allocated_kb = torch.cuda.max_memory_allocated(device=None) / 1024
        return False


if __name__ == '__main__':
    pass
