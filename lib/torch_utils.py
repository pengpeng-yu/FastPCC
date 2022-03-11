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
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
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
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f" CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"  # bytes to MB
        if local_rank == -1:
            cuda_ids = devices
            torch_device = torch.device('cuda:0')
        else:
            assert 0 <= local_rank < n
            cuda_ids = [devices[local_rank]]
            torch_device = torch.device('cuda', cuda_ids[0])
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
                 bn: Optional[str] = 'nn.bn1d',
                 act: Optional[str] = 'leaky_relu(0.2)',
                 version: str = 'linear',
                 skip_connection: Optional[str] = None):
        super(MLPBlock, self).__init__()
        assert version in ['linear', 'conv']
        assert act is None or act.split('(', 1)[0] in ['relu', 'leaky_relu']
        assert bn in ['nn.bn1d', 'custom', None]
        assert skip_connection in ['sum', 'concat', None]

        if bn == 'nn.bn1d':
            self.bn = nn.BatchNorm1d(out_channels)
        elif bn == 'custom':
            assert version == 'linear'
            self.bn = BatchNorm1dChnlLast(out_channels)
        elif bn is None:
            self.bn = None
        else: raise NotImplementedError

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
        else: raise NotImplementedError

        if self.bn is None and self.act is None and version == 'linear':
            print('Warning: You are using a MLPBlock without activation nor batchnorm, '
                  'which is identical to a nn.Linear(bias=True) object')

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
            elif self.bn is not None: x = self.bn(x)
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
        return f'MLPBlock(in_ch={self.in_channels}, out_ch={self.out_channels}, ' \
               f'act={self.act}, bn={self.bn}, version={self.version}, ' \
               f'skip={self.skip_connection}'


class BatchNorm1dChnlLast(nn.Module):
    # very slow
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1dChnlLast, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            var, mean = torch.var_mean(x, dim=[0, 1], keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            if self.track_running_stats:
                self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
                self.running_var = self.running_var * (1 - self.momentum) + var * self.momentum
        else:
            if self.track_running_stats:
                x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            else:
                var, mean = torch.var_mean(x, dim=[0, 1], keepdim=True, unbiased=False)
                x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        return x


def concat_loss_dicts(loss_dict_a: Dict[str, torch.Tensor],
                      loss_dict_b: Dict[str, torch.Tensor],
                      b_key_to_a_key_f: Callable[[str], str] = lambda x: x,
                      b_value_transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
    for b_key in loss_dict_b:
        a_key = b_key_to_a_key_f(b_key)
        if a_key in loss_dict_a:
            loss_dict_a[a_key] += b_value_transform(loss_dict_b[b_key])
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


def gumbel_sigmoid(logits, tau=1, hard=False):
    gumbels = -torch.empty_like(
        logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.sigmoid()
    if hard:
        # Straight through.
        y_hard = y_soft.round()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GumbelSigmoidMLPBlock(MLPBlock):
    """
    if version == 'linear':
        input: (N, L_1, ..., L_n, C_in)
        output: (N, L_1, ..., L_n, C_out)
    elif version == 'conv':
        input: (N, C_in, L_1, ..., L_n,)
        output: (N, C_out, L_1, ..., L_n)
    """
    def __init__(self, in_channels, batchnorm='nn.bn1d', version='linear',
                 skip_connection=None, tau=1, hard=False):
        super(GumbelSigmoidMLPBlock, self).__init__(
            in_channels=in_channels, out_channels=1, bn=batchnorm, act=None,
            version=version, skip_connection=skip_connection
        )
        self.tau = tau
        self.hard = hard

    def forward(self, x):
        logits = super(GumbelSigmoidMLPBlock, self).forward(x)
        logits = gumbel_sigmoid(logits, tau=self.tau, hard=self.hard)
        return logits * x


class KDNode:
    def __init__(self, point: torch.Tensor):
        super(KDNode, self).__init__()
        self.point = point
        self.left: Union[torch.Tensor, KDNode, None] = None
        self.right: Union[torch.Tensor, KDNode, None] = None


def create_kd_tree(data: torch.Tensor, max_num: int = 1) -> Union[KDNode, torch.Tensor]:
    if len(data) <= max_num:
        return data

    dim_index = torch.argmax(torch.var(data, dim=0))
    data_sorted = data[torch.argsort(data[:, dim_index])]

    point_index = len(data) // 2
    point = data_sorted[point_index]

    kd_node = KDNode(point)
    kd_node.left = create_kd_tree(data_sorted[:point_index], max_num)
    kd_node.right = create_kd_tree(data_sorted[point_index:], max_num)
    return kd_node


def kd_tree_partition(data: torch.Tensor, max_num: int, extras: List[torch.Tensor] = None)\
        -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[List[torch.Tensor]]]]:
    if extras is None or extras == []:
        return kd_tree_partition_base(data, max_num)
    else:
        return kd_tree_partition_extended(data, max_num, extras)


def kd_tree_partition_base(data: torch.Tensor, max_num: int) \
        -> List[torch.Tensor]:
    if len(data) <= max_num:
        return [data]

    dim_index = torch.argmax(torch.var(data, dim=0))
    data_sorted = data[torch.argsort(data[:, dim_index])]

    index_point = len(data) // 2
    left_partitions = kd_tree_partition_base(data_sorted[:index_point], max_num)
    right_partitions = kd_tree_partition_base(data_sorted[index_point:], max_num)
    left_partitions.extend(right_partitions)

    return left_partitions


def kd_tree_partition_extended(data: torch.Tensor, max_num: int, extras: List[torch.Tensor])\
        -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
    if len(data) <= max_num:
        return [data], [[extra] for extra in extras]

    dim_index = torch.argmax(torch.var(data, dim=0))
    arg_sorted = torch.argsort(data[:, dim_index])
    data_sorted = data[arg_sorted]

    for idx, extra in enumerate(extras):
        extras[idx] = extra[arg_sorted]
    del arg_sorted

    index_point = len(data) // 2
    left_partitions, left_extra_partitions = kd_tree_partition_extended(
        data_sorted[:index_point], max_num, [extra[:index_point] for extra in extras]
    )
    right_partitions, right_extra_partitions = kd_tree_partition_extended(
        data_sorted[index_point:], max_num, [extra[index_point:] for extra in extras]
    )
    left_partitions.extend(right_partitions)
    for idx, p in enumerate(right_extra_partitions):
        left_extra_partitions[idx].extend(p)

    return left_partitions, left_extra_partitions


class TorchCudaMaxMemoryAllocated:
    def __enter__(self, device=None):
        torch.cuda.reset_peak_memory_stats(device=device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.max_memory_allocated = torch.cuda.max_memory_allocated(device=None)
        return False


if __name__ == '__main__':
    pass
