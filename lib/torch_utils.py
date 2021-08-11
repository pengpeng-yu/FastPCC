import os
import platform
import math
from functools import wraps
from typing import List, Tuple, Union, Dict, Optional, Callable

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


def select_device(logger, local_rank, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = ''
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space} CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)"  # bytes to MB
    else:
        s += 'CPU'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    if cuda and local_rank == -1:
        return torch.device('cuda:0')
    elif cuda and local_rank != -1:
        return torch.device('cuda', local_rank)
    else:
        return torch.device('cpu')


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

        if self.bn is None and self.act is None:
            print('Warning: You are using a MLPBlock without activation nor batchnorm, '
                  'which is identical to a nn.Linear(bias=True) object')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.version = version
        self.skip_connection = skip_connection

    def forward(self, x):
        ori_x = x

        if self.version == 'linear':

            ori_shape = x.shape
            if len(ori_shape) != 3:
                x = x.view(ori_shape[0], -1, ori_shape[-1])

            x = self.mlp(x)
            if isinstance(self.bn, nn.BatchNorm1d):
                x = x.permute(0, 2, 1)
                x = self.bn(x)
                x = x.permute(0, 2, 1)
            elif self.bn is not None:
                x = self.bn(x)

            if self.act is not None:
                x = self.act(x)

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
            ori_shape = x.shape
            if len(ori_shape) != 3:
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
        return f'MLPBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'activation={self.act}, bn={self.bn}, version={self.version}, ' \
               f'skip_connection={self.skip_connection}'


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


def unbatched_coordinates(coords: torch.Tensor):
    # Used for Minkowski batched sparse tensor
    assert len(coords.shape) == 2 and coords.shape[1] == 4
    return [coords[coords[:, 0] == batch_idx, 1:] for batch_idx in range(coords[:, 0].max() + 1)]


def minkowski_tensor_wrapped(inout_mapping_dict: Dict[Union[int, str], Optional[int]] = None,
                             add_batch_dim: bool = True,
                             extra_preparation: Dict[Union[int, str], Callable] = None):

    inout_mapping_dict = inout_mapping_dict or {}
    predefined_extra_preparation = extra_preparation or {}

    def func_decorator(func):
        @wraps(func)
        def wrapped_func(*args,
                         extra_preparation: Dict[Union[int, str], Callable] = None,
                         **kwargs):
            if extra_preparation is None:
                extra_preparation = predefined_extra_preparation

            arg_coords_keys = {}
            arg_coord_mgs = {}

            new_pos_args = list(args)

            for idx, in_idx in enumerate(inout_mapping_dict):
                if isinstance(in_idx, int):
                    collection = new_pos_args
                elif isinstance(in_idx, str):
                    collection = kwargs
                else:
                    raise NotImplementedError

                obj = collection[in_idx]
                if not isinstance(obj, ME.SparseTensor):
                    assert idx == 0
                    needs_recover = False
                    break

                arg_coords_keys[in_idx] = obj.coordinate_map_key
                arg_coord_mgs[in_idx] = obj.coordinate_manager
                collection[in_idx] = obj.F
                if add_batch_dim:
                    collection[in_idx] = collection[in_idx][None]

            else:
                needs_recover = True

            if inout_mapping_dict == {}:
                needs_recover = False

            for in_idx in extra_preparation:
                if isinstance(in_idx, int):
                    collection = new_pos_args
                elif isinstance(in_idx, str):
                    collection = kwargs
                else:
                    raise NotImplementedError

                if isinstance(collection[in_idx], torch.Tensor):
                    collection[in_idx] = extra_preparation[in_idx](collection[in_idx])

                elif isinstance(collection[in_idx], ME.SparseTensor):
                    collection[in_idx] = ME.SparseTensor(
                        extra_preparation[in_idx](collection[in_idx].F),
                        coordinate_map_key=collection[in_idx].coordinate_map_key,
                        coordinate_manager=collection[in_idx].coordinate_manager
                    )

                else: raise NotImplementedError

            new_pos_args = tuple(new_pos_args)

            ret = func(*new_pos_args, **kwargs)

            if needs_recover:
                ret = list(ret)
                for in_idx, out_idx in inout_mapping_dict.items():
                    if out_idx is not None:
                        assert isinstance(out_idx, int)
                        assert isinstance(ret[out_idx], torch.Tensor)
                        ret[out_idx] = ME.SparseTensor(
                            ret[out_idx][0] if add_batch_dim else ret[out_idx],
                            coordinate_map_key=arg_coords_keys[in_idx],
                            coordinate_manager=arg_coord_mgs[in_idx]
                        )

                if len(ret) == 1:
                    ret = ret[0]
                else:
                    ret = tuple(ret)

            return ret
        return wrapped_func
    return func_decorator


def minkowski_tensor_split(x: ME.SparseTensor, split_size: Union[int, List[int]]) \
        -> List[ME.SparseTensor]:
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
        super(GumbelSigmoidMLPBlock, self).__init__(in_channels=in_channels, out_channels=1, bn=batchnorm, act=None,
                                                    version=version, skip_connection=skip_connection)
        self.tau = tau
        self.hard = hard

    def forward(self, x):
        logits = super(GumbelSigmoidMLPBlock, self).forward(x)
        logits = gumbel_sigmoid(logits, tau=self.tau, hard=self.hard)
        return logits * x


if __name__ == '__main__':
    pass
