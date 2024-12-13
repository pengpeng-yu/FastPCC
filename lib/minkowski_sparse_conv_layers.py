import math
from functools import wraps
from typing import Union, Callable, Optional, Any, Tuple, Dict, List

import torch
from torch import nn as nn

import MinkowskiEngine as ME


def get_act_module(act: Union[str, nn.Module, None]) -> Optional[nn.Module]:
    if isinstance(act, nn.Module):
        act_module = act
    elif act is None or act == 'None':
        act_module = None
    elif act == 'relu':
        act_module = ME.MinkowskiReLU(inplace=True)
    elif act.startswith('leaky_relu'):
        act_module = ME.MinkowskiLeakyReLU(
            negative_slope=float(act.split('(', 1)[1].split(')', 1)[0]),
            inplace=True)
    elif act == 'sigmoid':
        act_module = ME.MinkowskiSigmoid()
    elif act == 'prelu':
        act_module = ME.MinkowskiPReLU()
    else:
        raise NotImplementedError(act)
    return act_module


class MEMLPBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(MEMLPBlock, self).__init__()
        self.mlp = ME.MinkowskiLinear(in_channels, out_channels, bias=not bn)
        self.bn = ME.MinkowskiBatchNorm(out_channels) if bn else None
        self.act = get_act_module(act)

    def forward(self, x):
        x = self.mlp(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def __repr__(self):
        return f'MEMLPBlock(in_ch={self.mlp.linear.in_features}, ' \
               f'out_ch={self.mlp.linear.out_features}, ' \
               f'bn={self.bn is not None}, act={self.act})'


class BaseConvBlock(nn.Module):
    def __init__(self,
                 conv_class: Callable,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 bias: Optional[bool] = None,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(BaseConvBlock, self).__init__()
        self.region_type = getattr(ME.RegionType, region_type)
        self.conv = conv_class(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias if bias is not None else not bn,
            kernel_generator=ME.KernelGenerator(
                kernel_size,
                stride,
                dilation,
                region_type=self.region_type,
                dimension=dimension),
            dimension=dimension
        )
        self.bn = ME.MinkowskiBatchNorm(out_channels) if bn else None
        self.act = act
        self.act_module = get_act_module(act)

    def forward(self, x, *args, **kwargs):
        x = self.conv(x, *args, **kwargs)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_module is not None:
            x = self.act_module(x)
        return x

    def __repr__(self):
        kernel_size = self.conv.kernel_generator.kernel_size
        if len(set(kernel_size)) == 1:
            kernel_size = kernel_size[0]
        stride = self.conv.kernel_generator.kernel_stride
        if len(set(stride)) == 1:
            stride = stride[0]
        dilation = self.conv.kernel_generator.kernel_dilation
        if len(set(dilation)) == 1:
            dilation = dilation[0]
        return \
            f'{self.conv.__class__.__name__.replace("Minkowski", "ME", 1).replace("Convolution", "Conv", 1)}(' \
            f'in={self.conv.in_channels}, out={self.conv.out_channels}, ' \
            f'kernel_volume={self.conv.kernel_generator.kernel_volume}, ' \
            f'kernel_size={kernel_size}, ' \
            f'stride={stride}, ' \
            f'dilation={dilation}, ' \
            f'bn={self.bn is not None}, ' \
            f'act={self.act_module.__class__.__name__.replace("Minkowski", "ME", 1)})'


class ConvBlock(BaseConvBlock):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 bias: Optional[bool] = None,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(ConvBlock, self).__init__(
            ME.MinkowskiConvolution,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, bias, act
        )


class ConvTransBlock(BaseConvBlock):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 bias: Optional[bool] = None,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(ConvTransBlock, self).__init__(
            ME.MinkowskiConvolutionTranspose,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, bias, act
        )


class GenConvTransBlock(BaseConvBlock):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 bias: Optional[bool] = None,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(GenConvTransBlock, self).__init__(
            ME.MinkowskiGenerativeConvolutionTranspose,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, bias, act
        )


class ResBlock(nn.Module):
    def __init__(self, channels, region_type: str, bn: bool, act: Optional[str],
                 kernel_size: int = 3, last_act: bool = False):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.bn = bn
        self.act = act
        self.region_type = region_type
        self.kernel_size = kernel_size
        self.last_act = get_act_module(act) if last_act is True else None
        self.conv0 = ConvBlock(channels, channels, kernel_size, 1, region_type=region_type, bn=bn, act=act)
        self.conv1 = ConvBlock(channels, channels, kernel_size, 1, region_type=region_type, bn=bn, act=None)

    def forward(self, x):
        out = self.conv1(self.conv0(x))
        out += x
        if self.last_act is not None:
            out = self.last_act(out)
        return out

    def __repr__(self):
        info = f'MEResBlock(channels={self.channels}, ' \
               f'bn={self.bn}, act={self.act}'
        if self.kernel_size != 3:
            info += f', kernel_size={self.kernel_size}'
        if self.region_type != 'HYPER_CUBE':
            info += f', region_type={self.region_type}'
        info += ')'
        return info


class InceptionResBlock(nn.Module):
    def __init__(self, channels, region_type: str, bn: bool, act: Optional[str], kernel_size: int = 3):
        super(InceptionResBlock, self).__init__()
        self.channels = channels
        self.bn = bn
        self.act = act
        self.region_type = region_type
        self.kernel_size = kernel_size
        self.path_0 = nn.Sequential(
            ConvBlock(channels, channels // 4, kernel_size, 1, region_type=region_type, bn=bn, act=act),
            ConvBlock(channels // 4, channels // 2, kernel_size, 1, region_type=region_type, bn=bn, act=None)
        )
        self.path_1 = nn.Sequential(
            ConvBlock(channels, channels // 4, 1, 1, region_type=region_type, bn=bn, act=act),
            ConvBlock(channels // 4, channels // 4, kernel_size, 1, region_type=region_type, bn=bn, act=act),
            ConvBlock(channels // 4, channels // 2, 1, 1, region_type=region_type, bn=bn, act=None)
        )

    def forward(self, x):
        out0 = self.path_0(x)
        out1 = self.path_1(x)
        out = ME.cat(out0, out1) + x
        return out

    def __repr__(self):
        info = f'MEInceptionResBlock(channels={self.channels}, ' \
               f'bn={self.bn}, act={self.act}'
        if self.kernel_size != 3:
            info += f', kernel_size={self.kernel_size}'
        if self.region_type != 'HYPER_CUBE':
            info += f', region_type={self.region_type}'
        info += ')'
        return info


class NNSequentialWithArgs(nn.Sequential):
    target_block_class = None

    def forward(self, x, *args, **kwargs):
        used_flag = False
        for m in self:
            if used_flag is False and isinstance(m, self.target_block_class):
                x = m(x, *args, **kwargs)
                used_flag = True
            else:
                x = m(x)
        if args or kwargs:
            assert used_flag
        return x


class NNSequentialWithConvTransBlockArgs(NNSequentialWithArgs):
    target_block_class = ConvTransBlock


class NNSequentialWithConvBlockArgs(NNSequentialWithArgs):
    target_block_class = ConvBlock


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
