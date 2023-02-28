from typing import Tuple, List, Dict, Union, Callable, Optional

import MinkowskiEngine as ME
from torch import nn as nn


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
    else:
        raise NotImplementedError
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
                 act: Union[str, nn.Module, None] = 'relu'):
        super(BaseConvBlock, self).__init__()

        self.region_type = getattr(ME.RegionType, region_type)

        self.conv = conv_class(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=not bn,
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
                 act: Union[str, nn.Module, None] = 'relu'):
        super(ConvBlock, self).__init__(
            ME.MinkowskiConvolution,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, act
        )


class ConvTransBlock(BaseConvBlock):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(ConvTransBlock, self).__init__(
            ME.MinkowskiConvolutionTranspose,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, act
        )


class GenConvTransBlock(BaseConvBlock):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(GenConvTransBlock, self).__init__(
            ME.MinkowskiGenerativeConvolutionTranspose,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, act
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
            out = ME.SparseTensor(
                out.F.relu(),
                coordinate_manager=out.coordinate_manager,
                coordinate_map_key=out.coordinate_map_key
            )
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
