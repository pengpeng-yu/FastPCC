from typing import List, Tuple, Union, Optional, Any, Callable
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.sparse_conv_layers import GenerativeUpsample, GenerativeUpsampleMessage


class BaseConvBlock(nn.Module):
    def __init__(self,
                 conv_class: Callable,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(BaseConvBlock, self).__init__()

        if region_type in ['HYPER_CUBE', 'HYPER_CROSS']:
            self.region_type = getattr(ME.RegionType, region_type)
        else:
            raise NotImplementedError

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
        if act is None or isinstance(act, nn.Module):
            self.act = act
        elif act == 'relu':
            self.act = ME.MinkowskiReLU(inplace=True)
        elif act.startswith('leaky_relu'):
            self.act = ME.MinkowskiLeakyReLU(
                negative_slope=float(act.split('(', 1)[1].split(')', 1)[0]),
                inplace=True)
        else: raise NotImplementedError

    def forward(self, x, **kwargs):
        x = self.conv(x, **kwargs)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def __repr__(self):
        return f'{str(self.conv).replace("Minkowski", "ME", 1)}, ' \
               f'region_type={self.region_type.name} ' \
               f'bn={self.bn is not None}, ' \
               f'act={str(self.act).replace("Minkowski", "ME", 1).rstrip("()")}'


class ConvBlock(BaseConvBlock):
    def __init__(self, *args, **kwargs):
        super(ConvBlock, self).__init__(ME.MinkowskiConvolution, *args, **kwargs)


class GenConvTransBlock(BaseConvBlock):
    def __init__(self, *args, **kwargs):
        super(GenConvTransBlock, self).__init__(ME.MinkowskiGenerativeConvolutionTranspose, *args, **kwargs)


class ResBlock(nn.Module):
    def __init__(self, channels, region_type: str, bn: bool, act: Optional[str]):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.bn = bn
        self.act = act
        self.region_type = region_type

        self.conv0 = ConvBlock(channels, channels, 3, 1, region_type=region_type, bn=bn, act=act)
        self.conv1 = ConvBlock(channels, channels, 3, 1, region_type=region_type, bn=bn, act=None)

    def forward(self, x):
        out = self.conv1(self.conv0(x))
        out += x
        return out

    def __repr__(self):
        return f'MEResBlock(channels={self.channels}, ' \
               f'region_type={self.region_type}, ' \
               f'bn={self.bn}, act={self.act})'


class InceptionResBlock(nn.Module):
    def __init__(self, channels, region_type: str, bn: bool, act: Optional[str]):
        super(InceptionResBlock, self).__init__()
        self.channels = channels
        self.bn = bn
        self.act = act
        self.region_type = region_type

        self.path_0 = nn.Sequential(
            ConvBlock(channels, channels // 4, 3, 1, region_type=region_type, bn=bn, act=act),
            ConvBlock(channels // 4, channels // 2, 3, 1, region_type=region_type, bn=bn, act=None))

        self.path_1 = nn.Sequential(
            ConvBlock(channels, channels // 4, 1, 1, region_type=region_type, bn=bn, act=act),
            ConvBlock(channels // 4, channels // 4, 3, 1, region_type=region_type, bn=bn, act=act),
            ConvBlock(channels // 4, channels // 2, 1, 1, region_type=region_type, bn=bn, act=None))

    def forward(self, x):
        out0 = self.path_0(x)
        out1 = self.path_1(x)
        out = ME.cat(out0, out1) + x
        return out

    def __repr__(self):
        return f'MEInceptionResBlock(channels={self.channels}, ' \
               f'region_type={self.region_type}, ' \
               f'bn={self.bn}, act={self.act})'


BLOCKS_LIST = [ResBlock, InceptionResBlock]
BLOCKS_DICT = {_.__name__: _ for _ in BLOCKS_LIST}


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 intra_channels: Tuple[int, ...],
                 basic_block_type: str,
                 region_type: str,
                 basic_blocks_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(Encoder, self).__init__()

        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)

        if intra_channels[0] != 0:
            self.first_block = ConvBlock(in_channels, intra_channels[0], 3, 1,
                                         region_type=region_type,
                                         bn=use_batch_norm, act=act)
        else:
            self.first_block = None
            intra_channels = (in_channels, *intra_channels[1:])

        self.blocks = nn.ModuleList()

        for idx in range(len(intra_channels) - 1):
            block = [
                ConvBlock(
                    intra_channels[idx],
                    intra_channels[idx + 1],
                    2, 2,
                    region_type='HYPER_CUBE',
                    bn=use_batch_norm, act=act
                ),

                *[basic_block(intra_channels[idx + 1]) for _ in range(basic_blocks_num)],

                # bn is always performed for the last conv of encoder
                ConvBlock(
                    intra_channels[idx + 1],
                    intra_channels[idx + 1] if idx != len(intra_channels) - 2 else out_channels,
                    3, 1,
                    region_type=region_type,
                    bn=use_batch_norm if idx != len(intra_channels) - 2 else True,
                    act=act if idx != len(intra_channels) - 2 else None
                ),
            ]

            self.blocks.append(nn.Sequential(*block))

    def forward(self, x) -> Union[ME.SparseTensor, List[ME.SparseTensor], List[List[int]]]:
        points_num_list = [[_.shape[0] for _ in x.decomposed_coordinates]]
        cached_feature_list = []

        if self.first_block is not None:
            x = self.first_block(x)

        for idx, block in enumerate(self.blocks):
            cached_feature_list.append(x)
            x = block(x)
            if idx != len(self.blocks) - 1:
                points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])

        return x, cached_feature_list, points_num_list


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 upsample_out_channels,
                 classifier_in_channels,
                 basic_block_type: str,
                 region_type: str,
                 basic_blocks_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 conv_trans_near_pruning: bool,
                 **kwargs):
        super(DecoderBlock, self).__init__()

        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)

        if conv_trans_near_pruning:
            upsample_block = nn.Sequential(
                ConvBlock(
                    in_channels, in_channels, 3, 1,
                    region_type=region_type,
                    bn=use_batch_norm, act=act
                ),

                *[basic_block(in_channels) for _ in range(basic_blocks_num)],

                GenConvTransBlock(
                    in_channels, upsample_out_channels, 2, 2,
                    region_type='HYPER_CUBE',
                    bn=use_batch_norm, act=act
                )
            )

        else:
            upsample_block = nn.Sequential(
                GenConvTransBlock(
                    in_channels, upsample_out_channels, 2, 2,
                    region_type='HYPER_CUBE',
                    bn=use_batch_norm, act=act
                ),

                ConvBlock(
                    upsample_out_channels, upsample_out_channels, 3, 1,
                    region_type=region_type,
                    bn=use_batch_norm, act=act
                ),

                *[basic_block(upsample_out_channels) for _ in range(basic_blocks_num)]
            )

        classify_block = ConvBlock(
            classifier_in_channels, 1,
            3, 1,
            region_type=region_type,
            bn=use_batch_norm,
            act='relu' if kwargs.get('loss_type', None) == 'Dist' else None
        )

        self.generative_upsample = GenerativeUpsample(upsample_block, classify_block, **kwargs)

    def forward(self, msg: GenerativeUpsampleMessage):
        return self.generative_upsample(msg)

    def __repr__(self):
        return str(self.generative_upsample)


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 intra_channels: Tuple[int, ...],
                 basic_block_type: str,
                 region_type: str,
                 basic_blocks_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 conv_trans_near_pruning: bool,
                 **kwargs):
        super(Decoder, self).__init__()

        self.blocks = nn.Sequential(*[
            DecoderBlock(in_channels if idx == 0 else intra_channels[idx - 1],
                         ch, ch,
                         basic_block_type,
                         region_type,
                         basic_blocks_num,
                         use_batch_norm,
                         act,
                         conv_trans_near_pruning,
                         **kwargs) for idx, ch in enumerate(intra_channels)])

    def forward(self, msg: GenerativeUpsampleMessage):
        return self.blocks(msg)

    def __repr__(self):
        return str(self.blocks)
