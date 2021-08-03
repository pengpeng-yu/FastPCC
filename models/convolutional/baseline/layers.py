from typing import List, Tuple, Union, Optional, Any, Callable

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
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(BaseConvBlock, self).__init__()

        self.conv = conv_class(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=not bn,
            dimension=dimension)
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


class ConvBlock(BaseConvBlock):
    def __init__(self, *args, **kwargs):
        super(ConvBlock, self).__init__(ME.MinkowskiConvolution, *args, **kwargs)


class GenConvTransBlock(BaseConvBlock):
    def __init__(self, *args, **kwargs):
        super(GenConvTransBlock, self).__init__(ME.MinkowskiGenerativeConvolutionTranspose, *args, **kwargs)


class ResBlock(nn.Module):
    def __init__(self, channels, bn: bool = False):
        super(ResBlock, self).__init__()
        self.conv0 = ConvBlock(channels, channels, 3, 1, bn=bn, act=None)
        self.conv1 = ConvBlock(channels, channels, 3, 1, bn=bn, act=None)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.conv0(x)))
        out += x
        return out


class InceptionResBlock(nn.Module):
    def __init__(self, channels, out_channels=None, bn: bool = False):
        super(InceptionResBlock, self).__init__()
        if out_channels is None: out_channels = channels
        self.path_0 = nn.Sequential(
            ConvBlock(channels, out_channels // 4, 3, 1, bn=bn),
            ConvBlock(out_channels // 4, out_channels // 2, 3, 1, bn=bn, act=None))

        self.path_1 = nn.Sequential(
            ConvBlock(channels, out_channels // 4, 1, 1, bn=bn),
            ConvBlock(out_channels // 4, out_channels // 4, 3, 1, bn=bn),
            ConvBlock(out_channels // 4, out_channels // 2, 1, 1, bn=bn, act=None))

        if out_channels != channels:
            self.skip = ConvBlock(channels, out_channels, 3, 1, bn=bn, act=None)
        else:
            self.skip = None

    def forward(self, x):
        out0 = self.path_0(x)
        out1 = self.path_1(x)
        out = ME.cat(out0, out1) + (self.skip(x) if self.skip is not None else x)
        return out


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 res_blocks_num: int,
                 res_block_type: str,
                 use_batch_norm: bool,
                 intra_channels: Tuple[int] = (16, 32, 64, 32),
                 use_skip_connection=False):
        super(Encoder, self).__init__()
        assert len(intra_channels) == 4

        self.use_skip_connection = use_skip_connection
        if res_block_type == 'ResNet':
            basic_block = ResBlock
        elif res_block_type == 'InceptionResNet':
            basic_block = InceptionResBlock
        else: raise NotImplementedError

        ch = [*intra_channels, out_channels]

        self.block0 = nn.Sequential(
            ConvBlock(in_channels, ch[0], 3, 1, bn=use_batch_norm),
            ConvBlock(ch[0], ch[1], 2, 2, bn=use_batch_norm),
            *[basic_block(ch[1]) for _ in range(res_blocks_num)])

        self.block1 = nn.Sequential(
            ConvBlock(ch[1], ch[1], 3, 1, bn=use_batch_norm),
            ConvBlock(ch[1], ch[2], 2, 2, bn=use_batch_norm),
            *[basic_block(ch[2]) for _ in range(res_blocks_num)])

        self.block2 = nn.Sequential(
            ConvBlock(ch[2], ch[2], 3, 1, bn=use_batch_norm),
            ConvBlock(ch[2], ch[3], 2, 2, bn=use_batch_norm),
            *[basic_block(ch[3]) for _ in range(res_blocks_num)],
            ConvBlock(ch[3], ch[4], 3, 1, bn=use_batch_norm, act=None))

        if self.use_skip_connection:
            self.skip_block0 = nn.Sequential(
                ConvBlock(ch[1], ch[1], 3, 1, bn=use_batch_norm),
                ConvBlock(ch[1], ch[3], 4, 4, bn=use_batch_norm),
                ConvBlock(ch[3], ch[3], 3, 1, bn=use_batch_norm, act=None))

            self.skip_block1 = nn.Sequential(
                ConvBlock(ch[2], ch[2], 3, 1, bn=use_batch_norm),
                ConvBlock(ch[2], ch[3], 2, 2, bn=use_batch_norm),
                ConvBlock(ch[3], ch[3], 3, 1, bn=use_batch_norm, act=None))

            self.skip_block0_channels = ch[3]
            self.skip_block1_channels = ch[3]

        else:
            self.skip_block0 = self.skip_block1 = None
            self.skip_block0_channels = self.skip_block1_channels = 0

    def forward(self, x) -> Union[ME.SparseTensor, List[ME.SparseTensor], List[List[int]]]:
        points_num_list = [[_.shape[0] for _ in x.decomposed_coordinates]]
        cached_feature_list = []

        x = self.block0(x)
        if not self.use_skip_connection:
            cached_feature_list.append(x)
        else:
            cached_feature_list.append(self.skip_block0(x))

        points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])
        x = self.block1(x)
        if not self.use_skip_connection:
            cached_feature_list.append(x)
        else:
            cached_feature_list.append(self.skip_block1(x))

        points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])
        x = self.block2(x)

        return x, cached_feature_list, points_num_list


class SequentialKwArgs(nn.Sequential):
    def __init__(self, *args):
        super(SequentialKwArgs, self).__init__(*args)

    def forward(self, x, **kwargs):
        for idx, module in enumerate(self):
            if idx == 0: x = module(x, **kwargs)
            else: x = module(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 res_blocks_num: int,
                 res_block_type: str,
                 use_batch_norm: bool,
                 **kwargs):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_blocks_num = res_blocks_num

        if res_block_type == 'ResNet':
            self.basic_block = ResBlock
        elif res_block_type == 'InceptionResNet':
            self.basic_block = InceptionResBlock

        upsample_block = SequentialKwArgs(
            GenConvTransBlock(self.in_channels, self.out_channels, 2, 2, bn=use_batch_norm),
            ConvBlock(self.out_channels, self.out_channels, 3, 1, bn=use_batch_norm),
            *[self.basic_block(self.out_channels) for _ in range(self.res_blocks_num)])

        classify_block = ConvBlock(self.out_channels, 1, 3, 1, bn=use_batch_norm,
                                   act='relu' if kwargs.get('loss_type', None) == 'Dist' else None)

        self.generative_upsample = GenerativeUpsample(upsample_block, classify_block, **kwargs)

    def forward(self, x: GenerativeUpsampleMessage):
        return self.generative_upsample(x)


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 res_blocks_num: int,
                 res_block_type: str,
                 use_batch_norm: bool,
                 intra_channels: Tuple[int] = (64, 32, 16),
                 use_skip_connection=False,
                 skip_connection_channels=(0, 0),
                 skipped_fea_fusion_method='Cat',
                 **kwargs):
        super(Decoder, self).__init__()
        assert len(intra_channels) == 3
        assert len(skip_connection_channels) == 2

        self.blocks = nn.Sequential(
            DecoderBlock(in_channels, intra_channels[0],
                         res_blocks_num, res_block_type, use_batch_norm,
                         use_cached_feature=use_skip_connection,
                         cached_feature_fusion_method=skipped_fea_fusion_method,
                         **kwargs),
            DecoderBlock(intra_channels[0], intra_channels[1],
                         res_blocks_num, res_block_type, use_batch_norm,
                         use_cached_feature=use_skip_connection,
                         cached_feature_fusion_method=skipped_fea_fusion_method,
                         **kwargs),
            DecoderBlock(intra_channels[1], intra_channels[2],
                         res_blocks_num, res_block_type, use_batch_norm,
                         use_cached_feature=False,  # TODO
                         cached_feature_fusion_method=skipped_fea_fusion_method,
                         **kwargs))

        self.use_skip_connection = use_skip_connection
        if self.use_skip_connection:
            self.skip_block0 = GenConvTransBlock(skip_connection_channels[0], intra_channels[1], 4, 4,
                                                 bn=use_batch_norm, act=None)
            self.skip_block1 = GenConvTransBlock(skip_connection_channels[1], intra_channels[0], 2, 2,
                                                 bn=use_batch_norm, act=None)

        else:
            self.skip_block0 = self.skip_block1 = None

    def forward(self, x: GenerativeUpsampleMessage):
        if self.use_skip_connection:
            x.cached_fea_list[0] = self.skip_block0(x.cached_fea_list[0])
            x.cached_fea_list[1] = self.skip_block1(x.cached_fea_list[1])
        return self.blocks(x)
