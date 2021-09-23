from typing import List, Tuple, Union, Optional
from functools import partial

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.sparse_conv_layers import \
    ConvBlock, ConvTransBlock, GenConvTransBlock, \
    GenerativeUpsample, GenerativeUpsampleMessage


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


def make_downsample_blocks(
        in_channels,
        out_channels,
        intra_channels: Tuple[int, ...],
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str]) -> nn.ModuleList:

    basic_block = partial(BLOCKS_DICT[basic_block_type],
                          region_type=region_type,
                          bn=use_batch_norm, act=act)
    blocks = nn.ModuleList()

    for idx in range(len(intra_channels)):
        blocks.append(nn.Sequential(
            ConvBlock(
                intra_channels[idx - 1] if idx != 0 else in_channels,
                intra_channels[idx],
                2, 2,
                region_type='HYPER_CUBE',
                bn=use_batch_norm, act=act
            ),

            *[basic_block(intra_channels[idx]) for _ in range(basic_block_num)],

            # Bn is always performed for the last conv layer of an encoder.
            ConvBlock(
                intra_channels[idx],
                intra_channels[idx] if idx != len(intra_channels) - 1 else out_channels,
                3, 1,
                region_type=region_type,
                bn=use_batch_norm if idx != len(intra_channels) - 1 else True,
                act=act if idx != len(intra_channels) - 1 else None
            ),
        ))

    return blocks


def make_generative_upsample_block(
        in_channels,
        out_channels,
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str],
        conv_trans_last: bool):
    basic_block = partial(BLOCKS_DICT[basic_block_type],
                          region_type=region_type,
                          bn=use_batch_norm, act=act)

    if conv_trans_last:
        return nn.Sequential(
            ConvBlock(
                in_channels, in_channels, 3, 1,
                region_type=region_type,
                bn=use_batch_norm, act=act
            ),

            *[basic_block(in_channels) for _ in range(basic_block_num)],

            GenConvTransBlock(
                in_channels, out_channels, 2, 2,
                region_type='HYPER_CUBE',
                bn=use_batch_norm, act=act
            )
        )
    else:
        ret = [
            GenConvTransBlock(
                in_channels, out_channels, 2, 2,
                region_type='HYPER_CUBE',
                bn=use_batch_norm, act=act
            ),

            ConvBlock(
                out_channels, out_channels, 3, 1,
                region_type=region_type,
                bn=use_batch_norm, act=act
            ),

            *[basic_block(out_channels) for _ in range(basic_block_num)]
        ]
        return nn.Sequential(*ret)


def make_upsample_block(
        in_channels,
        out_channels,
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str],
        conv_trans_last: bool):
    basic_block = partial(BLOCKS_DICT[basic_block_type],
                          region_type=region_type,
                          bn=use_batch_norm, act=act)

    class NNSequentialWithConvTransBlockArgs(nn.Module):
        def __init__(self, *modules: nn.Module):
            super(NNSequentialWithConvTransBlockArgs, self).__init__()
            self.modules_list = nn.ModuleList()
            for m in modules:
                self.modules_list.append(m)

        def forward(self, x, *args, **kwargs):
            for m in self.modules_list:
                if isinstance(m, ConvTransBlock):
                    x = m(x, *args, **kwargs)
                else:
                    x = m(x)
            return x

    if conv_trans_last:
        return NNSequentialWithConvTransBlockArgs(
            ConvBlock(
                in_channels, in_channels, 3, 1,
                region_type=region_type,
                bn=use_batch_norm, act=act
            ),

            *[basic_block(in_channels) for _ in range(basic_block_num)],

            ConvTransBlock(
                in_channels, out_channels, 2, 2,
                region_type='HYPER_CUBE',
                bn=use_batch_norm, act=act
            )
        )
    else:
        ret = [
            ConvTransBlock(
                in_channels, out_channels, 2, 2,
                region_type='HYPER_CUBE',
                bn=use_batch_norm, act=act
            ),

            ConvBlock(
                out_channels, out_channels, 3, 1,
                region_type=region_type,
                bn=use_batch_norm, act=act
            ),

            *[basic_block(out_channels) for _ in range(basic_block_num)]
        ]
        return NNSequentialWithConvTransBlockArgs(*ret)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 intra_channels: Tuple[int, ...],
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 requires_points_num_list: bool,
                 points_num_scaler: float):
        super(Encoder, self).__init__()

        self.requires_points_num_list = requires_points_num_list
        self.points_num_scaler = points_num_scaler

        if intra_channels[0] != 0:
            self.first_block = ConvBlock(in_channels, intra_channels[0], 3, 1,
                                         region_type=region_type,
                                         bn=use_batch_norm, act=act)
        else:
            self.first_block = None
            intra_channels = (in_channels, *intra_channels[1:])

        self.blocks = make_downsample_blocks(
            intra_channels[0], out_channels, intra_channels[1:],
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act
        )

    def forward(self, x) -> Union[ME.SparseTensor, List[ME.SparseTensor], Optional[List[List[int]]]]:
        points_num_list = [[_.shape[0] for _ in x.decomposed_coordinates]]
        cached_feature_list = []

        if self.first_block is not None:
            x = self.first_block(x)

        for idx, block in enumerate(self.blocks):
            cached_feature_list.append(x)
            x = block(x)
            if idx != len(self.blocks) - 1:
                points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])

        if not self.requires_points_num_list:
            points_num_list = None
        else:
            points_num_list = [points_num_list[0]] + \
                              [[int(n * self.points_num_scaler) for n in _]
                               for _ in points_num_list[1:]]

        return x, cached_feature_list, points_num_list


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 upsample_out_channels,
                 classifier_in_channels,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 conv_trans_near_pruning: bool,
                 **kwargs):
        super(DecoderBlock, self).__init__()
        upsample_block = make_generative_upsample_block(
            in_channels, upsample_out_channels,
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act,
            conv_trans_near_pruning
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
                 basic_block_num: int,
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
                         basic_block_num,
                         use_batch_norm,
                         act,
                         conv_trans_near_pruning,
                         **kwargs) for idx, ch in enumerate(intra_channels)])

    def forward(self, msg: GenerativeUpsampleMessage):
        return self.blocks(msg)

    def __repr__(self):
        return str(self.blocks)


def make_hyper_coder(
        in_channels,
        out_channels,
        intra_channels: Tuple[int, ...],
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str]):
    basic_block = partial(BLOCKS_DICT[basic_block_type],
                          region_type=region_type,
                          bn=use_batch_norm, act=act)

    return nn.Sequential(
        *[nn.Sequential(

            ConvBlock(
                intra_channels[idx - 1] if idx != 0 else in_channels,
                intra_channels[idx],
                3, 1,
                region_type=region_type,
                bn=use_batch_norm, act=act
            ),

            *[basic_block(intra_channels[idx]) for _ in range(basic_block_num)]

        ) for idx in range(len(intra_channels))],

        # BN is always performed for the last conv layer of a hyper encoder or a hyper decoder.
        ConvBlock(
            intra_channels[-1],
            out_channels,
            3, 1,
            region_type=region_type,
            bn=True, act=None
        )
    )


HyperEncoder = HyperDecoder = make_hyper_coder


class HyperEncoderForGeoLossLess(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 intra_channels: Tuple[int, ...],
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 ):
        super(HyperEncoderForGeoLossLess, self).__init__()
        self.main_blocks = make_downsample_blocks(
            in_channels, out_channels, intra_channels,
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act
        )
        # BN is always performed for the last conv layer of an encoder.
        self.out_blocks = nn.ModuleList(
            ConvBlock(
                ch, out_channels,
                3, 1,
                region_type=region_type,
                bn=True, act=None
            ) for ch in (in_channels, *intra_channels[:-1])
        )

    def forward(self, x):
        intra_results = [x]
        for main_block in self.main_blocks:
            intra_results.append(main_block(intra_results[-1]))
        ret = []
        for out_block, intra_result in zip(self.out_blocks, intra_results[:-1]):
            ret.append(out_block(intra_result))
        ret.append(intra_results[-1])
        return ret


def hyper_decoder_list_for_geo_lossless(
        in_channels: Tuple[int, ...],
        out_channels: Tuple[int, ...],
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str]):
    return nn.ModuleList([make_generative_upsample_block(
        a, b,
        basic_block_type, region_type, basic_block_num,
        use_batch_norm, act,
        True
    ) for a, b in zip(in_channels, out_channels)])


HyperDecoderListForGeoLossLess = hyper_decoder_list_for_geo_lossless


def decoder_list_for_geo_lossless(
        in_channels: Tuple[int, ...],
        out_channels: Tuple[int, ...],
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str]):

    return nn.ModuleList([make_upsample_block(
        a, b,
        basic_block_type, region_type, basic_block_num,
        use_batch_norm, act,
        False
    ) for a, b in zip(in_channels, out_channels)])


DecoderListForGeoLossLess = decoder_list_for_geo_lossless
