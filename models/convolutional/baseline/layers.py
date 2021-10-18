from typing import List, Tuple, Union, Optional
from functools import partial

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.sparse_conv_layers import \
    ConvBlock, ConvTransBlock, GenConvTransBlock, \
    GenerativeUpsample, GenerativeUpsampleMessage
from lib.torch_utils import minkowski_tensor_wrapped_op


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
        act: Optional[str],
        use_bn_for_last: bool,
        act_for_last: Optional[str]) -> nn.ModuleList:

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

            ConvBlock(
                intra_channels[idx],
                intra_channels[idx] if idx != len(intra_channels) - 1 else out_channels,
                3, 1,
                region_type=region_type,
                bn=use_batch_norm if idx != len(intra_channels) - 1 else use_bn_for_last,
                act=act if idx != len(intra_channels) - 1 else act_for_last
            ),
        ))

    return blocks


class NNSequentialWithConvTransBlockArgs(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for m in self:
            if isinstance(m, ConvTransBlock):
                x = m(x, *args, **kwargs)
            else:
                x = m(x)
        return x


def make_upsample_block(
        generative: bool,
        in_channels,
        out_channels,
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str],
        conv_trans_last: bool,
        use_bn_for_last: bool,
        act_for_last: Optional[str]):
    basic_block = partial(BLOCKS_DICT[basic_block_type],
                          region_type=region_type,
                          bn=use_batch_norm, act=act)
    if generative is True:
        upsample_block = GenConvTransBlock
    else:
        upsample_block = ConvTransBlock

    if conv_trans_last:
        ret = [
            ConvBlock(
                in_channels, in_channels, 3, 1,
                region_type=region_type,
                bn=use_batch_norm, act=act
            ),

            *[basic_block(in_channels) for _ in range(basic_block_num)],

            upsample_block(
                in_channels, out_channels, 2, 2,
                region_type='HYPER_CUBE',
                bn=use_bn_for_last,
                act=act_for_last
            )
        ]
    else:
        ret = [
            upsample_block(
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

    if generative is True:
        return nn.Sequential(*ret)
    else:
        return NNSequentialWithConvTransBlockArgs(*ret)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 intra_channels: Tuple[int, ...],
                 requires_points_num_list: bool,
                 points_num_scaler: float,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
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
            use_batch_norm, act,
            use_batch_norm, None
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
                 conv_trans_near_pruning: bool,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 **kwargs):
        super(DecoderBlock, self).__init__()
        upsample_block = make_upsample_block(
            True,
            in_channels, upsample_out_channels,
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act,
            conv_trans_near_pruning,
            use_batch_norm, act
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
                 conv_trans_near_pruning: bool,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 **kwargs):
        super(Decoder, self).__init__()

        self.blocks = nn.Sequential(*[
            DecoderBlock(in_channels if idx == 0 else intra_channels[idx - 1],
                         ch, ch,
                         conv_trans_near_pruning,
                         basic_block_type,
                         region_type,
                         basic_block_num,
                         use_batch_norm,
                         act,
                         **kwargs) for idx, ch in enumerate(intra_channels)])

    def forward(self, msg: GenerativeUpsampleMessage):
        return self.blocks(msg)

    def __repr__(self):
        return str(self.blocks)


def make_hyper_coder(
        coder_type: str,
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

        ConvBlock(
            intra_channels[-1],
            out_channels,
            3, 1,
            region_type=region_type,
            bn=use_batch_norm if coder_type == 'encoder' else False,
            act=None
        )
    )


HyperEncoder = partial(make_hyper_coder, 'encoder')
HyperDecoder = partial(make_hyper_coder, 'decoder')


class HyperEncoderForGeoLossLess(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 intra_channels: Tuple[int, ...],
                 shared_coder: bool,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(HyperEncoderForGeoLossLess, self).__init__()

        if not shared_coder:
            self.attn_blocks = nn.ModuleList(
                [nn.Sequential(
                    BLOCKS_DICT[basic_block_type](
                        ch, region_type, use_batch_norm, act
                    ),
                    ConvBlock(
                        ch, ch, 3, 1,
                        region_type=region_type, bn=False, act='sigmoid'
                    )
                ) for ch in (in_channels, *intra_channels[:-1])]
            )

            self.main_blocks = make_downsample_blocks(
                in_channels, intra_channels[-1], intra_channels,
                basic_block_type, region_type, basic_block_num,
                use_batch_norm, act,
                use_batch_norm, act
            )

            self.out_blocks = nn.ModuleList(
                [ConvBlock(
                    ch, out_channels,
                    3, 1,
                    region_type=region_type,
                    bn=use_batch_norm, act=None
                ) for ch in (in_channels, *intra_channels)]
            )

        else:
            for ch in intra_channels:
                assert ch == in_channels

            self.attn_blocks = nn.ModuleList(
                [nn.Sequential(
                    BLOCKS_DICT[basic_block_type](
                        in_channels, region_type, use_batch_norm, act
                    ),
                    ConvBlock(
                        in_channels, in_channels, 3, 1,
                        region_type=region_type, bn=False, act='sigmoid'
                    )
                )] * len(intra_channels)
            )

            self.main_blocks = nn.ModuleList([make_downsample_blocks(
                in_channels, in_channels, (in_channels,),
                basic_block_type, region_type, basic_block_num,
                use_batch_norm, act,
                use_batch_norm, act
            )[0]] * len(intra_channels))

            self.out_blocks = nn.ModuleList(
                [ConvBlock(
                    in_channels, out_channels,
                    3, 1,
                    region_type=region_type,
                    bn=use_batch_norm, act=None
                )] * (len(intra_channels) + 1)
            )

    def forward(self, x):
        intra_results = [x]
        ret = []

        for attn_block, main_block in zip(self.attn_blocks, self.main_blocks):
            attn = attn_block(intra_results[-1])
            ret.append(
                minkowski_tensor_wrapped_op(attn, lambda _: 1 - _) *
                intra_results[-1]
            )
            intra_results.append(main_block(attn * intra_results[-1]))

        for idx in range(len(ret)):
            ret[idx] = self.out_blocks[idx](ret[idx])
        ret.append(self.out_blocks[-1](intra_results[-1]))

        return ret


class EltWiseSelfAttn(nn.Module):
    def __init__(self, channels, basic_block_type, region_type, use_batch_norm, act):
        super(EltWiseSelfAttn, self).__init__()
        self.attn_block = nn.Sequential(
            BLOCKS_DICT[basic_block_type](
                channels, region_type, use_batch_norm, act
            ),
            ConvBlock(
                channels, channels, 3, 1,
                region_type=region_type, bn=False, act='sigmoid'
            )
        )

    def forward(self, x):
        return self.attn_block(x) * x


def decoder_list_for_geo_lossless(
        coder_type: str,
        generative: bool,
        in_channels: Tuple[int, ...],
        out_channels: Tuple[int, ...],
        shared_coder: bool,
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str]):

    if not shared_coder:
        return nn.ModuleList(
            [NNSequentialWithConvTransBlockArgs(
                EltWiseSelfAttn(a, basic_block_type, region_type, use_batch_norm, act),
                *make_upsample_block(
                    generative,
                    a, b,
                    basic_block_type, region_type, basic_block_num,
                    use_batch_norm, act,
                    False,
                    False if coder_type == 'hyper' else use_batch_norm,
                    None if coder_type == 'hyper' else act
                )
            ) for a, b in zip(in_channels, out_channels)]
        )

    else:
        for ch in in_channels[1:-1]:
            assert ch == in_channels[0]

        for ch in out_channels[1:]:
            assert ch == out_channels[0]

        shared_block = NNSequentialWithConvTransBlockArgs(
            EltWiseSelfAttn(in_channels[0], basic_block_type, region_type, use_batch_norm, act),
            *make_upsample_block(
                generative,
                in_channels[0], out_channels[0],
                basic_block_type, region_type, basic_block_num,
                use_batch_norm, act,
                False,
                False if coder_type == 'hyper' else use_batch_norm,
                None if coder_type == 'hyper' else act
            )
        )

        if in_channels[0] != in_channels[-1]:
            last_block = NNSequentialWithConvTransBlockArgs(
                ConvBlock(
                    in_channels[-1], in_channels[0],
                    3, 1,
                    region_type=region_type,
                    bn=use_batch_norm, act=act
                ), *shared_block
            )
            return nn.ModuleList(
                [*[shared_block] * (len(in_channels) - 1), last_block]
            )

        else:
            return nn.ModuleList([shared_block] * (len(in_channels)))


HyperDecoderListForGeoLossLess = partial(decoder_list_for_geo_lossless, 'hyper')
DecoderListForGeoLossLess = partial(decoder_list_for_geo_lossless, 'normal', False)
