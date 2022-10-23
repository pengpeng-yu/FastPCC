from typing import List, Tuple, Union, Optional
from functools import partial

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.sparse_conv_layers import \
    ConvBlock, ConvTransBlock, GenConvTransBlock, MEMLPBlock, ResBlock, InceptionResBlock, \
    NNSequentialWithConvTransBlockArgs, NNSequentialWithConvBlockArgs, \
    GenerativeUpsample, GenerativeUpsampleMessage


BLOCKS_LIST = [ResBlock, InceptionResBlock]
BLOCKS_DICT = {_.__name__: _ for _ in BLOCKS_LIST}
residuals_num_per_scale = 2


def make_downsample_blocks(
        in_channels: int,
        out_channels: int,
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
                region_type='HYPER_CUBE', bn=use_batch_norm, act=act
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


def make_upsample_block(
        generative: bool,
        in_channels: int,
        out_channels: int,
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str]):
    basic_block = partial(BLOCKS_DICT[basic_block_type],
                          region_type=region_type,
                          bn=use_batch_norm, act=act)
    if generative is True:
        upsample_block = GenConvTransBlock
    else:
        upsample_block = ConvTransBlock

    ret = [
        upsample_block(
            in_channels, out_channels, 2, 2,
            region_type='HYPER_CUBE', bn=use_batch_norm, act=act
        ),

        ConvBlock(
            out_channels, out_channels, 3, 1,
            region_type=region_type, bn=use_batch_norm, act=act
        ),

        *[basic_block(out_channels) for _ in range(basic_block_num)]
    ]

    if generative is True:
        return nn.Sequential(*ret)
    else:
        return NNSequentialWithConvTransBlockArgs(*ret)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 intra_channels: Tuple[int, ...],
                 first_conv_kernel_size: int,
                 requires_points_num_list: bool,
                 points_num_scaler: float,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 last_act: Optional[str]):
        super(Encoder, self).__init__()
        self.requires_points_num_list = requires_points_num_list
        self.points_num_scaler = points_num_scaler
        self.first_block = ConvBlock(
            in_channels, intra_channels[0], first_conv_kernel_size, 1,
            region_type=region_type, bn=use_batch_norm, act=act
        )
        self.blocks = make_downsample_blocks(
            intra_channels[0], out_channels, intra_channels[1:],
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act,
            use_batch_norm, last_act
        )

    def forward(self, x) -> Tuple[List[ME.SparseTensor], Optional[List[List[int]]]]:
        points_num_list = [[_.shape[0] for _ in x.decomposed_coordinates]]
        strided_fea_list = []
        x = self.first_block(x)
        strided_fea_list.append(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            strided_fea_list.append(x)
            if idx != len(self.blocks) - 1:
                points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])

        if not self.requires_points_num_list:
            points_num_list = None
        else:
            points_num_list = [points_num_list[0]] + \
                              [[int(n * self.points_num_scaler) for n in _]
                               for _ in points_num_list[1:]]

        return strided_fea_list, points_num_list


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 upsample_out_channels: int,
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
            use_batch_norm, act
        )
        classify_block = ConvBlock(
            upsample_out_channels, 1, 3, 1,
            region_type=region_type, bn=use_batch_norm,
            act='relu' if kwargs.get('loss_type', None) == 'Dist' else None
        )
        self.generative_upsample = GenerativeUpsample(
            upsample_block, classify_block, None, None, **kwargs
        )

    def forward(self, msg: GenerativeUpsampleMessage):
        msg = self.generative_upsample(msg)
        return msg

    def __repr__(self):
        return f'{self._get_name()} Wrapped {repr(self.generative_upsample)}'


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, ...]],
                 intra_channels: Tuple[int, ...],
                 coord_recon_p2points_weighted_bce: bool,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 **kwargs):
        super(Decoder, self).__init__()
        in_channels = [in_channels, *intra_channels[:-1]]
        self.coord_recon_p2points_weighted_bce = coord_recon_p2points_weighted_bce
        self.blocks = nn.Sequential(*[
            DecoderBlock(
                in_ch, intra_ch,
                basic_block_type, region_type, basic_block_num, use_batch_norm, act,
                **kwargs
            ) for in_ch, intra_ch in zip(in_channels, intra_channels)]
        )

    def forward(self, msg: GenerativeUpsampleMessage):
        msg.cached_fea_list = []
        msg = self.blocks[:-1](msg)
        msg.bce_weights_type = 'p2point' if self.coord_recon_p2points_weighted_bce else ''
        msg = self.blocks[-1](msg)
        return msg

    def __repr__(self):
        return f'{self._get_name()} Wrapped {repr(self.blocks)}'


class HyperCoder(nn.Module):
    def __init__(self,
                 coder_type: str,
                 in_channels: int,
                 out_channels: int,
                 intra_channels: Tuple[int, ...],
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(HyperCoder, self).__init__()
        self.coder_type = coder_type
        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)
        main_modules = []
        for idx in range(len(intra_channels)):
            if coder_type.startswith('upsample') and idx == 0:
                main_modules.append(
                    ConvTransBlock(
                        in_channels, intra_channels[0], 2, 2,
                        region_type='HYPER_CUBE', bn=use_batch_norm, act=act
                    ))
            else:
                main_modules.append(
                    ConvBlock(
                        intra_channels[idx - 1] if idx != 0 else in_channels,
                        intra_channels[idx], 3, 1,
                        region_type=region_type, bn=use_batch_norm, act=act
                    ))
            for _ in range(basic_block_num):
                main_modules.append(basic_block(intra_channels[idx]))

        main_modules.append(
            ConvBlock(
                intra_channels[-1], out_channels, 3, 1,
                region_type=region_type, bn=use_batch_norm, act=None
            ))
        self.main = NNSequentialWithConvTransBlockArgs(*main_modules)

    def forward(self, x, *args, **kwargs):
        x = self.main(x, *args, **kwargs)
        return x


HyperEncoder = partial(HyperCoder, 'encoder')
HyperDecoder = partial(HyperCoder, 'decoder')


class HyperDecoderUpsample(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 intra_channels: int,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(HyperDecoderUpsample, self).__init__()
        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)
        self.blocks = nn.ModuleList([
            NNSequentialWithConvBlockArgs(
                ConvBlock(in_channels, intra_channels, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=act),
                *(basic_block(intra_channels) for _ in range(basic_block_num)),
                ConvBlock(intra_channels, out_channels, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=None)
            ) for __ in range(residuals_num_per_scale - 1)
        ])
        self.blocks.append(
            NNSequentialWithConvTransBlockArgs(
                ConvTransBlock(in_channels, intra_channels, 2, 2,
                               region_type=region_type, bn=use_batch_norm, act=act),
                *(basic_block(intra_channels) for _ in range(basic_block_num)),
                ConvBlock(intra_channels, out_channels, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=None)
            ))

    def __getitem__(self, idx):
        return self.blocks[idx % residuals_num_per_scale]


class HyperDecoderGenUpsample(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 intra_channels: int,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(HyperDecoderGenUpsample, self).__init__()
        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                GenConvTransBlock(in_channels, intra_channels, 2, 2,
                                  region_type=region_type, bn=use_batch_norm, act=act),
                *(basic_block(intra_channels) for _ in range(basic_block_num)),
                ConvBlock(intra_channels, out_channels, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=None)
            )
        ])

    def __getitem__(self, idx):
        tgt_idx = (idx % residuals_num_per_scale) - (residuals_num_per_scale - 1)
        if 0 <= tgt_idx < len(self.blocks):
            return self.blocks[tgt_idx]
        else:
            return None


class EncoderRecurrent(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(EncoderRecurrent, self).__init__()
        hidden_channels = in_channels
        gate_channels = 1 * hidden_channels
        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)
        self.blocks_gate = nn.ModuleList([
            MEMLPBlock(
                out_channels, gate_channels, bn=use_batch_norm, act=None
            ) for _ in range(residuals_num_per_scale)
        ])
        self.block_out = MEMLPBlock(
            hidden_channels, out_channels, bn=use_batch_norm, act=None
        )
        with torch.no_grad():
            for block_gate in self.blocks_gate:
                torch.zero_(block_gate.mlp.linear.bias)
                block_gate.mlp.linear.weight[...] = 1
            torch.zero_(self.block_out.mlp.linear.bias)
            torch.zero_(self.block_out.mlp.linear.weight)
            self.block_out.mlp.linear.weight[:, :out_channels] = torch.eye(out_channels)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(hidden_channels, hidden_channels, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=act),
                *(basic_block(hidden_channels) for _ in range(basic_block_num)),
                ConvBlock(hidden_channels, hidden_channels, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=act),
            ) for __ in range(residuals_num_per_scale - 1)
        ])
        self.blocks.append(nn.Sequential(
            ConvBlock(
                hidden_channels, hidden_channels, 2, 2,
                region_type='HYPER_CUBE', bn=use_batch_norm, act=act
            ),
            *[basic_block(hidden_channels) for _ in range(basic_block_num)],
            ConvBlock(
                hidden_channels, hidden_channels, 3, 1,
                region_type=region_type, bn=use_batch_norm, act=act
            )
        ))
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

    def forward(self, x: ME.SparseTensor, batch_size: int):
        if not self.training: assert batch_size == 1
        strided_fea_list = []
        cm = x.coordinate_manager
        cx = x
        hx = self.block_out(cx)
        strided_fea_list.append(hx)

        while True:
            for block, block_gate in zip(self.blocks, self.blocks_gate):
                gate = block_gate(hx)
                forget_gate = torch.sigmoid(gate.F)
                cx = ME.SparseTensor(
                    features=(forget_gate * cx.F),
                    coordinate_map_key=cx.coordinate_map_key,
                    coordinate_manager=cm
                )
                cx = block(cx)
                hx = self.block_out(cx)
                strided_fea_list.append(hx)
            if hx.C.shape[0] == batch_size:
                break

        return strided_fea_list
