from typing import List, Tuple, Union, Optional
from functools import partial

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.sparse_conv_layers import \
    ConvBlock, ConvTransBlock, GenConvTransBlock, \
    GenerativeUpsample, GenerativeUpsampleMessage, get_act_module
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
                 value_scaler: float,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(Encoder, self).__init__()

        self.requires_points_num_list = requires_points_num_list
        self.points_num_scaler = points_num_scaler
        self.value_scaler = value_scaler

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

        if self.value_scaler != 1:
            x = minkowski_tensor_wrapped_op(x, lambda _: _ * self.value_scaler)
            for idx in range(len(cached_feature_list)):
                cached_feature_list[idx] = minkowski_tensor_wrapped_op(
                    cached_feature_list[idx], lambda _: _ * self.value_scaler
                )

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


class HyperCoder(nn.Module):
    def __init__(self,
                 coder_type: str,
                 value_scaler: float,
                 in_channels,
                 out_channels,
                 intra_channels: Tuple[int, ...],
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(HyperCoder, self).__init__()
        self.value_scaler = value_scaler

        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)

        self.main = nn.Sequential(
            *[nn.Sequential(

                GenConvTransBlock(
                    intra_channels[idx - 1] if idx != 0 else in_channels,
                    intra_channels[idx],
                    2, 2,
                    region_type=region_type,
                    bn=use_batch_norm, act=act
                ) if coder_type.startswith('generative_upsample') and idx == 0 else
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
                bn=use_batch_norm if coder_type.endswith('encoder') else False,
                act=None
            )
        )

    def forward(self, x):
        x = self.main(x)
        if self.value_scaler != 1:
            x = minkowski_tensor_wrapped_op(x, lambda _: _ * self.value_scaler)
        return x


HyperEncoder = partial(HyperCoder, 'encoder')
HyperDecoder = partial(HyperCoder, 'decoder', 1.0)
HyperEncoderForGeoLossLess = partial(HyperCoder, 'generative_upsample_encoder')
HyperDecoderForGeoLossLess = HyperDecoder


class EncoderForGeoLossLess(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 coder_num: int,
                 value_scaler: Union[float, Tuple[float, ...]],
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(EncoderForGeoLossLess, self).__init__()
        self.blocks_num = coder_num
        if isinstance(value_scaler, float):
            self.value_scaler = [value_scaler] * self.blocks_num
        else:
            assert isinstance(value_scaler, List) or isinstance(value_scaler, Tuple)
            self.value_scaler = value_scaler
        self.first_act = get_act_module(act)

        hidden_channels = in_channels
        gate_channels = 4 * hidden_channels

        self.conv_i = ConvBlock(
            in_channels,
            gate_channels,
            2, 2,
            region_type='HYPER_CUBE',
            bn=use_batch_norm, act=act
        )

        self.conv_h = ConvBlock(
            hidden_channels,
            gate_channels,
            2, 2,
            region_type='HYPER_CUBE',
            bn=use_batch_norm, act=act
        )

        self.conv_c = ConvBlock(
            hidden_channels,
            hidden_channels,
            2, 2,
            region_type='HYPER_CUBE',
            bn=use_batch_norm, act=act
        )

        self.conv_out = ConvBlock(
            hidden_channels,
            out_channels,
            3, 1,
            region_type=region_type,
            bn=use_batch_norm, act=None
        )

        self.conv_coord = make_downsample_blocks(
            1, hidden_channels, (hidden_channels,),
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act,
            use_batch_norm, act
        )[0]

        self.hidden_channels = hidden_channels

    def forward(self, x):
        x = self.first_act(x)
        strided_fea_for_coord_list = []
        strided_fea_list = []

        cm = x.coordinate_manager
        coordinate_map_key = x.coordinate_map_key
        make_hidden = lambda: ME.SparseTensor(
            features=torch.zeros(
                (x.shape[0], self.hidden_channels),
                dtype=x.F.dtype,
                device=x.F.device
            ),
            coordinate_map_key=coordinate_map_key,
            coordinate_manager=cm
        )
        hx = make_hidden()
        cx = make_hidden()

        for idx in range(self.blocks_num):
            strided_fea_for_coord_list.append(
                self.conv_coord(ME.SparseTensor(
                    features=torch.ones(
                        (hx.shape[0], 1), dtype=hx.F.dtype, device=hx.F.device
                    ),
                    coordinate_map_key=coordinate_map_key,
                    coordinate_manager=cm
                ))
            )

            cx = self.conv_c(cx)
            gates = self.conv_h(hx)
            coordinate_map_key = cx.coordinate_map_key
            if idx == 0:
                gates = gates + self.conv_i(x)

            in_gate, forget_gate, cell_gate, out_gate = gates.F.chunk(4, 1)
            in_gate = torch.sigmoid(in_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.sigmoid(cell_gate)
            out_gate = torch.sigmoid(out_gate)

            cx = ME.SparseTensor(
                features=(forget_gate * cx.F) + (in_gate * cell_gate),
                coordinate_map_key=coordinate_map_key,
                coordinate_manager=cm
            )
            hx = ME.SparseTensor(
                out_gate * torch.tanh(cx.F),
                coordinate_map_key=coordinate_map_key,
                coordinate_manager=cm
            )
            strided_fea_list.append(hx)

        for idx, fea in enumerate(strided_fea_list):
            strided_fea_list[idx] = minkowski_tensor_wrapped_op(
                self.conv_out(fea),
                lambda _: _ * self.value_scaler[idx]
            )

        for idx, fea in enumerate(strided_fea_for_coord_list):
            strided_fea_for_coord_list[idx] = minkowski_tensor_wrapped_op(
                fea,
                lambda _: _ * self.value_scaler[idx]
            )

        return strided_fea_for_coord_list, strided_fea_list


class DecoderForGeoLossLess(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 coder_num: int,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(DecoderForGeoLossLess, self).__init__()
        self.blocks_num = coder_num
        gate_channels = 4 * hidden_channels

        self.conv_i = ConvTransBlock(
            in_channels,
            gate_channels,
            2, 2,
            region_type='HYPER_CUBE',
            bn=use_batch_norm, act=act
        )

        self.conv_h = ConvTransBlock(
            hidden_channels,
            gate_channels,
            2, 2,
            region_type='HYPER_CUBE',
            bn=use_batch_norm, act=act
        )

        self.conv_c = ConvTransBlock(
            hidden_channels,
            hidden_channels,
            2, 2,
            region_type='HYPER_CUBE',
            bn=use_batch_norm, act=act
        )

        self.hidden_channels = hidden_channels

    def forward(self, x_list: List[Union[ME.SparseTensor, ME.CoordinateMapKey]]):
        assert len(x_list) == self.blocks_num + 1

        cm = x_list[0].coordinate_manager
        coordinate_map_key = x_list[0].coordinate_map_key
        make_hidden = lambda: ME.SparseTensor(
            features=torch.zeros(
                (x_list[0].shape[0], self.hidden_channels),
                dtype=x_list[0].F.dtype,
                device=x_list[0].F.device
            ),
            coordinate_map_key=coordinate_map_key,
            coordinate_manager=cm
        )
        hx = make_hidden()
        cx = make_hidden()

        for idx in range(self.blocks_num):
            if idx == self.blocks_num - 1:
                coordinate_map_key = x_list[idx + 1]
            else:
                coordinate_map_key = x_list[idx + 1].coordinate_map_key
            assert isinstance(coordinate_map_key, ME.CoordinateMapKey)

            cx = self.conv_c(cx, coordinate_map_key)
            gates = self.conv_h(hx, coordinate_map_key) + \
                self.conv_i(x_list[idx], coordinate_map_key)

            in_gate, forget_gate, cell_gate, out_gate = gates.F.chunk(4, 1)
            in_gate = torch.sigmoid(in_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.sigmoid(cell_gate)
            out_gate = torch.sigmoid(out_gate)

            cx = ME.SparseTensor(
                features=(forget_gate * cx.F) + (in_gate * cell_gate),
                coordinate_map_key=coordinate_map_key,
                coordinate_manager=cm
            )
            hx = ME.SparseTensor(
                out_gate * torch.tanh(cx.F),
                coordinate_map_key=coordinate_map_key,
                coordinate_manager=cm
            )

        return hx
