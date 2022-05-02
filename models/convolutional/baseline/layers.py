from typing import List, Tuple, Union, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from lib.torch_utils import concat_loss_dicts
from lib.sparse_conv_layers import \
    ConvBlock, ConvTransBlock, GenConvTransBlock, \
    GenerativeUpsample, GenerativeUpsampleMessage
from lib.torch_utils import minkowski_tensor_wrapped_op
from lib.entropy_models.continuous_indexed import ContinuousIndexedEntropyModel
from lib.entropy_models.hyperprior.noisy_deep_factorized.sparse_tensor_specialized import \
    BytesListUtils


class Scaler(nn.Module):
    def __init__(self, scaler):
        super(Scaler, self).__init__()
        self.scaler = scaler

    def forward(self, x):
        return minkowski_tensor_wrapped_op(x, lambda _: _ * self.scaler)


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
            ConvBlock(channels // 4, channels // 2, 3, 1, region_type=region_type, bn=bn, act=None)
        )
        self.path_1 = nn.Sequential(
            ConvBlock(channels, channels // 4, 1, 1, region_type=region_type, bn=bn, act=act),
            ConvBlock(channels // 4, channels // 4, 3, 1, region_type=region_type, bn=bn, act=act),
            ConvBlock(channels // 4, channels // 2, 1, 1, region_type=region_type, bn=bn, act=None)
        )

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
        used_flag = False
        for m in self:
            if isinstance(m, ConvTransBlock):
                assert used_flag is False
                x = m(x, *args, **kwargs)
                used_flag = True
            else:
                x = m(x)
        if args or kwargs:
            assert used_flag
        return x


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
                 in_channels: int,
                 out_channels: int,
                 intra_channels: Tuple[int, ...],
                 first_conv_kernel_size: int,
                 requires_points_num_list: bool,
                 points_num_scaler: float,
                 res_feature_channels: int,
                 keep_raw_fea_in_strided_list: bool,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 last_act: Optional[str]):
        super(Encoder, self).__init__()
        self.requires_points_num_list = requires_points_num_list
        self.points_num_scaler = points_num_scaler
        self.res_feature_channels = res_feature_channels
        self.if_gen_res_fea = res_feature_channels != 0 and len(intra_channels) >= 3
        self.keep_raw_fea_in_strided_list = keep_raw_fea_in_strided_list
        self.first_block = ConvBlock(
            in_channels, intra_channels[0], first_conv_kernel_size, 1,
            region_type=region_type,
            bn=use_batch_norm, act=act
        )
        self.blocks = make_downsample_blocks(
            intra_channels[0], out_channels, intra_channels[1:],
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act,
            use_batch_norm, last_act
        )
        if self.if_gen_res_fea:
            self.conv_gate_list = nn.ModuleList([
                ConvBlock(
                    res_feature_channels, intra_ch, 2, 2,
                    region_type='HYPER_CUBE',
                    bn=use_batch_norm, act=act
                ) for intra_ch in intra_channels[2:]
            ])
            self.conv_out_list = nn.ModuleList([
                ConvBlock(
                    intra_ch, res_feature_channels, 3, 1,
                    region_type=region_type,
                    bn=use_batch_norm, act=None
                ) for intra_ch in intra_channels[1:-1]
            ])
            if not self.keep_raw_fea_in_strided_list:
                self.conv_gate_list.insert(
                    0, ConvBlock(
                        in_channels, intra_channels[1], 2, 2,
                        region_type='HYPER_CUBE',
                        bn=use_batch_norm, act=act
                    )
                )
                self.conv_out_list.insert(
                    0, ConvBlock(
                        intra_channels[0], in_channels, 3, 1,
                        region_type=region_type,
                        bn=use_batch_norm, act=None
                    ))

    def forward(self, x) -> Tuple[List[ME.SparseTensor], Optional[List[List[int]]]]:
        points_num_list = [[_.shape[0] for _ in x.decomposed_coordinates]]
        strided_fea_list = []
        cm = x.coordinate_manager

        if not self.if_gen_res_fea:
            if self.keep_raw_fea_in_strided_list:
                strided_fea_list.append(x)
            x = self.first_block(x)
            if not self.keep_raw_fea_in_strided_list:
                strided_fea_list.append(x)
            for idx, block in enumerate(self.blocks):
                x = block(x)
                strided_fea_list.append(x)
                if idx != len(self.blocks) - 1:
                    points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])

        elif self.if_gen_res_fea:
            if self.keep_raw_fea_in_strided_list:
                strided_fea_list.append(x)
                x = self.first_block(x)
                for idx, block in enumerate(self.blocks):
                    x = block(x)
                    if idx != 0:
                        forget = torch.sigmoid((self.conv_gate_list[idx - 1](strided_fea_list[-1])).F)
                        x = ME.SparseTensor(
                            features=(forget * x.F),
                            coordinate_map_key=x.coordinate_map_key,
                            coordinate_manager=cm
                        )
                    if idx != len(self.blocks) - 1:
                        strided_fea_list.append(self.conv_out_list[idx](x))
                    else:
                        strided_fea_list.append(x)
                    if idx != len(self.blocks) - 1:
                        points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])

            elif not self.keep_raw_fea_in_strided_list:
                x = self.first_block(x)
                strided_fea_list.append(self.conv_out_list[0](x))
                for idx, block in enumerate(self.blocks):
                    x = block(x)
                    forget = torch.sigmoid((self.conv_gate_list[idx](strided_fea_list[-1])).F)
                    x = ME.SparseTensor(
                        features=(forget * x.F),
                        coordinate_map_key=x.coordinate_map_key,
                        coordinate_manager=cm
                    )
                    if idx != len(self.blocks) - 1:
                        strided_fea_list.append(self.conv_out_list[idx + 1](x))
                    else:
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
                 out_channels: int,
                 upsample_out_channels: int,
                 residual_channels: int,
                 residual_indexes_scaler: float,
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
            upsample_out_channels, 1,
            3, 1,
            region_type=region_type,
            bn=use_batch_norm,
            act='relu' if kwargs.get('loss_type', None) == 'Dist' else None
        )
        if out_channels != 0 and residual_channels != 0:
            predict_block = nn.Sequential(
                ConvBlock(
                    upsample_out_channels, out_channels,
                    3, 1,
                    region_type=region_type,
                    bn=use_batch_norm,
                    act=None
                ),
                Scaler(residual_indexes_scaler)
            )
            residual_block = ConvBlock(
                residual_channels, residual_channels,
                3, 1,
                region_type=region_type,
                bn=use_batch_norm,
                act=None
            )
        elif out_channels == residual_channels == 0:
            predict_block = residual_block = None
        else:
            raise ValueError(f'out_channels: {out_channels}, '
                             f'residual_channels: {residual_channels}')
        self.generative_upsample = GenerativeUpsample(
            upsample_block, classify_block, predict_block, residual_block, **kwargs
        )

    def forward(self, msg: GenerativeUpsampleMessage):
        msg = self.generative_upsample(msg)
        return msg


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, ...]],
                 out_channels: Union[int, Tuple[int, ...]],
                 intra_channels: Tuple[int, ...],
                 residual_channels: Optional[Tuple[int, ...]],
                 residual_indexes_scaler: Optional[float],
                 indexed_em: Optional[ContinuousIndexedEntropyModel],
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 **kwargs):
        super(Decoder, self).__init__()
        if indexed_em is None:
            self.use_residual = False
            assert isinstance(in_channels, int) and out_channels == 0
            in_channels = [in_channels, *intra_channels[:-1]]
            out_channels = [0] * len(intra_channels)
            residual_channels = [0] * len(intra_channels)
        else:
            self.use_residual = True
            assert len(in_channels) == len(out_channels) == \
                   len(intra_channels) == len(residual_channels)
            assert isinstance(residual_indexes_scaler, float)
            self.indexed_em = indexed_em
        self.blocks = nn.Sequential(*[
            DecoderBlock(
                in_ch, out_ch, intra_ch, residual_ch, residual_indexes_scaler,
                basic_block_type, region_type, basic_block_num, use_batch_norm, act,
                **kwargs
            ) for in_ch, out_ch, intra_ch, residual_ch in
            zip(in_channels, out_channels, intra_channels, residual_channels)]
        )

    def forward(self, msg: GenerativeUpsampleMessage):
        if self.use_residual:
            msg.indexed_em = self.indexed_em
            msg: GenerativeUpsampleMessage = self.blocks(msg)
            loss_dict = {}
            for idx, sub_loss_dict in enumerate(msg.em_loss_dict_list[::-1]):
                concat_loss_dicts(loss_dict, sub_loss_dict, lambda k: f'fea_lossy_{idx}_' + k)
            msg.em_loss_dict_list = [loss_dict]

        else:
            msg.cached_fea_list = []
            msg = self.blocks(msg)
        return msg

    def compress(self, msg: GenerativeUpsampleMessage):
        assert self.use_residual and not self.training
        msg.indexed_em = self.indexed_em
        msg.em_flag = 'compress'
        msg: GenerativeUpsampleMessage = self.blocks(msg)
        msg.em_bytes_list = [BytesListUtils.concat_bytes_list(msg.em_bytes_list)]
        return msg

    def decompress(self, msg: GenerativeUpsampleMessage, em_bytes_list_len: int):
        assert self.use_residual and not self.training
        msg.indexed_em = self.indexed_em
        msg.em_flag = 'decompress'
        assert len(msg.em_bytes_list) == 1
        msg.em_bytes_list = BytesListUtils.split_bytes_list(
            msg.em_bytes_list[0], em_bytes_list_len
        )
        msg: GenerativeUpsampleMessage = self.blocks(msg)
        return msg


class HyperCoder(nn.Module):
    def __init__(self,
                 coder_type: str,
                 post_value_scaler: float,
                 in_channels: int,
                 out_channels: int,
                 intra_channels: Tuple[int, ...],
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(HyperCoder, self).__init__()
        self.post_value_scaler = post_value_scaler
        self.coder_type = coder_type
        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)
        main_modules = []
        for idx in range(len(intra_channels)):
            if coder_type.startswith('generative_upsample') and idx == 0:
                main_modules.append(
                    GenConvTransBlock(
                        in_channels,
                        intra_channels[0],
                        2, 2,
                        region_type=region_type,
                        bn=use_batch_norm, act=act
                    )
                )
            elif coder_type.startswith('upsample') and idx == 0:
                main_modules.append(
                    ConvTransBlock(
                        in_channels,
                        intra_channels[0],
                        2, 2,
                        region_type=region_type,
                        bn=use_batch_norm, act=act
                    )
                )
            else:
                main_modules.append(
                    ConvBlock(
                        intra_channels[idx - 1] if idx != 0 else in_channels,
                        intra_channels[idx],
                        3, 1,
                        region_type=region_type,
                        bn=use_batch_norm, act=act
                    )
                )
            for _ in range(basic_block_num):
                main_modules.append(basic_block(intra_channels[idx]))

        main_modules.append(
            ConvBlock(
                intra_channels[-1],
                out_channels,
                3, 1,
                region_type=region_type,
                bn=use_batch_norm, act=None
            )
        )
        self.main = NNSequentialWithConvTransBlockArgs(*main_modules)

    def forward(self, x, *args, **kwargs):
        x = self.main(x, *args, **kwargs)
        if self.post_value_scaler != 1:
            x = minkowski_tensor_wrapped_op(x, lambda _: _ * self.post_value_scaler)
        return x


HyperEncoder = partial(HyperCoder, 'encoder')
HyperDecoder = partial(HyperCoder, 'decoder')
HyperDecoderGenUpsample = partial(HyperCoder, 'generative_upsample_decoder')
HyperDecoderUpsample = partial(HyperCoder, 'upsample_decoder')


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
        self.conv_h = ConvBlock(
            out_channels,
            gate_channels,
            2, 2,
            region_type='HYPER_CUBE',
            bn=use_batch_norm, act=act
        )
        self.conv_c = make_downsample_blocks(
            hidden_channels, hidden_channels, (hidden_channels,),
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act,
            use_batch_norm, None
        )[0]
        self.conv_out = ConvBlock(
            hidden_channels,
            out_channels,
            3, 1,
            region_type=region_type,
            bn=use_batch_norm, act=None
        )
        self.hidden_channels = hidden_channels

    def forward(self, x: ME.SparseTensor, coder_num: int):
        strided_fea_list = []
        cm = x.coordinate_manager
        cx = x
        hx = self.conv_out(cx)
        strided_fea_list.append(hx)

        for idx in range(coder_num):
            cx = self.conv_c(cx)
            gates = self.conv_h(hx)
            forget_gate = torch.sigmoid(gates.F)
            cx = ME.SparseTensor(
                features=(forget_gate * cx.F),
                coordinate_map_key=cx.coordinate_map_key,
                coordinate_manager=cm
            )
            hx = self.conv_out(cx)
            strided_fea_list.append(hx)

        return strided_fea_list


class EncoderPartiallyRecurrent(nn.Module):
    """
    A simple wrapper of an Encoder object and an EncoderRecurrent object.
    """
    def __init__(self, encoder: Encoder, encoder_recurrent: EncoderRecurrent = None):
        super(EncoderPartiallyRecurrent, self).__init__()
        self.encoder_main = encoder
        self.encoder_recurrent = encoder_recurrent

    def forward(self, x: ME.SparseTensor, coder_num: int):
        strided_fea_list = self.encoder_main(x)[0]
        if self.encoder_recurrent is not None:
            strided_fea_list_2 = self.encoder_recurrent(
                strided_fea_list[-1], coder_num - len(strided_fea_list) + 1
            )
            strided_fea_list = strided_fea_list[:-1] + strided_fea_list_2
        return strided_fea_list


class HyperDecoderPartiallyRecurrent(nn.ModuleList):
    def __init__(self,
                 coder_type: str,
                 post_value_scaler: float,
                 in_channels: int,
                 out_channels: Tuple[int, ...],
                 intra_channels: Tuple[int, ...],
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],):
        modules = [HyperCoder(
            coder_type, post_value_scaler,
            in_channels, out_ch, (intra_ch, ), basic_block_type,
            region_type, basic_block_num, use_batch_norm, act
        ) for out_ch, intra_ch in zip(out_channels, intra_channels)]
        super(HyperDecoderPartiallyRecurrent, self).__init__(modules)

    def __getitem__(self, idx) -> nn.Module:
        if isinstance(idx, int) and idx >= len(self):
            idx = len(self) - 1
        return super(HyperDecoderPartiallyRecurrent, self).__getitem__(idx)


HyperDecoderGenUpsamplePartiallyRecurrent = \
    partial(HyperDecoderPartiallyRecurrent, 'generative_upsample_decoder')
HyperDecoderUpsamplePartiallyRecurrent = \
    partial(HyperDecoderPartiallyRecurrent, 'upsample_decoder')
