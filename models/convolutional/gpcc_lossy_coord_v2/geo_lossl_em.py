import os
import io
import math
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.mpeg_gpcc_utils import gpcc_octree_lossless_geom_encode
from lib.data_utils import write_ply_file
from lib.entropy_models.rans_coder import BinaryRansCoder, IndexedRansCoder
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel

from lib.torch_utils import concat_loss_dicts


class GeoLosslessEntropyModel(nn.Module):
    def __init__(self,
                 compressed_channels: int,
                 bottleneck_process: str,
                 bottleneck_scaler: int,
                 skip_encoding_fea: int,
                 encoder: Union[nn.Module, nn.ModuleList],
                 residual_block: nn.ModuleList,
                 decoder_block: nn.ModuleList,
                 hyper_decoder_fea: nn.ModuleList
                 ):
        super(GeoLosslessEntropyModel, self).__init__()
        self.compressed_channels = compressed_channels
        self.broadcast_shape_bytes = 3
        self.bottleneck_scaler = bottleneck_scaler
        self.skip_encoding_fea = skip_encoding_fea

        def make_em():
            return NoisyDeepFactorizedEntropyModel(
                batch_shape=torch.Size([compressed_channels]),
                coding_ndim=2,
                bottleneck_process=bottleneck_process,
                bottleneck_scaler=bottleneck_scaler,
                init_scale=10,
                broadcast_shape_bytes=(self.broadcast_shape_bytes,),
            )
        self.bottom_fea_entropy_model = make_em()
        self.rans_coder = IndexedRansCoder(False, 1)
        assert len(encoder) == len(residual_block) == \
               len(decoder_block) == len(hyper_decoder_fea)
        self.encoder = encoder
        self.residual_block = residual_block
        self.decoder_block = decoder_block
        self.hyper_decoder_fea = hyper_decoder_fea
        self.binary_rans_coder = BinaryRansCoder(1)

    def rans_encode_with_cdf(self, target: np.ndarray, bs: io.BytesIO, offset: int = None):
        bs.write(int_to_bytes(target.shape[0], self.broadcast_shape_bytes))
        if offset is None:
            offset = target.min().item()
            bs.write(int_to_bytes(-offset, 1))

        pmf = np.bincount((target - offset).reshape(-1))
        self.rans_coder.init_with_pmfs(pmf[None], np.array([offset], dtype=np.int32))
        cdf = self.rans_coder.get_cdfs()[0]
        bs.write(int_to_bytes(len(cdf) - 2, 1))
        for cd in cdf[1:-1]:
            bs.write(int_to_bytes(cd, 2))

        target_bytes = self.rans_coder.encode(target.reshape(1, -1))[0]
        bs.write(int_to_bytes(len(target_bytes), 3))
        bs.write(target_bytes)

    def rans_decode_with_cdf(self, bs: io.BytesIO, dtype=torch.float, offset: int = None, channels: int = None)\
            -> Tuple[torch.Tensor, List[int]]:
        shape_sum = bytes_to_int(bs.read(self.broadcast_shape_bytes))
        if offset is None:
            offset = -bytes_to_int(bs.read(1))
        cdf = [0, *(bytes_to_int(bs.read(2))
                    for _ in range(bytes_to_int(bs.read(1)))), 1 << 16]
        self.rans_coder.init_with_quantized_cdfs([cdf], np.array([offset], dtype=np.int32))

        bytes_len = bytes_to_int(bs.read(3))
        target_bytes = bs.read(bytes_len)
        target = np.empty((1, shape_sum * (channels or self.compressed_channels)), np.int32)
        self.rans_coder.decode([target_bytes], target)

        target = torch.from_numpy(
            target.reshape(shape_sum, (channels or self.compressed_channels))
        ).to(dtype=dtype, device=next(self.parameters()).device)
        return target, cdf

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(*args, **kwargs)
        else:
            return self.test_forward(*args, **kwargs)

    def train_forward(self, y_top: ME.SparseTensor, *args, **kwargs):
        assert self.training
        strided_fea_list = self.encoder(y_top, *args, **kwargs)
        *strided_fea_list, bottom_fea = strided_fea_list
        strided_fea_list_len = len(strided_fea_list)
        loss_dict = {}
        strided_fea_tilde_list = []

        bottom_fea_tilde, fea_loss_dict = self.bottom_fea_entropy_model(bottom_fea)
        strided_fea_tilde_list.append(bottom_fea_tilde)
        concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: 'fea_bottom_' + k)
        lower_fea_tilde = bottom_fea_tilde

        for idx in range(strided_fea_list_len - 1, -1, -1):
            sub_hyper_decoder_fea = self.hyper_decoder_fea[idx]
            sub_residual_block = self.residual_block[idx]
            sub_decoder_block = self.decoder_block[idx]
            fea = strided_fea_list[idx]
            coord_target_key = fea.coordinate_map_key

            fea_pred = sub_hyper_decoder_fea(lower_fea_tilde, coord_target_key)
            if idx > self.skip_encoding_fea:
                fea_res = sub_residual_block(fea, fea_pred)
                fea_res_tilde, fea_loss_dict = self.bottom_fea_entropy_model(fea_res)
                concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: f'fea_{idx}_' + k)
                lower_fea_tilde = sub_decoder_block(fea_res_tilde, fea_pred)
            else:
                lower_fea_tilde = sub_decoder_block(fea_pred)
            strided_fea_tilde_list.append(lower_fea_tilde)

        return strided_fea_tilde_list[-1], loss_dict

    def test_forward(self, y_top: ME.SparseTensor, *args, **kwargs) \
            -> Tuple[ME.SparseTensor, bytes]:
        strided_fea_list = self.encoder(y_top, *args, **kwargs)
        del y_top
        *strided_fea_list, bottom_fea_recon = strided_fea_list
        strided_fea_list_len = len(strided_fea_list)
        fea_res_list = []
        bottom_fea_recon_f = bottom_fea_recon.F
        bottom_fea_recon_f *= self.bottleneck_scaler
        bottom_fea_recon_f.round_()
        fea_res_list.append(bottom_fea_recon_f.cpu().to(torch.int32).numpy())
        bottom_fea_recon_f /= self.bottleneck_scaler
        lower_fea_recon = bottom_fea_recon

        for idx in range(strided_fea_list_len - 1, -1, -1):
            sub_hyper_decoder_fea = self.hyper_decoder_fea[idx]
            sub_residual_block = self.residual_block[idx]
            sub_decoder_block = self.decoder_block[idx]
            fea = strided_fea_list[idx]
            strided_fea_list[idx] = None
            coord_target_key = fea.coordinate_map_key

            fea_pred = sub_hyper_decoder_fea(lower_fea_recon, coord_target_key)
            if idx > self.skip_encoding_fea:
                fea_res = sub_residual_block(fea, fea_pred).F
                del fea
                fea_res *= self.bottleneck_scaler
                fea_res.round_()
                fea_res_list.append(fea_res.cpu().to(torch.int32).numpy())

                fea_res /= self.bottleneck_scaler
                lower_fea_recon = sub_decoder_block(fea_res, fea_pred)
            else:
                lower_fea_recon = sub_decoder_block(fea_pred)
            del fea_pred

        with io.BytesIO() as bs:
            bottom_stride = bottom_fea_recon.tensor_stride[0]
            bs.write(int_to_bytes(int(math.log2(bottom_stride)), 1))
            bs.write(int_to_bytes(bottom_fea_recon.shape[0], self.broadcast_shape_bytes))

            fea_res_concat = np.concatenate(fea_res_list, axis=0)
            del fea_res_list
            self.rans_encode_with_cdf(fea_res_concat, bs)

            tmp_file_path = f'tmp-{torch.rand(1).item()}'
            write_ply_file(lower_fea_recon.C[:, 1:] // lower_fea_recon.tensor_stride[0], f'{tmp_file_path}.ply')
            gpcc_octree_lossless_geom_encode(
                f'{tmp_file_path}.ply', f'{tmp_file_path}.bin'
            )
            with open(f'{tmp_file_path}.bin', 'rb') as f:
                sparse_tensor_coords_bytes = f.read()
            os.remove(f'{tmp_file_path}.ply')
            os.remove(f'{tmp_file_path}.bin')

            concat_bytes = bs.getvalue() + sparse_tensor_coords_bytes
        return lower_fea_recon, concat_bytes


def int_to_bytes(x, length, byteorder='little', signed=False):
    assert isinstance(x, int)
    return x.to_bytes(length, byteorder=byteorder, signed=signed)


def bytes_to_int(s, byteorder='little', signed=False):
    assert isinstance(s, bytes)
    return int.from_bytes(s, byteorder=byteorder, signed=signed)

