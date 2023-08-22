import io
import math
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import MinkowskiEngine as ME

from lib.entropy_models.rans_coder import BinaryRansCoder, IndexedRansCoder
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel
from lib.entropy_models.hyperprior.noisy_deep_factorized.utils import BytesListUtils

from lib.torch_utils import concat_loss_dicts

RANDOM_DECODE_IDX = -1


class GeoLosslessEntropyModel(nn.Module):
    def __init__(self,
                 compressed_channels: int,
                 bottleneck_process: str,
                 bottleneck_scaler: int,
                 encoder: Union[nn.Module, nn.ModuleList],
                 residual_block: nn.ModuleList,
                 decoder_block: nn.ModuleList,

                 hyper_decoder_coord: nn.ModuleList,
                 hyper_decoder_fea: nn.ModuleList
                 ):
        super(GeoLosslessEntropyModel, self).__init__()
        self.compressed_channels = compressed_channels
        self.broadcast_shape_bytes = 3
        self.bottleneck_scaler = bottleneck_scaler

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
        self.hyper_decoder_coord = hyper_decoder_coord
        self.hyper_decoder_fea = hyper_decoder_fea
        self.binary_rans_coder = BinaryRansCoder(1)

    def rans_encode_with_cdf(self, target: np.ndarray, bs: io.BytesIO, offset: int = None):
        bs.write(int_to_bytes(target.size, self.broadcast_shape_bytes))
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

    def rans_decode_with_cdf(self, bs: io.BytesIO, dtype=torch.float, offset: int = None)\
            -> Tuple[torch.Tensor, List[int]]:
        shape_sum = bytes_to_int(bs.read(self.broadcast_shape_bytes))
        if offset is None:
            offset = -bytes_to_int(bs.read(1))
        cdf = [0, *(bytes_to_int(bs.read(2))
                    for _ in range(bytes_to_int(bs.read(1)))), 1 << 16]
        self.rans_coder.init_with_quantized_cdfs([cdf], np.array([offset], dtype=np.int32))

        bytes_len = bytes_to_int(bs.read(3))
        target_bytes = bs.read(bytes_len)
        target = np.empty((1, shape_sum * self.compressed_channels), np.int32)
        self.rans_coder.decode([target_bytes], target)

        target = torch.from_numpy(
            target.reshape(shape_sum, self.compressed_channels)
        ).to(dtype=dtype, device=next(self.parameters()).device)
        return target, cdf

    @staticmethod
    def init_prob(dist: torch.Tensor):
        return np.clip(
            np.round(dist.sigmoid().cpu().numpy().astype(np.float64) * (1 << 16)).astype(np.uint32),
            1, (1 << 16) - 1)

    def binary_encode(self, dist: torch.Tensor, x: torch.Tensor):
        assert dist.shape[0] == x.shape[0] and dist.shape[1] == 1
        (binary_data_bytes,) = self.binary_rans_coder.encode(
            x.cpu().numpy().reshape(1, -1),  # batch_size == 1
            self.init_prob(dist).reshape(1, -1))
        return binary_data_bytes

    def binary_decode(self, dist: torch.Tensor, binary_data_bytes):
        symbols = np.empty((1, dist.shape.numel()), dtype=bool)
        self.binary_rans_coder.decode(
            [binary_data_bytes],
            self.init_prob(dist).reshape(1, -1), symbols)
        symbols = torch.from_numpy(symbols).to(device=dist.device).reshape(dist.shape[0])
        return symbols

    def forward(self, y_top: ME.SparseTensor, *args, **kwargs):
        assert self.training
        cm = y_top.coordinate_manager
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
            sub_hyper_decoder_coord = self.hyper_decoder_coord[idx]
            sub_hyper_decoder_fea = self.hyper_decoder_fea[idx]
            sub_residual_block = self.residual_block[idx]
            sub_decoder_block = self.decoder_block[idx]
            fea = strided_fea_list[idx]

            coord_target_key = fea.coordinate_map_key
            if lower_fea_tilde.coordinate_map_key.get_tensor_stride() != coord_target_key.get_tensor_stride():
                coord_mask_pred = sub_hyper_decoder_coord(lower_fea_tilde)
                coord_mask = self.get_coord_mask(
                    cm, coord_target_key,
                    coord_mask_pred.coordinate_map_key, coord_mask_pred.C, torch.float
                )
                loss_dict[f'coord_{idx}_bits_loss'] = torch.nn.functional.binary_cross_entropy_with_logits(
                    coord_mask_pred.F.squeeze(1), coord_mask, reduction='sum'
                ) / math.log(2)

            fea_pred = sub_hyper_decoder_fea(lower_fea_tilde, coord_target_key)
            fea_res = sub_residual_block(fea, fea_pred)
            fea_res_tilde, fea_loss_dict = self.bottom_fea_entropy_model(fea_res)
            concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: f'fea_{idx}_' + k)

            lower_fea_tilde = sub_decoder_block(fea_res_tilde, fea_pred)
            strided_fea_tilde_list.append(lower_fea_tilde)

        return strided_fea_tilde_list[-1], loss_dict

    def compress(self, y_top: ME.SparseTensor, *args, **kwargs) -> bytes:
        strided_fea_list = self.encoder(y_top, *args, **kwargs)
        cm = y_top.coordinate_manager
        del y_top
        *strided_fea_list, bottom_fea_recon = strided_fea_list
        strided_fea_list_len = len(strided_fea_list)
        fea_res_list = []
        coord_bytes_list = []
        bottom_fea_recon_f = bottom_fea_recon.F
        bottom_fea_recon_f *= self.bottleneck_scaler
        bottom_fea_recon_f.round_()
        fea_res_list.append(bottom_fea_recon_f.cpu().to(torch.int32).numpy())
        bottom_fea_recon_f /= self.bottleneck_scaler
        lower_fea_recon = bottom_fea_recon
        coord_recon_key = lower_fea_recon.coordinate_map_key

        for idx in range(strided_fea_list_len - 1, -1, -1):
            sub_hyper_decoder_coord = self.hyper_decoder_coord[idx]
            sub_hyper_decoder_fea = self.hyper_decoder_fea[idx]
            sub_residual_block = self.residual_block[idx]
            sub_decoder_block = self.decoder_block[idx]
            fea = strided_fea_list[idx]
            strided_fea_list[idx] = None

            coord_target_key = fea.coordinate_map_key
            if coord_recon_key.get_tensor_stride() != coord_target_key.get_tensor_stride():
                coord_mask_pred = sub_hyper_decoder_coord(lower_fea_recon)
                coord_mask = self.get_coord_mask(
                    cm, coord_target_key,
                    coord_mask_pred.coordinate_map_key, coord_mask_pred.C, torch.bool
                )
                coord_bytes = self.binary_encode(coord_mask_pred.F, coord_mask)
                coord_bytes_list.append(coord_bytes)
                coord_recon_key = cm.insert_and_map(
                    coord_mask_pred.C[coord_mask],
                    coord_mask_pred.tensor_stride,
                    coord_mask_pred.coordinate_map_key.get_key()[1] + 'pruned'
                )[0]
                del coord_mask, coord_mask_pred
            else:
                assert sub_hyper_decoder_coord is None

            permutation_kernel_map = cm.kernel_map(
                coord_target_key,
                coord_recon_key,
                kernel_size=1)[0][0].to(torch.long)
            fea = ME.SparseTensor(
                features=fea.F[permutation_kernel_map],
                coordinate_map_key=coord_recon_key,
                coordinate_manager=cm
            )
            del permutation_kernel_map, coord_target_key

            fea_pred = sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key)
            fea_res = sub_residual_block(fea, fea_pred).F
            del fea
            fea_res *= self.bottleneck_scaler
            fea_res.round_()
            fea_res_list.append(fea_res.cpu().to(torch.int32).numpy())

            fea_res /= self.bottleneck_scaler
            lower_fea_recon = sub_decoder_block(fea_res, fea_pred)
            del fea_pred
        del lower_fea_recon

        with io.BytesIO() as bs:
            bottom_stride = bottom_fea_recon.tensor_stride[0]
            bs.write(int_to_bytes(int(math.log2(bottom_stride)), 1))
            bs.write(int_to_bytes(bottom_fea_recon.shape[0], self.broadcast_shape_bytes))

            fea_res_concat = np.concatenate(fea_res_list, axis=0)
            del fea_res_list
            self.rans_encode_with_cdf(fea_res_concat, bs)

            bs.write(int_to_bytes(len(coord_bytes_list), 1))
            BytesListUtils.concat_bytes_list(coord_bytes_list, bs)

            self.rans_encode_with_cdf(
                (bottom_fea_recon.C[:, 1:] // bottom_stride).cpu().to(torch.int32).numpy(), bs, 0)

            concat_bytes = bs.getvalue()
        return concat_bytes

    def decompress(self, concat_bytes: bytes, cm: ME.CoordinateManager) -> ME.SparseTensor:
        with io.BytesIO(concat_bytes) as bs:
            bottom_stride = 2 ** bytes_to_int(bs.read(1))
            bottom_fea_recon_shape = bytes_to_int(bs.read(self.broadcast_shape_bytes))

            fea_res_concat, fea_res_cdf = self.rans_decode_with_cdf(bs)
            fea_res_concat = fea_res_concat.reshape(-1, self.compressed_channels)
            fea_res_concat /= self.bottleneck_scaler
            coord_bytes_list_len = bytes_to_int(bs.read(1))
            coord_bytes_list = BytesListUtils.split_bytes_list(
                None, coord_bytes_list_len, bs
            )

            bottom_fea_recon, fea_res_concat = torch.split(
                fea_res_concat,
                [bottom_fea_recon_shape, fea_res_concat.shape[0] - bottom_fea_recon_shape], dim=0)
            bottom_coord = self.rans_decode_with_cdf(bs, torch.int32, 0)[0].reshape(-1, 3) * bottom_stride

        lower_fea_recon = ME.SparseTensor(
            bottom_fea_recon, torch.cat((
                torch.zeros((bottom_coord.shape[0], 1),
                            device=bottom_coord.device, dtype=bottom_coord.dtype),
                bottom_coord), 1),
            tensor_stride=[bottom_stride] * 3, coordinate_manager=cm
        )
        coord_recon_key = lower_fea_recon.coordinate_map_key
        cur_voxels_num = lower_fea_recon.shape[0]

        fea_res_cdf = torch.tensor(fea_res_cdf, dtype=torch.int32, device=lower_fea_recon.device)
        fea_res_pmf = fea_res_cdf[1:] - fea_res_cdf[:-1]
        fea_res_dist = dist.Categorical(fea_res_pmf / fea_res_pmf.sum())

        for idx in range(len(self.residual_block) - 1, -1, -1):  # extra length due to bottom fea
            sub_hyper_decoder_coord = self.hyper_decoder_coord[idx]
            sub_hyper_decoder_fea = self.hyper_decoder_fea[idx]
            sub_decoder_block = self.decoder_block[idx]
            if sub_hyper_decoder_coord is None:
                assert coord_recon_key == lower_fea_recon.coordinate_map_key
            else:
                coord_bytes = coord_bytes_list.pop(0)
                coord_mask_pred = sub_hyper_decoder_coord(lower_fea_recon)
                cur_stride = coord_mask_pred.tensor_stride
                coord_recon_key_id = coord_mask_pred.coordinate_map_key.get_key()[1] + 'pruned'
                if idx > RANDOM_DECODE_IDX:
                    coord_mask = self.binary_decode(coord_mask_pred.F, coord_bytes)
                else:
                    coord_mask = dist.Bernoulli(coord_mask_pred.F.reshape(-1).sigmoid()).sample().bool().reshape(-1)
                coord_mask_true = coord_mask_pred.C[coord_mask]
                del coord_mask, coord_mask_pred
                cur_voxels_num = coord_mask_true.shape[0]
                coord_recon_key = cm.insert_and_map(
                    coord_mask_true, cur_stride, coord_recon_key_id
                )[0]
                del coord_mask_true

            if idx > RANDOM_DECODE_IDX:
                cur_fea_recon, fea_res_concat = torch.split(
                    fea_res_concat, [cur_voxels_num, fea_res_concat.shape[0] - cur_voxels_num], dim=0)
            else:
                cur_fea_recon = fea_res_dist.sample(torch.Size((cur_voxels_num, 1))).to(torch.float)
            lower_fea_recon = sub_decoder_block(
                cur_fea_recon,
                sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key)
            )
            del cur_fea_recon
        assert len(coord_bytes_list) == 0
        if RANDOM_DECODE_IDX < 0:
            assert fea_res_concat.shape[0] == 0
        return lower_fea_recon

    @staticmethod
    def get_coord_mask(
            cm: ME.CoordinateManager,
            coord_target_key: ME.CoordinateMapKey,
            current_key: ME.CoordinateMapKey,
            current_coord: torch.Tensor,
            output_dtype) -> torch.Tensor:
        kernel_map = cm.kernel_map(current_key, coord_target_key, kernel_size=1)
        keep_target = torch.zeros(current_coord.shape[0], dtype=output_dtype, device=current_coord.device)
        for _, curr_in in kernel_map.items():
            keep_target[curr_in[0].type(torch.long)] = 1
        return keep_target


def int_to_bytes(x, length, byteorder='little', signed=False):
    assert isinstance(x, int)
    return x.to_bytes(length, byteorder=byteorder, signed=signed)


def bytes_to_int(s, byteorder='little', signed=False):
    assert isinstance(s, bytes)
    return int.from_bytes(s, byteorder=byteorder, signed=signed)

