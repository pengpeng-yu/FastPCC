import io
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.entropy_models.rans_coder import BinaryRansCoder, IndexedRansCoder
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel
from lib.entropy_models.hyperprior.noisy_deep_factorized.utils import BytesListUtils

from lib.torch_utils import concat_loss_dicts


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
                init_scale=5,
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

    def compress(self, y_top: ME.SparseTensor, *args, **kwargs) -> \
            Tuple[bytes, ME.SparseTensor, ME.SparseTensor]:
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

        with io.BytesIO() as bs:
            bs.write(int_to_bytes(len(fea_res_list), 2))
            for fea_res in fea_res_list:
                bs.write(int_to_bytes(fea_res.shape[0], self.broadcast_shape_bytes))

            fea_res_concat = np.concatenate(fea_res_list, axis=0)
            del fea_res_list
            fea_res_offset = fea_res_concat.min()
            fea_res_pm = np.histogram(
                fea_res_concat,
                range(fea_res_offset.item(), fea_res_concat.max().item() + 2))[0]
            self.rans_coder.init_with_pmfs(fea_res_pm[None], fea_res_offset[None])
            fea_res_cdf = self.rans_coder.get_cdfs()[0]
            fea_bytes = self.rans_coder.encode(fea_res_concat.reshape(1, -1))[0]

            bs.write(int_to_bytes(len(fea_res_cdf) - 2, 1))
            for cd in fea_res_cdf[1:-1]:
                bs.write(int_to_bytes(cd, 2))
            bs.write(int_to_bytes(-fea_res_offset.item(), 1))

            bs.write(int_to_bytes(len(coord_bytes_list), 1))
            bs.write(BytesListUtils.concat_bytes_list(
                [*coord_bytes_list, fea_bytes]
            ))
            concat_bytes = bs.getvalue()
        return concat_bytes, bottom_fea_recon, lower_fea_recon

    def decompress(self, concat_bytes: bytes,
                   sparse_tensor_coords_tuple: Tuple[ME.CoordinateMapKey, ME.CoordinateManager]) \
            -> ME.SparseTensor:
        target_device = next(self.parameters()).device
        cm = sparse_tensor_coords_tuple[1]

        with io.BytesIO(concat_bytes) as bs:
            strided_fea_list_len = bytes_to_int(bs.read(2))
            fea_res_shape_list = [bytes_to_int(bs.read(self.broadcast_shape_bytes))
                                  for _ in range(strided_fea_list_len)]
            fea_res_cdf = [0, *(bytes_to_int(bs.read(2))
                                for _ in range(bytes_to_int(bs.read(1)))), 1 << 16]
            fea_res_offset = -bytes_to_int(bs.read(1))
            coord_bytes_list_len = bytes_to_int(bs.read(1))
            *coord_bytes_list, fea_bytes = BytesListUtils.split_bytes_list(
                bs.read(), coord_bytes_list_len + 1
            )

            self.rans_coder.init_with_quantized_cdfs([fea_res_cdf], np.array([fea_res_offset]))
            fea_res_concat = np.empty((1, sum(fea_res_shape_list) * self.compressed_channels), np.int32)
            self.rans_coder.decode([fea_bytes], fea_res_concat)
            fea_res_concat = torch.from_numpy(
                fea_res_concat.reshape(sum(fea_res_shape_list), self.compressed_channels)
            ).to(dtype=torch.float, device=target_device)
            fea_res_concat /= self.bottleneck_scaler
            fea_res_list = list(reversed(torch.split(fea_res_concat, fea_res_shape_list, dim=0)))

        lower_fea_recon = ME.SparseTensor(
            fea_res_list.pop(),
            coordinate_map_key=sparse_tensor_coords_tuple[0],
            coordinate_manager=sparse_tensor_coords_tuple[1]
        )
        coord_recon_key = lower_fea_recon.coordinate_map_key

        for idx in range(strided_fea_list_len - 2, -1, -1):  # extra length due to bottom fea
            sub_hyper_decoder_coord = self.hyper_decoder_coord[idx]
            sub_hyper_decoder_fea = self.hyper_decoder_fea[idx]
            sub_decoder_block = self.decoder_block[idx]
            if sub_hyper_decoder_coord is None:
                assert coord_recon_key == lower_fea_recon.coordinate_map_key
            else:
                coord_bytes = coord_bytes_list.pop(0)
                coord_mask_pred = sub_hyper_decoder_coord(lower_fea_recon)
                coord_mask = self.binary_decode(coord_mask_pred.F, coord_bytes)
                coord_recon_key = cm.insert_and_map(
                    coord_mask_pred.C[coord_mask],
                    coord_mask_pred.tensor_stride,
                    coord_mask_pred.coordinate_map_key.get_key()[1] + 'pruned'
                )[0]
                del coord_mask, coord_mask_pred

            lower_fea_recon = sub_decoder_block(
                fea_res_list.pop(),
                sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key)
            )
        assert len(coord_bytes_list) == len(fea_res_list) == 0
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

