import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.entropy_models.rans_coder import BinaryRansCoder
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

        def make_em():
            return NoisyDeepFactorizedEntropyModel(
                batch_shape=torch.Size([compressed_channels]),
                coding_ndim=2,
                bottleneck_process=bottleneck_process,
                bottleneck_scaler=bottleneck_scaler,
                init_scale=5,
                broadcast_shape_bytes=(3,),
            )
        self.bottom_fea_entropy_model = make_em()
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
        *strided_fea_list, bottom_fea = strided_fea_list
        strided_fea_list_len = len(strided_fea_list)
        fea_bytes_list = []
        coord_bytes_list = []
        cm = y_top.coordinate_manager

        (bottom_fea_bytes,), coding_batch_shape, bottom_fea_recon = \
            self.bottom_fea_entropy_model.compress(bottom_fea)
        lower_fea_recon = bottom_fea_recon
        coord_recon_key = bottom_fea_recon.coordinate_map_key

        for idx in range(strided_fea_list_len - 1, -1, -1):
            sub_hyper_decoder_coord = self.hyper_decoder_coord[idx]
            sub_hyper_decoder_fea = self.hyper_decoder_fea[idx]
            sub_residual_block = self.residual_block[idx]
            sub_decoder_block = self.decoder_block[idx]
            fea = strided_fea_list[idx]

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
            del coord_target_key

            fea_pred = sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key)
            fea_res = sub_residual_block(fea, fea_pred)
            (fea_bytes,), coding_batch_shape, fea_res_recon = \
                self.bottom_fea_entropy_model.compress(fea_res)

            lower_fea_recon = sub_decoder_block(fea_res_recon, fea_pred)
            fea_bytes_list.append(fea_bytes)

        concat_bytes = BytesListUtils.concat_bytes_list(
            [*coord_bytes_list, bottom_fea_bytes, *fea_bytes_list]
        ) + len(coord_bytes_list).to_bytes(1, 'little', signed=False) + \
            len(fea_bytes_list).to_bytes(1, 'little', signed=False)
        return concat_bytes, bottom_fea_recon, lower_fea_recon

    def decompress(self, concat_bytes: bytes,
                   sparse_tensor_coords_tuple: Tuple[ME.CoordinateMapKey, ME.CoordinateManager]) \
            -> ME.SparseTensor:
        target_device = next(self.parameters()).device
        coord_bytes_list_len = int.from_bytes(concat_bytes[-2:-1], 'little', signed=False)
        fea_bytes_list_len = int.from_bytes(concat_bytes[-1:], 'little', signed=False)
        concat_bytes = concat_bytes[:-2]
        split_bytes = BytesListUtils.split_bytes_list(
            concat_bytes, coord_bytes_list_len + fea_bytes_list_len + 1
        )
        coord_bytes_list = split_bytes[: coord_bytes_list_len]
        bottom_fea_bytes = split_bytes[coord_bytes_list_len]
        fea_bytes_list = split_bytes[coord_bytes_list_len + 1:]
        cm = sparse_tensor_coords_tuple[1]

        bottom_fea_recon = self.bottom_fea_entropy_model.decompress(
            [bottom_fea_bytes], torch.Size([1]), target_device,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )
        lower_fea_recon = bottom_fea_recon
        coord_recon_key = lower_fea_recon.coordinate_map_key

        for idx in range(fea_bytes_list_len - 1, -1, -1):
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

            fea_bytes = fea_bytes_list.pop(0)
            fea_pred = sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key)
            fea_res_recon = self.bottom_fea_entropy_model.decompress(
                [fea_bytes], torch.Size([1]), target_device,
                sparse_tensor_coords_tuple=
                (fea_pred.coordinate_map_key, fea_pred.coordinate_manager)
            )
            lower_fea_recon = sub_decoder_block(fea_res_recon, fea_pred)
        assert len(coord_bytes_list) == len(fea_bytes_list) == 0
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
