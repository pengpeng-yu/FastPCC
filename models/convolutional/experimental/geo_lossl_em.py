from typing import Tuple, Union, Dict, Any, Callable
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Distribution
import MinkowskiEngine as ME

from lib.entropy_models.hyperprior.noisy_deep_factorized.basic import EntropyModel as HyperPriorEntropyModel
from lib.entropy_models.continuous_batched import ContinuousBatchedEntropyModel
from lib.entropy_models.continuous_indexed import ContinuousIndexedEntropyModel,\
    noisy_scale_normal_indexed_entropy_model_init, \
    noisy_deep_factorized_indexed_entropy_model_init
from lib.entropy_models.distributions.uniform_noise import NoisyNormal, NoisyDeepFactorized
from lib.entropy_models.hyperprior.noisy_deep_factorized.utils import BytesListUtils

from lib.torch_utils import concat_loss_dicts


class GeoLosslessEntropyModel(nn.Module):
    """
    Note:
        For lossless geometric compression.
        Only supports batch size == 1 during testing.
    """
    def __init__(self,
                 bottom_fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,

                 hyper_decoder_coord: Union[nn.Module, nn.ModuleList],
                 hyper_decoder_fea: Union[nn.Module, nn.ModuleList],
                 hybrid_hyper_decoder_fea: bool,

                 coord_prior_fn: Callable[..., Distribution],
                 coord_index_ranges: Tuple[int, ...],
                 coord_parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],
                 fea_prior_fn: Callable[..., Distribution],
                 fea_index_ranges: Tuple[int, ...],
                 fea_parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],

                 hyper_decoder_coord_post_op: Callable = lambda x: x,
                 hyper_decoder_fea_post_op: Callable = lambda x: x,

                 skip_encoding_fea: Tuple[int] = (),
                 upper_fea_grad_scaler_for_bits_loss: float = 1.0,
                 bottleneck_fea_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 init_scale: float = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        super(GeoLosslessEntropyModel, self).__init__()
        self.bottom_fea_entropy_model = bottom_fea_entropy_model
        self.encoder = encoder
        self.hyper_decoder_coord = hyper_decoder_coord
        self.hyper_decoder_fea = hyper_decoder_fea
        self.hybrid_hyper_decoder_fea = hybrid_hyper_decoder_fea
        self.hyper_decoder_coord_post_op = hyper_decoder_coord_post_op
        self.hyper_decoder_fea_post_op = hyper_decoder_fea_post_op
        self.skip_encoding_fea = skip_encoding_fea
        self.upper_fea_grad_scaler_for_bits_loss = upper_fea_grad_scaler_for_bits_loss
        assert all((_ >= 0 for _ in self.skip_encoding_fea))
        self.indexed_entropy_model_coord = ContinuousIndexedEntropyModel(
            prior_fn=coord_prior_fn,
            index_ranges=coord_index_ranges,
            parameter_fns=coord_parameter_fns,
            coding_ndim=2,
            bottleneck_process='',
            quantize_bottleneck_in_eval=False,
            indexes_bound_gradient=indexes_bound_gradient,
            quantize_indexes=quantize_indexes,
            indexes_scaler=indexes_scaler,
            lower_bound=0,
            upper_bound=1,
            range_coder_precision=range_coder_precision,
            overflow_coding=False
        )
        self.indexed_entropy_model_fea = ContinuousIndexedEntropyModel(
            prior_fn=fea_prior_fn,
            index_ranges=fea_index_ranges,
            parameter_fns=fea_parameter_fns,
            coding_ndim=2,
            bottleneck_process=bottleneck_fea_process,
            bottleneck_scaler=bottleneck_scaler,
            indexes_bound_gradient=indexes_bound_gradient,
            quantize_indexes=quantize_indexes,
            indexes_scaler=indexes_scaler,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def get_sub_hyper_decoder_coord(self, idx):
        if hasattr(self.hyper_decoder_coord, '__getitem__'):
            return self.hyper_decoder_coord[idx]
        else:
            return self.hyper_decoder_coord

    def get_sub_hyper_decoder_fea(self, idx):
        if hasattr(self.hyper_decoder_fea, '__getitem__'):
            return self.hyper_decoder_fea[idx]
        else:
            return self.hyper_decoder_fea

    def forward(self, y_top: ME.SparseTensor, *args, **kwargs):
        if self.training:
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

            skipped_fea_num = 0
            for idx in range(strided_fea_list_len - 1, -1, -1):
                sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(idx)
                sub_hyper_decoder_fea = self.get_sub_hyper_decoder_fea(idx)
                fea = strided_fea_list[idx]

                coord_target_key = fea.coordinate_map_key
                # Same strides indicate same coordinates.
                if lower_fea_tilde.coordinate_map_key.get_tensor_stride() != coord_target_key.get_tensor_stride():
                    pre_coord_mask_indexes = sub_hyper_decoder_coord(lower_fea_tilde)
                    coord_mask_indexes = self.hyper_decoder_coord_post_op(pre_coord_mask_indexes)
                    coord_mask = self.get_coord_mask(
                        cm, coord_target_key,
                        pre_coord_mask_indexes.coordinate_map_key, pre_coord_mask_indexes.C, torch.float
                    )
                    coord_mask_f_, coord_loss_dict = self.indexed_entropy_model_coord(
                        coord_mask[None, :, None], coord_mask_indexes,
                        is_first_forward=idx == strided_fea_list_len - 1
                    )
                    concat_loss_dicts(loss_dict, coord_loss_dict, lambda k: f'coord_{idx}_' + k)
                else:
                    assert sub_hyper_decoder_coord is None

                if idx in self.skip_encoding_fea:
                    fea_loss_dict = {}
                    skipped_fea_num += 1
                    if idx == 0 and sub_hyper_decoder_fea is None:
                        lower_fea_tilde = None
                    else:
                        lower_fea_tilde = ME.SparseTensor(
                            features=sub_hyper_decoder_fea(lower_fea_tilde, coord_target_key).F[:, :fea.F.shape[1]],
                            coordinate_map_key=coord_target_key,
                            coordinate_manager=cm
                        )
                elif self.hybrid_hyper_decoder_fea is True:  # no_skip
                    fea_info_pred = sub_hyper_decoder_fea(lower_fea_tilde, coord_target_key).F
                    assert fea.F.shape[1] * (
                        len(self.indexed_entropy_model_fea.index_ranges) + 1
                    ) == fea_info_pred.shape[1]
                    fea_pred, pre_fea_indexes = torch.split(
                        fea_info_pred,
                        [fea.F.shape[1], fea_info_pred.shape[1] - fea.F.shape[1]], dim=1
                    )
                    fea_indexes = self.hyper_decoder_fea_post_op(pre_fea_indexes)[None]
                    fea_pred_res_tilde, fea_loss_dict = self.indexed_entropy_model_fea(
                        (fea.F - fea_pred)[None], fea_indexes, is_first_forward=idx == strided_fea_list_len - 1,
                        x_grad_scaler_for_bits_loss=self.upper_fea_grad_scaler_for_bits_loss
                    )
                    lower_fea_tilde = ME.SparseTensor(
                        features=fea_pred_res_tilde[0] + fea_pred,
                        coordinate_map_key=coord_target_key,
                        coordinate_manager=cm
                    )
                else:
                    fea_indexes = self.hyper_decoder_fea_post_op(
                        sub_hyper_decoder_fea(lower_fea_tilde, coord_target_key)
                    )
                    fea_tilde, fea_loss_dict = self.indexed_entropy_model_fea(
                        fea.F[None], fea_indexes, is_first_forward=idx == strided_fea_list_len - 1,
                        x_grad_scaler_for_bits_loss=self.upper_fea_grad_scaler_for_bits_loss
                    )
                    lower_fea_tilde = ME.SparseTensor(
                        features=fea_tilde[0],
                        coordinate_map_key=coord_target_key,
                        coordinate_manager=cm
                    )
                strided_fea_tilde_list.append(lower_fea_tilde)
                concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: f'fea_{idx}_' + k)
            assert skipped_fea_num == len(self.skip_encoding_fea)

            return strided_fea_tilde_list[-1], loss_dict

        else:
            concat_bytes, bottom_fea_recon, fea_recon_ = self.compress(y_top, *args, **kwargs)
            # You can clear the shared coordinate manager in an upper module
            # after compression to save memory.
            fea_recon = self.decompress(
                concat_bytes,
                (bottom_fea_recon.coordinate_map_key,
                 bottom_fea_recon.coordinate_manager)
            )
            return bottom_fea_recon, fea_recon, concat_bytes

    def compress(self, y_top: ME.SparseTensor, *args, **kwargs) -> \
            Tuple[bytes, ME.SparseTensor, ME.SparseTensor]:
        # Batch dimension of sparse tensor feature is supposed to
        # be added in minkowski_tensor_wrapped_fn(),
        # thus all the inputs of entropy models are supposed to
        # have batch size == 1.
        # TODO: Call each em.compress() only once?

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
            sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(idx)
            sub_hyper_decoder_fea = self.get_sub_hyper_decoder_fea(idx)
            fea = strided_fea_list[idx]

            coord_target_key = fea.coordinate_map_key
            if coord_recon_key.get_tensor_stride() != coord_target_key.get_tensor_stride():
                pre_coord_mask_indexes = sub_hyper_decoder_coord(lower_fea_recon)
                coord_mask_indexes = self.hyper_decoder_coord_post_op(pre_coord_mask_indexes)
                coord_mask = self.get_coord_mask(
                    cm, coord_target_key,
                    pre_coord_mask_indexes.coordinate_map_key, pre_coord_mask_indexes.C, torch.bool
                )
                (coord_bytes,), coord_mask_f_ = self.indexed_entropy_model_coord.compress(
                    coord_mask[None, :, None], coord_mask_indexes
                )
                coord_bytes_list.append(coord_bytes)
                coord_recon = pre_coord_mask_indexes.C[coord_mask]
                coord_recon_key = cm.insert_and_map(
                    coord_recon, pre_coord_mask_indexes.tensor_stride,
                    pre_coord_mask_indexes.coordinate_map_key.get_key()[1] + 'pruned'
                )[0]
            else:
                assert sub_hyper_decoder_coord is None

            if idx in self.skip_encoding_fea:
                if idx == 0 and sub_hyper_decoder_fea is None:
                    lower_fea_recon = None
                else:
                    lower_fea_recon = ME.SparseTensor(
                        features=sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key).F[:, :fea.F.shape[1]],
                        coordinate_map_key=coord_recon_key,
                        coordinate_manager=cm
                    )
            else:  # no_skip
                # Permute features from encoders to fit in with order of features from decoders.
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

                if self.hybrid_hyper_decoder_fea is True:
                    fea_info_pred = sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key).F
                    assert fea.F.shape[1] * (
                        len(self.indexed_entropy_model_fea.index_ranges) + 1
                    ) == fea_info_pred.shape[1]
                    fea_pred, pre_fea_indexes = torch.split(
                        fea_info_pred,
                        [fea.F.shape[1], fea_info_pred.shape[1] - fea.F.shape[1]], dim=1
                    )
                    fea_indexes = self.hyper_decoder_fea_post_op(pre_fea_indexes)[None]
                    (fea_bytes,), fea_pred_res_recon = self.indexed_entropy_model_fea.compress(
                        (fea.F - fea_pred)[None], fea_indexes,
                    )
                    lower_fea_recon = ME.SparseTensor(
                        features=fea_pred_res_recon[0] + fea_pred,
                        coordinate_map_key=coord_recon_key,
                        coordinate_manager=cm
                    )
                else:
                    fea_indexes = self.hyper_decoder_fea_post_op(
                        sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key)
                    )
                    (fea_bytes,), fea_recon = self.indexed_entropy_model_fea.compress(
                        fea.F[None], fea_indexes
                    )
                    lower_fea_recon = ME.SparseTensor(
                        features=fea_recon[0],
                        coordinate_map_key=coord_recon_key,
                        coordinate_manager=cm
                    )
                fea_bytes_list.append(fea_bytes)

        assert strided_fea_list_len - len(self.skip_encoding_fea) == len(fea_bytes_list)
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
        strided_fea_list_len = fea_bytes_list_len + len(self.skip_encoding_fea)
        strided_fea_recon_list = []
        cm = sparse_tensor_coords_tuple[1]

        bottom_fea_recon = self.bottom_fea_entropy_model.decompress(
            [bottom_fea_bytes], torch.Size([1]), target_device,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )
        lower_fea_recon = bottom_fea_recon
        coord_recon_key = lower_fea_recon.coordinate_map_key

        for idx in range(strided_fea_list_len - 1, -1, -1):
            sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(idx)
            sub_hyper_decoder_fea = self.get_sub_hyper_decoder_fea(idx)
            if sub_hyper_decoder_coord is None:
                assert coord_recon_key == lower_fea_recon.coordinate_map_key
                coord_recon = lower_fea_recon
            else:
                coord_bytes = coord_bytes_list.pop(0)
                coord_mask_pre_indexes = sub_hyper_decoder_coord(lower_fea_recon)
                coord_mask_indexes = self.hyper_decoder_coord_post_op(coord_mask_pre_indexes)
                coord_mask: ME.SparseTensor = self.indexed_entropy_model_coord.decompress(
                    [coord_bytes], coord_mask_indexes, target_device,
                    sparse_tensor_coords_tuple=(coord_mask_pre_indexes.coordinate_map_key, cm)
                )
                coord_recon = coord_mask.C[coord_mask.F.to(torch.bool)[:, 0]]
                coord_recon_key = cm.insert_and_map(
                    coord_recon, coord_mask_pre_indexes.tensor_stride,
                    coord_mask_pre_indexes.coordinate_map_key.get_key()[1] + 'pruned'
                )[0]

            if idx in self.skip_encoding_fea:
                if idx == 0 and sub_hyper_decoder_fea is None:
                    lower_fea_recon = ME.SparseTensor(
                        features=torch.ones(
                            (coord_recon.shape[0], 1),
                            dtype=bottom_fea_recon.F.dtype, device=target_device
                        ),
                        coordinate_map_key=coord_recon_key,
                        coordinate_manager=cm
                    )
                else:
                    fea_info_pred = sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key).F
                    fea_recon_channels = fea_info_pred.shape[1] // (
                            len(self.indexed_entropy_model_fea.index_ranges) + 1
                    )
                    lower_fea_recon = ME.SparseTensor(
                        features=fea_info_pred[:, :fea_recon_channels],
                        coordinate_map_key=coord_recon_key,
                        coordinate_manager=cm
                    )
            else:
                fea_bytes = fea_bytes_list.pop(0)
                if self.hybrid_hyper_decoder_fea is True:  # no_skip
                    fea_info_pred = sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key).F
                    fea_recon_channels = fea_info_pred.shape[1] // (
                        len(self.indexed_entropy_model_fea.index_ranges) + 1
                    )
                    fea_pred, pre_fea_indexes = torch.split(
                        fea_info_pred,
                        [fea_recon_channels, fea_info_pred.shape[1] - fea_recon_channels],
                        dim=1
                    )
                    fea_indexes = self.hyper_decoder_fea_post_op(pre_fea_indexes)[None]
                    lower_fea_pred_res_recon = self.indexed_entropy_model_fea.decompress(
                        [fea_bytes], fea_indexes, target_device
                    )
                    lower_fea_recon = ME.SparseTensor(
                        features=lower_fea_pred_res_recon[0] + fea_pred,
                        coordinate_map_key=coord_recon_key,
                        coordinate_manager=cm
                    )
                else:
                    fea_indexes = self.hyper_decoder_fea_post_op(
                        sub_hyper_decoder_fea(lower_fea_recon, coord_recon_key)
                    )
                    lower_fea_recon = self.indexed_entropy_model_fea.decompress(
                        [fea_bytes], fea_indexes, target_device,
                        sparse_tensor_coords_tuple=(coord_recon_key, cm)
                    )
            strided_fea_recon_list.append(lower_fea_recon)
        assert len(coord_bytes_list) == len(fea_bytes_list) == 0
        return strided_fea_recon_list[-1]

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


class GeoLosslessScaleNoisyNormalEntropyModel(GeoLosslessEntropyModel):
    def __init__(self,
                 bottom_fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,

                 hyper_decoder_coord: Union[nn.Module, nn.ModuleList],
                 hyper_decoder_fea: Union[nn.Module, nn.ModuleList],
                 hybrid_hyper_decoder_fea: bool,

                 coord_index_num_scales: int = 64,
                 coord_index_scale_min: float = 0.11,
                 coord_index_scale_max: float = 256,
                 fea_index_num_scales: int = 64,
                 fea_index_scale_min: float = 0.11,
                 fea_index_scale_max: float = 256,

                 skip_encoding_fea: Tuple[int] = (),
                 upper_fea_grad_scaler_for_bits_loss: float = 1.0,
                 bottleneck_fea_process: str = 'noise',
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 init_scale: float = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        coord_parameter_fns = noisy_scale_normal_indexed_entropy_model_init(
            coord_index_scale_min, coord_index_scale_max, coord_index_num_scales
        )
        fea_parameter_fns = noisy_scale_normal_indexed_entropy_model_init(
            fea_index_scale_min, fea_index_scale_max, fea_index_num_scales
        )
        super(GeoLosslessScaleNoisyNormalEntropyModel, self).__init__(
            bottom_fea_entropy_model, encoder,
            hyper_decoder_coord, hyper_decoder_fea, hybrid_hyper_decoder_fea,
            NoisyNormal, (coord_index_num_scales,), coord_parameter_fns,
            NoisyNormal, (fea_index_num_scales,), fea_parameter_fns,
            lambda x: x, lambda x: x,
            skip_encoding_fea, upper_fea_grad_scaler_for_bits_loss,
            bottleneck_fea_process, 1,
            indexes_bound_gradient, quantize_indexes, indexes_scaler,
            init_scale, tail_mass, range_coder_precision
        )


class GeoLosslessNoisyDeepFactorizedEntropyModel(GeoLosslessEntropyModel):
    def __init__(self,
                 bottom_fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,

                 hyper_decoder_coord: Union[nn.Module, nn.ModuleList],
                 hyper_decoder_fea: Union[nn.Module, nn.ModuleList],
                 hybrid_hyper_decoder_fea: bool,

                 coord_index_ranges: Tuple[int, ...] = (8, 8, 8, 8),
                 coord_parameter_fns_type: str = 'transform',
                 coord_parameter_fns_factory: Callable[..., nn.Module] = None,
                 coord_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 1),
                 fea_index_ranges: Tuple[int, ...] = (16, 16, 16, 16),
                 fea_parameter_fns_type: str = 'transform',
                 fea_parameter_fns_factory: Callable[..., nn.Module] = None,
                 fea_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 1),

                 skip_encoding_fea: Tuple[int] = (),
                 upper_fea_grad_scaler_for_bits_loss: float = 1.0,
                 bottleneck_fea_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 init_scale: float = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        coord_parameter_fns, coord_indexes_view_fn, coord_modules_to_add = \
            noisy_deep_factorized_indexed_entropy_model_init(
                coord_index_ranges, coord_parameter_fns_type,
                coord_parameter_fns_factory, coord_num_filters
            )
        fea_parameter_fns, fea_indexes_view_fn, fea_modules_to_add = \
            noisy_deep_factorized_indexed_entropy_model_init(
                fea_index_ranges, fea_parameter_fns_type,
                fea_parameter_fns_factory, fea_num_filters
            )
        super(GeoLosslessNoisyDeepFactorizedEntropyModel, self).__init__(
            bottom_fea_entropy_model, encoder,
            hyper_decoder_coord, hyper_decoder_fea, hybrid_hyper_decoder_fea,
            NoisyDeepFactorized, coord_index_ranges, coord_parameter_fns,
            partial(NoisyDeepFactorized, noise_width=1 / bottleneck_scaler),
            fea_index_ranges, fea_parameter_fns,
            coord_indexes_view_fn, fea_indexes_view_fn,
            skip_encoding_fea, upper_fea_grad_scaler_for_bits_loss,
            bottleneck_fea_process, bottleneck_scaler,
            indexes_bound_gradient, quantize_indexes, indexes_scaler,
            init_scale, tail_mass, range_coder_precision
        )
        for module_name, module in coord_modules_to_add.items():
            setattr(self, 'coord_' + module_name, module)
        for module_name, module in fea_modules_to_add.items():
            setattr(self, 'fea_' + module_name, module)

    def _apply(self, fn):
        super(GeoLosslessNoisyDeepFactorizedEntropyModel, self)._apply(fn)
        self.indexed_entropy_model_coord.update_prior()
        self.indexed_entropy_model_fea.update_prior()
