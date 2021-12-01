import io
from typing import List, Tuple, Union, Dict, Any, Callable, Optional
import math

import torch
import torch.nn as nn
from torch.distributions import Distribution
import MinkowskiEngine as ME

from ..noisy_deep_factorized import \
    _noisy_deep_factorized_entropy_model_init, \
    EntropyModel as HyperPriorEntropyModel
from ...continuous_batched import ContinuousBatchedEntropyModel
from ...continuous_indexed import ContinuousIndexedEntropyModel
from ...distributions.uniform_noise import NoisyNormal, NoisyDeepFactorized

from lib.torch_utils import minkowski_tensor_wrapped_op, concat_loss_dicts


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
                 detach_higher_fea: bool,

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
                 fea_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,

                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        super(GeoLosslessEntropyModel, self).__init__()
        self.bottom_fea_entropy_model = bottom_fea_entropy_model
        self.encoder = encoder
        self.detach_higher_fea = detach_higher_fea
        self.hyper_decoder_coord = hyper_decoder_coord
        self.hyper_decoder_fea = hyper_decoder_fea
        self.hybrid_hyper_decoder_fea = hybrid_hyper_decoder_fea
        self.hyper_decoder_coord_post_op = hyper_decoder_coord_post_op
        self.hyper_decoder_fea_post_op = hyper_decoder_fea_post_op
        self.fea_bytes_num_bytes = fea_bytes_num_bytes
        self.coord_bytes_num_bytes = coord_bytes_num_bytes

        self.indexed_entropy_model_coord = ContinuousIndexedEntropyModel(
            prior_fn=coord_prior_fn,
            index_ranges=coord_index_ranges,
            parameter_fns=coord_parameter_fns,
            coding_ndim=2,
            indexes_bound_gradient=indexes_bound_gradient,
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

        self.indexed_entropy_model_fea = ContinuousIndexedEntropyModel(
            prior_fn=fea_prior_fn,
            index_ranges=fea_index_ranges,
            parameter_fns=fea_parameter_fns,
            coding_ndim=2,
            indexes_bound_gradient=indexes_bound_gradient,
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def get_sub_hyper_decoder_coord(self, idx):
        if isinstance(self.hyper_decoder_coord, nn.ModuleList):
            return self.hyper_decoder_coord[idx]
        else:
            return self.hyper_decoder_coord

    def get_sub_hyper_decoder_fea(self, idx):
        if isinstance(self.hyper_decoder_fea, nn.ModuleList):
            return self.hyper_decoder_fea[idx]
        else:
            return self.hyper_decoder_fea

    def forward(self, y_top: ME.SparseTensor, coder_num: int):
        if self.training:
            cm = y_top.coordinate_manager
            strided_fea_list = self.encoder(y_top, coder_num)
            *strided_fea_list, bottom_fea = strided_fea_list
            assert len(strided_fea_list) == coder_num

            loss_dict = {}
            strided_fea_tilde_list = []

            bottom_fea_tilde, fea_loss_dict = self.bottom_fea_entropy_model(
                bottom_fea, return_aux_loss=True
            )
            strided_fea_tilde_list.append(bottom_fea_tilde)
            concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: 'fea_bottom_' + k)
            lower_fea_tilde = bottom_fea_tilde

            for idx in range(coder_num - 1, -1, -1):
                sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(idx)
                sub_hyper_decoder_fea = self.get_sub_hyper_decoder_fea(idx)
                fea = strided_fea_list[idx]

                pre_coord_mask_indexes = sub_hyper_decoder_coord(lower_fea_tilde)
                coord_mask_indexes = self.hyper_decoder_coord_post_op(pre_coord_mask_indexes)
                coord_target_key = fea.coordinate_map_key
                coord_mask = self.get_coord_mask((coord_target_key, cm), pre_coord_mask_indexes)
                coord_mask_f_, coord_loss_dict = self.indexed_entropy_model_coord(
                    coord_mask.F[None], coord_mask_indexes,
                    return_aux_loss=idx == coder_num - 1,
                    additive_uniform_noise=False
                )
                concat_loss_dicts(loss_dict, coord_loss_dict, lambda k: f'coord_{idx}_' + k)

                if self.hybrid_hyper_decoder_fea is True:
                    fea_info_pred = sub_hyper_decoder_fea(lower_fea_tilde, coord_target_key).F
                    assert fea.F.shape[1] * (
                            len(self.indexed_entropy_model_fea.index_ranges) + 1
                    ) == fea_info_pred.shape[1]
                    fea_pred, pre_fea_indexes = torch.split(
                        fea_info_pred,
                        [fea.F.shape[1], fea_info_pred.shape[1] - fea.F.shape[1]],
                        dim=1
                    )
                    fea_indexes = self.hyper_decoder_fea_post_op(pre_fea_indexes)[None]
                    fea_pred_res_tilde, fea_loss_dict = self.indexed_entropy_model_fea(
                        (fea.F - fea_pred)[None], fea_indexes,
                        return_aux_loss=idx == coder_num - 1,
                        detach_value_for_bits_loss=self.detach_higher_fea
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
                        fea.F[None], fea_indexes,
                        return_aux_loss=idx == coder_num - 1,
                        detach_value_for_bits_loss=self.detach_higher_fea
                    )
                    lower_fea_tilde = ME.SparseTensor(
                        features=fea_tilde[0],
                        coordinate_map_key=coord_target_key,
                        coordinate_manager=cm
                    )

                strided_fea_tilde_list.append(lower_fea_tilde)
                concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: f'fea_{idx}_' + k)

            return strided_fea_tilde_list[-1], loss_dict

        else:
            concat_string, bottom_fea_recon = self.compress(y_top, coder_num)

            # You can clear the shared coordinate manager in a upper module
            # after compression to save memory.

            recon = self.decompress(
                concat_string,
                bottom_fea_recon.device,
                (bottom_fea_recon.coordinate_map_key,
                 bottom_fea_recon.coordinate_manager),
                coder_num
            )
            return bottom_fea_recon, recon, concat_string

    def compress(self, y_top: ME.SparseTensor, coder_num: int) -> \
            Tuple[bytes, ME.SparseTensor]:
        # Batch dimension of sparse tensor feature is supposed to
        # be added in minkowski_tensor_wrapped_fn(),
        # thus all the inputs of entropy models are supposed to
        # have batch size == 1.

        strided_fea_list = self.encoder(y_top, coder_num)
        *strided_fea_list, bottom_fea = strided_fea_list
        assert len(strided_fea_list) == coder_num

        fea_strings = []
        coord_strings = []
        cm = y_top.coordinate_manager

        (bottom_fea_string,), coding_batch_shape, bottom_fea_recon = \
            self.bottom_fea_entropy_model.compress(
                bottom_fea, return_dequantized=True
            )
        lower_fea_recon = bottom_fea_recon

        for idx in range(coder_num - 1, -1, -1):
            sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(idx)
            sub_hyper_decoder_fea = self.get_sub_hyper_decoder_fea(idx)
            fea = strided_fea_list[idx]

            pre_coord_mask_indexes = sub_hyper_decoder_coord(lower_fea_recon)
            coord_mask_indexes = self.hyper_decoder_coord_post_op(pre_coord_mask_indexes)
            coord_target_key = fea.coordinate_map_key
            coord_mask = self.get_coord_mask((coord_target_key, cm), pre_coord_mask_indexes)
            (coord_string,), coord_mask_f_ = self.indexed_entropy_model_coord.compress(
                coord_mask.F[None], coord_mask_indexes, skip_quantization=True
            )
            coord_strings.append(coord_string)
            coord_recon = coord_mask.C[coord_mask.F.to(torch.bool)[:, 0]]
            coord_recon_key = cm.insert_and_map(
                coord_recon, pre_coord_mask_indexes.tensor_stride,
                pre_coord_mask_indexes.coordinate_map_key.get_key()[1] + 'pruned'
            )[0]

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
                    [fea.F.shape[1], fea_info_pred.shape[1] - fea.F.shape[1]],
                    dim=1
                )
                fea_indexes = self.hyper_decoder_fea_post_op(pre_fea_indexes)[None]
                (fea_string,), fea_pred_res_recon = self.indexed_entropy_model_fea.compress(
                    (fea.F - fea_pred)[None], fea_indexes, return_dequantized=True,
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
                (fea_string,), fea_recon = self.indexed_entropy_model_fea.compress(
                    fea.F[None], fea_indexes, return_dequantized=True,
                )
                lower_fea_recon = ME.SparseTensor(
                    features=fea_recon[0],
                    coordinate_map_key=coord_recon_key,
                    coordinate_manager=cm
                )

            fea_strings.append(fea_string)

        partial_concat_string = self.concat_strings(
            [fea_strings, coord_strings],
            [self.fea_bytes_num_bytes, self.coord_bytes_num_bytes]
        )
        concat_string = self.concat_strings(
            [[bottom_fea_string], [partial_concat_string]],
            [self.fea_bytes_num_bytes, 0]
        )

        return concat_string, bottom_fea_recon

    def decompress(self,
                   concat_string: bytes,
                   target_device: torch.device,
                   sparse_tensor_coords_tuple: Tuple[ME.CoordinateMapKey, ME.CoordinateManager],
                   coder_num: int) \
            -> ME.SparseTensor:
        (bottom_fea_string,), (partial_concat_string,) = self.split_strings(
            concat_string, 1, [self.fea_bytes_num_bytes, 0]
        )
        fea_strings, coord_strings = self.split_strings(
            partial_concat_string, coder_num,
            [self.fea_bytes_num_bytes, self.coord_bytes_num_bytes]
        )

        strided_fea_recon_list = []
        cm = sparse_tensor_coords_tuple[1]

        bottom_fea_recon = self.bottom_fea_entropy_model.decompress(
            [bottom_fea_string], torch.Size([1]), target_device,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )
        lower_fea_recon = bottom_fea_recon

        for idx in range(coder_num):
            sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(coder_num - 1 - idx)
            sub_hyper_decoder_fea = self.get_sub_hyper_decoder_fea(coder_num - 1 - idx)
            fea_string = fea_strings[idx]
            coord_string = coord_strings[idx]

            coord_mask_pre_indexes = sub_hyper_decoder_coord(lower_fea_recon)
            coord_mask_indexes = self.hyper_decoder_coord_post_op(coord_mask_pre_indexes)
            coord_mask: ME.SparseTensor = self.indexed_entropy_model_coord.decompress(
                [coord_string], coord_mask_indexes, target_device, skip_dequantization=True,
                sparse_tensor_coords_tuple=(coord_mask_pre_indexes.coordinate_map_key, cm)
            )
            coord_recon = coord_mask.C[coord_mask.F.to(torch.bool)[:, 0]]
            coord_recon_key = cm.insert_and_map(
                coord_recon, coord_mask_pre_indexes.tensor_stride,
                coord_mask_pre_indexes.coordinate_map_key.get_key()[1] + 'pruned'
            )[0]

            if self.hybrid_hyper_decoder_fea is True:
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
                    [fea_string], fea_indexes, target_device
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
                    [fea_string], fea_indexes, target_device,
                    sparse_tensor_coords_tuple=(coord_recon_key, cm)
                )
            strided_fea_recon_list.append(lower_fea_recon)

        return strided_fea_recon_list[-1]

    @staticmethod
    def concat_strings(strings_lists: List[List[bytes]],
                       length_bytes_numbers: List[int] = None) -> bytes:

        strings_lists_num = len(strings_lists)
        if length_bytes_numbers is None:
            length_bytes_numbers = [4] * strings_lists_num
        else:
            assert strings_lists_num == len(length_bytes_numbers)

        strings_num_one_list = len(strings_lists[0])
        for strings_list in strings_lists[1:]:
            assert len(strings_list) == strings_num_one_list

        with io.BytesIO() as bs:
            for idx, strings in enumerate(zip(*strings_lists)):
                for i, (string, bytes_num) in enumerate(zip(strings, length_bytes_numbers)):
                    if idx != strings_num_one_list - 1 or i != strings_lists_num - 1:
                        bs.write(len(string).to_bytes(bytes_num, 'little', signed=False))
                    bs.write(string)

            concat_string = bs.getvalue()

        return concat_string

    @staticmethod
    def split_strings(concat_string: bytes,
                      strings_num_one_list: int,
                      length_bytes_numbers: List[int]) -> List[List[bytes]]:

        strings_lists_num = len(length_bytes_numbers)
        strings_lists = [[] for _ in range(strings_lists_num)]

        with io.BytesIO(concat_string) as bs:
            for idx in range(strings_num_one_list):
                for i, bytes_num in enumerate(length_bytes_numbers):
                    if idx != strings_num_one_list - 1 or i != strings_lists_num - 1:
                        length = int.from_bytes(bs.read(bytes_num), 'little', signed=False)
                        strings_lists[i].append(bs.read(length))
                    else:
                        strings_lists[i].append(bs.read())

        return strings_lists

    @staticmethod
    def get_coord_mask(
            y_coords_tuple: Tuple[ME.CoordinateMapKey, ME.CoordinateManager],
            indexes: ME.SparseTensor,
            mapping_target_kernel_size=1,
            mapping_target_region_type='HYPER_CROSS') -> \
            ME.SparseTensor:

        mapping_target_region_type = getattr(ME.RegionType, mapping_target_region_type)

        cm = y_coords_tuple[1]
        assert cm is indexes.coordinate_manager

        target_key = y_coords_tuple[0]
        kernel_map = cm.kernel_map(indexes.coordinate_map_key,
                                   target_key,
                                   kernel_size=mapping_target_kernel_size,
                                   region_type=mapping_target_region_type)
        keep_target = torch.zeros(indexes.shape[0], dtype=torch.float, device=indexes.device)
        for _, curr_in in kernel_map.items():
            keep_target[curr_in[0].type(torch.long)] = 1

        target = ME.SparseTensor(
            features=keep_target[:, None],
            coordinate_map_key=indexes.coordinate_map_key,
            coordinate_manager=cm
        )

        return target


class GeoLosslessScaleNoisyNormalEntropyModel(GeoLosslessEntropyModel):
    def __init__(self,
                 bottom_fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,
                 detach_higher_fea: bool,

                 hyper_decoder_coord: Union[nn.Module, nn.ModuleList],
                 hyper_decoder_fea: Union[nn.Module, nn.ModuleList],
                 hybrid_hyper_decoder_fea: bool,

                 fea_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,

                 coord_index_num_scales: int = 64,
                 coord_index_scale_min: float = 0.11,
                 coord_index_scale_max: float = 256,
                 fea_index_num_scales: int = 64,
                 fea_index_scale_min: float = 0.11,
                 fea_index_scale_max: float = 256,

                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        coord_index_offset = math.log(coord_index_scale_min)
        coord_index_factor = (math.log(coord_index_scale_max) -
                              math.log(coord_index_scale_min)) / (coord_index_num_scales - 1)
        fea_index_offset = math.log(fea_index_scale_min)
        fea_index_factor = (math.log(fea_index_scale_max) -
                            math.log(fea_index_scale_min)) / (fea_index_num_scales - 1)

        super(GeoLosslessScaleNoisyNormalEntropyModel, self).__init__(
            bottom_fea_entropy_model, encoder, detach_higher_fea,
            hyper_decoder_coord, hyper_decoder_fea, hybrid_hyper_decoder_fea,
            NoisyNormal, (coord_index_num_scales,), {
                'loc': lambda _: 0,
                'scale': lambda i: torch.exp(coord_index_offset + coord_index_factor * i)},
            NoisyNormal, (fea_index_num_scales,), {
                'loc': lambda _: 0,
                'scale': lambda i: torch.exp(fea_index_offset + fea_index_factor * i)},
            lambda x: x, lambda x: x,
            fea_bytes_num_bytes, coord_bytes_num_bytes,
            indexes_bound_gradient, quantize_indexes,
            init_scale, tail_mass, range_coder_precision
        )

    def forward(self, y_top, *args, **kwargs):
        y_top = minkowski_tensor_wrapped_op(y_top, torch.abs)
        return super(GeoLosslessScaleNoisyNormalEntropyModel, self).forward(y_top, *args, **kwargs)


class GeoLosslessNoisyDeepFactorizedEntropyModel(GeoLosslessEntropyModel):
    def __init__(self,
                 bottom_fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,
                 detach_higher_fea: bool,

                 hyper_decoder_coord: Union[nn.Module, nn.ModuleList],
                 hyper_decoder_fea: Union[nn.Module, nn.ModuleList],
                 hybrid_hyper_decoder_fea: bool,

                 fea_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,

                 coord_index_ranges: Tuple[int, ...] = (4,) * 9,
                 coord_parameter_fns_type: str = 'split',
                 coord_parameter_fns_factory: Callable[..., nn.Module] = None,
                 coord_num_filters: Tuple[int, ...] = (1, 2, 1),
                 fea_index_ranges: Tuple[int, ...] = (4,) * 9,
                 fea_parameter_fns_type: str = 'split',
                 fea_parameter_fns_factory: Callable[..., nn.Module] = None,
                 fea_num_filters: Tuple[int, ...] = (1, 2, 1),

                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):

        coord_parameter_fns, coord_indexes_view_fn, coord_modules_to_add = \
            _noisy_deep_factorized_entropy_model_init(
                coord_index_ranges, coord_parameter_fns_type,
                coord_parameter_fns_factory, coord_num_filters
            )
        fea_parameter_fns, fea_indexes_view_fn, fea_modules_to_add = \
            _noisy_deep_factorized_entropy_model_init(
                fea_index_ranges, fea_parameter_fns_type,
                fea_parameter_fns_factory, fea_num_filters
            )

        super(GeoLosslessNoisyDeepFactorizedEntropyModel, self).__init__(
            bottom_fea_entropy_model, encoder, detach_higher_fea,
            hyper_decoder_coord, hyper_decoder_fea, hybrid_hyper_decoder_fea,
            NoisyDeepFactorized, coord_index_ranges, coord_parameter_fns,
            NoisyDeepFactorized, fea_index_ranges, fea_parameter_fns,
            coord_indexes_view_fn, fea_indexes_view_fn,
            fea_bytes_num_bytes, coord_bytes_num_bytes,
            indexes_bound_gradient, quantize_indexes,
            init_scale, tail_mass, range_coder_precision
        )

        for module_name, module in coord_modules_to_add.items():
            setattr(self, 'coord_' + module_name, module)
        for module_name, module in fea_modules_to_add.items():
            setattr(self, 'fea_' + module_name, module)
