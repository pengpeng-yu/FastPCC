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
from ...continuous_batched import ContinuousBatchedEntropyModel, \
    NoisyDeepFactorizedEntropyModel as NoisyDeepFactorizedPriorEntropyModel
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
                 fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,
                 decoder: nn.Module,

                 hyper_encoder_coord: Union[nn.Module, nn.ModuleList],
                 hyper_decoder_coord: Union[nn.Module, nn.ModuleList],

                 hyperprior_batch_channels: int,

                 prior_fn: Callable[..., Distribution],
                 index_ranges: Tuple[int, ...],
                 parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],

                 hyper_encoder_post_op: Callable = lambda x: x,
                 hyper_decoder_post_op: Callable = lambda x: x,
                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_channels_bytes: int = 2,
                 fea_bytes_num_bytes: int = 2,
                 coord_prior_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,

                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        super(GeoLosslessEntropyModel, self).__init__()

        self.fea_entropy_model = fea_entropy_model
        self.encoder = encoder
        self.decoder = decoder
        self.hyper_encoder_coord = hyper_encoder_coord
        self.hyper_decoder_coord = hyper_decoder_coord
        self.hyper_encoder_post_op = hyper_encoder_post_op
        self.hyper_decoder_post_op = hyper_decoder_post_op
        self.fea_bytes_num_bytes = fea_bytes_num_bytes
        self.coord_prior_bytes_num_bytes = coord_prior_bytes_num_bytes
        self.coord_bytes_num_bytes = coord_bytes_num_bytes

        self.hyperprior_entropy_model = NoisyDeepFactorizedPriorEntropyModel(
            batch_shape=torch.Size([hyperprior_batch_channels]),
            coding_ndim=2,
            num_filters=hyperprior_num_filters,
            init_scale=hyperprior_init_scale,
            tail_mass=hyperprior_tail_mass,
            range_coder_precision=range_coder_precision,
            broadcast_shape_bytes=(hyperprior_broadcast_channels_bytes,)
        )

        self.indexed_entropy_model = ContinuousIndexedEntropyModel(
            prior_fn=prior_fn,
            index_ranges=index_ranges,
            parameter_fns=parameter_fns,
            coding_ndim=2,
            indexes_bound_gradient=indexes_bound_gradient,
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def get_sub_hyper_encoder_coord(self, idx):
        if isinstance(self.hyper_encoder_coord, nn.ModuleList):
            return self.hyper_encoder_coord[idx]
        else:
            return self.hyper_encoder_coord

    def get_sub_hyper_decoder_coord(self, idx):
        if isinstance(self.hyper_decoder_coord, nn.ModuleList):
            return self.hyper_decoder_coord[idx]
        else:
            return self.hyper_decoder_coord

    def forward(self, y_top: ME.SparseTensor):
        if self.training:
            cm = y_top.coordinate_manager
            strided_fea_for_coord_list, (top_fea, *strided_fea_list), *encoder_args = self.encoder(y_top)
            strided_fea_list = [self.hyper_encoder_post_op(_) for _ in strided_fea_list]
            coder_num = len(strided_fea_list)

            loss_dict = {}
            strided_fea_tilde_list = []

            for idx in range(coder_num - 1, -1, -1):
                sub_hyper_encoder_coord = self.get_sub_hyper_encoder_coord(idx)
                sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(idx)
                fea = strided_fea_list[idx]
                fea_for_coord = strided_fea_for_coord_list[idx]

                fea_tilde, fea_loss_dict = self.fea_entropy_model(
                    fea, return_aux_loss=idx == coder_num - 1
                )
                strided_fea_tilde_list.append(fea_tilde)
                concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: 'fea_' + k)

                coord_mask_z = self.hyper_encoder_post_op(sub_hyper_encoder_coord(fea_for_coord))
                coord_mask_z_tilde, coord_hyperprior_loss_dict = \
                    self.hyperprior_entropy_model(coord_mask_z, return_aux_loss=idx == coder_num - 1)
                concat_loss_dicts(loss_dict, coord_hyperprior_loss_dict, lambda k: 'coord_hyper_' + k)

                pre_indexes = sub_hyper_decoder_coord(coord_mask_z_tilde)
                indexes = self.hyper_decoder_post_op(pre_indexes)
                coord_target_key = strided_fea_list[idx - 1].coordinate_map_key \
                    if idx != 0 else y_top.coordinate_map_key
                coord_mask = self.get_coord_mask((coord_target_key, cm), pre_indexes)
                coord_mask_f_, coord_loss_dict = self.indexed_entropy_model(
                    coord_mask.F[None], indexes,
                    return_aux_loss=idx == coder_num - 1,
                    additive_uniform_noise=False
                )
                concat_loss_dicts(loss_dict, coord_loss_dict, lambda k: 'coord_' + k)

            fea_tilde, fea_loss_dict = self.fea_entropy_model(
                top_fea, return_aux_loss=False
            )
            strided_fea_tilde_list.append(fea_tilde)
            concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: 'fea_top_' + k)

            y_top_recon, decoder_loss_dict = self.decoder(strided_fea_tilde_list, *encoder_args)
            concat_loss_dicts(loss_dict, decoder_loss_dict)
            return y_top_recon, loss_dict

        else:
            concat_string, bottom_rounded_fea, coder_num = self.compress(y_top)

            # You can clear the shared coordinate manager in a upper module
            # after compression to save memory.

            recon = self.decompress(
                concat_string,
                bottom_rounded_fea.device,
                (bottom_rounded_fea.coordinate_map_key,
                 bottom_rounded_fea.coordinate_manager),
                coder_num
            )
            return bottom_rounded_fea, recon, concat_string

    def compress(self, y_top: ME.SparseTensor) -> \
            Tuple[bytes, ME.SparseTensor, int]:
        # Batch dimension of sparse tensor feature is supposed to
        # be added in minkowski_tensor_wrapped_fn(),
        # thus all the inputs of entropy models are supposed to
        # have batch size == 1.

        strided_fea_for_coord_list, (top_fea, *strided_fea_list) = self.encoder(y_top)
        strided_fea_list = [self.hyper_encoder_post_op(_) for _ in strided_fea_list]
        coder_num = len(strided_fea_list)

        fea_strings = []
        coord_prior_strings = []
        coord_strings = []

        cm = y_top.coordinate_manager
        bottom_rounded_fea = None
        coord_target_key = None
        coord_recon_key = None

        for idx in range(coder_num - 1, -1, -1):
            sub_hyper_encoder_coord = self.get_sub_hyper_encoder_coord(idx)
            sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(idx)
            fea = strided_fea_list[idx]
            fea_for_coord = strided_fea_for_coord_list[idx]

            if idx != coder_num - 1:
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
                fea_for_coord = ME.SparseTensor(
                    features=fea_for_coord.F[permutation_kernel_map],
                    coordinate_map_key=coord_recon_key,
                    coordinate_manager=cm
                )

            coord_mask_z = self.hyper_encoder_post_op(sub_hyper_encoder_coord(fea_for_coord))
            (coord_prior_string,), coding_batch_shape, coord_mask_z_recon = \
                self.hyperprior_entropy_model.compress(
                    coord_mask_z, return_dequantized=True
                )
            coord_prior_strings.append(coord_prior_string)
            pre_indexes = sub_hyper_decoder_coord(coord_mask_z_recon)
            coord_mask_indexes = self.hyper_decoder_post_op(pre_indexes)
            coord_target_key = strided_fea_list[idx - 1].coordinate_map_key \
                if idx != 0 else y_top.coordinate_map_key
            coord_mask = self.get_coord_mask((coord_target_key, cm), pre_indexes)
            (coord_string,), coord_mask_f_ = self.indexed_entropy_model.compress(
                coord_mask.F[None], coord_mask_indexes, skip_quantization=True
            )
            coord_strings.append(coord_string)
            coord_recon = coord_mask.C[coord_mask.F.to(torch.bool)[:, 0]]
            coord_recon_key = cm.insert_and_map(
                coord_recon, pre_indexes.tensor_stride,
                pre_indexes.coordinate_map_key.get_key()[1] + 'pruned'
            )[0]

            (fea_string,), coding_batch_shape, rounded_fea = \
                self.fea_entropy_model.compress(fea)
            fea_strings.append(fea_string)
            if idx == coder_num - 1:
                bottom_rounded_fea = rounded_fea

        permutation_kernel_map = cm.kernel_map(
            coord_target_key,
            coord_recon_key,
            kernel_size=1)[0][0].to(torch.long)
        top_fea = ME.SparseTensor(
            features=top_fea.F[permutation_kernel_map],
            coordinate_map_key=coord_recon_key,
            coordinate_manager=cm
        )
        (top_fea_string,), coding_batch_shape, rounded_top_fea = \
            self.fea_entropy_model.compress(top_fea)

        partial_concat_string = self.concat_strings(
            [fea_strings, coord_prior_strings, coord_strings],
            [self.fea_bytes_num_bytes, self.coord_prior_bytes_num_bytes, self.coord_bytes_num_bytes]
        )
        concat_string = self.concat_strings(
            [[top_fea_string], [partial_concat_string]],
            [self.fea_bytes_num_bytes, 0]
        )

        return concat_string, bottom_rounded_fea, coder_num

    def decompress(self,
                   concat_string: bytes,
                   target_device: torch.device,
                   sparse_tensor_coords_tuple: Tuple[ME.CoordinateMapKey, ME.CoordinateManager],
                   coder_num: int) \
            -> ME.SparseTensor:
        (top_fea_string,), (partial_concat_string,) = self.split_strings(
            concat_string, 1, [self.fea_bytes_num_bytes, 0]
        )
        fea_strings, coord_prior_strings, coord_strings = self.split_strings(
            partial_concat_string, coder_num,
            [self.fea_bytes_num_bytes, self.coord_prior_bytes_num_bytes, self.coord_bytes_num_bytes]
        )

        strided_fea_recon_list = []
        cm = sparse_tensor_coords_tuple[1]
        gen_conv_trans = ME.MinkowskiGenerativeConvolutionTranspose(
            1, 1, 2, 2, dimension=3
        ).to(target_device)

        for idx in range(coder_num):
            sub_hyper_decoder_coord = self.get_sub_hyper_decoder_coord(coder_num - 1 - idx)
            fea_string = fea_strings[idx]
            coord_prior_string = coord_prior_strings[idx]
            coord_string = coord_strings[idx]

            fea = self.fea_entropy_model.decompress(
                [fea_string], torch.Size([1]), target_device,
                sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
            )
            strided_fea_recon_list.append(fea)

            coord_mask_z_recon = self.hyperprior_entropy_model.decompress(
                [coord_prior_string], torch.Size([1]), target_device,
                sparse_tensor_coords_tuple=(
                    gen_conv_trans(
                        ME.SparseTensor(
                            features=torch.zeros((fea.shape[0], 1),
                                                 dtype=fea.F.dtype, device=fea.device),
                            coordinate_map_key=fea.coordinate_map_key,
                            coordinate_manager=cm
                        )).coordinate_map_key, cm
                    )
                )

            pre_indexes = sub_hyper_decoder_coord(coord_mask_z_recon)

            indexes = self.hyper_decoder_post_op(pre_indexes)
            coord_mask: ME.SparseTensor = self.indexed_entropy_model.decompress(
                [coord_string], indexes, target_device, skip_dequantization=True,
                sparse_tensor_coords_tuple=(pre_indexes.coordinate_map_key, cm)
            )
            coord_recon = coord_mask.C[coord_mask.F.to(torch.bool)[:, 0]]
            coord_recon_key = cm.insert_and_map(
                coord_recon, pre_indexes.tensor_stride,
                pre_indexes.coordinate_map_key.get_key()[1] + 'pruned'
            )[0]

            sparse_tensor_coords_tuple = coord_recon_key, cm

        top_fea = self.fea_entropy_model.decompress(
            [top_fea_string], torch.Size([1]), target_device,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )
        strided_fea_recon_list.append(top_fea)

        return self.decoder(strided_fea_recon_list)

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
                 fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,
                 decoder: nn.Module,

                 hyper_encoder_coord: Union[nn.Module, nn.ModuleList],
                 hyper_decoder_coord: Union[nn.Module, nn.ModuleList],

                 hyperprior_batch_channels: int,

                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_channels_bytes: int = 2,
                 fea_bytes_num_bytes: int = 2,
                 coord_prior_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,
                 num_scales: int = 64,
                 scale_min: float = 0.11,
                 scale_max: float = 256,

                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min)) / (num_scales - 1)

        super(GeoLosslessScaleNoisyNormalEntropyModel, self).__init__(
            fea_entropy_model, encoder, decoder,
            hyper_encoder_coord, hyper_decoder_coord,
            hyperprior_batch_channels,
            NoisyNormal, (num_scales,), {'loc': lambda _: 0,
                                         'scale': lambda i: torch.exp(offset + factor * i)},
            lambda x: x, lambda x: x,
            hyperprior_num_filters, hyperprior_init_scale, hyperprior_tail_mass,
            hyperprior_broadcast_channels_bytes,
            fea_bytes_num_bytes, coord_prior_bytes_num_bytes, coord_bytes_num_bytes,
            indexes_bound_gradient, quantize_indexes,
            init_scale, tail_mass, range_coder_precision
        )

    def forward(self, y_top, *args, **kwargs):
        y_top = minkowski_tensor_wrapped_op(y_top, torch.abs_)
        return super(GeoLosslessScaleNoisyNormalEntropyModel, self).forward(y_top, *args, **kwargs)


class GeoLosslessNoisyDeepFactorizedEntropyModel(GeoLosslessEntropyModel):
    def __init__(self,
                 fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,
                 decoder: nn.Module,

                 hyper_encoder_coord: Union[nn.Module, nn.ModuleList],
                 hyper_decoder_coord: Union[nn.Module, nn.ModuleList],

                 hyperprior_batch_channels: int,

                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_channels_bytes: int = 2,
                 fea_bytes_num_bytes: int = 2,
                 coord_prior_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,

                 index_ranges: Tuple[int, ...] = (4,) * 9,
                 parameter_fns_type: str = 'split',
                 parameter_fns_factory: Callable[..., nn.Module] = None,
                 num_filters: Tuple[int, ...] = (1, 2, 1),
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):

        parameter_fns, indexes_view_fn, modules_to_add = \
            _noisy_deep_factorized_entropy_model_init(
                index_ranges, parameter_fns_type, parameter_fns_factory, num_filters
            )

        super(GeoLosslessNoisyDeepFactorizedEntropyModel, self).__init__(
            fea_entropy_model, encoder, decoder,
            hyper_encoder_coord, hyper_decoder_coord,
            hyperprior_batch_channels,
            NoisyDeepFactorized, index_ranges, parameter_fns,
            lambda x: x, indexes_view_fn,
            hyperprior_num_filters, hyperprior_init_scale, hyperprior_tail_mass,
            hyperprior_broadcast_channels_bytes,
            fea_bytes_num_bytes, coord_prior_bytes_num_bytes, coord_bytes_num_bytes,
            indexes_bound_gradient, quantize_indexes,
            init_scale, tail_mass, range_coder_precision
        )

        for module_name, module in modules_to_add.items():
            setattr(self, module_name, module)
