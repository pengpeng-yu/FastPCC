import io
from typing import List, Tuple, Union, Dict, Any, Callable
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
                 fea_prior_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 skip_connection: bool,
                 fusion_method: str,

                 hyper_encoder: nn.Module,
                 decoder: nn.ModuleList,
                 hyper_decoder: nn.ModuleList,

                 prior_fn: Callable[..., Distribution],
                 index_ranges: Tuple[int, ...],
                 parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],

                 hyper_encoder_post_op: Callable = lambda x: x,
                 hyper_decoder_post_op: Callable = lambda x: x,
                 fea_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,

                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        super(GeoLosslessEntropyModel, self).__init__()

        self.fea_prior_entropy_model = fea_prior_entropy_model
        self.hyper_encoder = hyper_encoder
        self.decoder = decoder
        self.hyper_decoder = hyper_decoder
        self.hyper_coder_num = len(hyper_decoder)
        self.skip_connection = skip_connection
        self.fusion_method = fusion_method
        self.hyper_encoder_post_op = hyper_encoder_post_op
        self.hyper_decoder_post_op = hyper_decoder_post_op
        self.fea_bytes_num_bytes = fea_bytes_num_bytes
        self.coord_bytes_num_bytes = coord_bytes_num_bytes

        self.indexed_entropy_model = ContinuousIndexedEntropyModel(
            prior_fn=prior_fn,
            index_ranges=index_ranges,
            parameter_fns=parameter_fns,
            coding_ndim=2,
            additive_uniform_noise=False,
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def forward(self, y_top: ME.SparseTensor):
        if self.training:
            cm = y_top.coordinate_manager
            strided_list = [self.hyper_encoder_post_op(_) for _ in self.hyper_encoder(y_top)]

            loss_dict = {}

            bottom_fea_tilde, bottom_fea_loss_dict = self.fea_prior_entropy_model(strided_list[-1])
            concat_loss_dicts(loss_dict, bottom_fea_loss_dict)
            fea = bottom_fea_tilde

            for idx in range(self.hyper_coder_num - 1, -1, -1):
                sub_hyper_decoder = self.hyper_decoder[idx]
                sub_decoder = self.decoder[idx]

                pre_indexes: ME.SparseTensor = sub_hyper_decoder(fea)
                indexes = self.hyper_decoder_post_op(pre_indexes)
                coord_target_key = strided_list[idx].coordinate_map_key
                coord_mask = self.get_coord_mask((coord_target_key, cm), pre_indexes)
                coord_mask_f_, coord_loss_dict = self.indexed_entropy_model(coord_mask.F[None], indexes)
                concat_loss_dicts(loss_dict, coord_loss_dict, lambda k: 'coord_' + k)

                fea = sub_decoder(fea)

                if self.skip_connection is True:
                    fea_skipped, fea_loss_dict = self.fea_prior_entropy_model(strided_list[idx])
                    concat_loss_dicts(loss_dict, fea_loss_dict)

                    if self.fusion_method == 'Add':
                        fea += fea_skipped

                    elif self.fusion_method == 'Cat':
                        fea = ME.cat(fea, fea_skipped)

                    else:
                        raise NotImplementedError

            return fea, loss_dict

        else:
            concat_string, bottom_recon, recon_ = self.compress(y_top)

            # You can clear the shared coordinate manager in a upper module
            # after compression to save memory.

            recon = self.decompress(
                concat_string,
                bottom_recon.device,
                (bottom_recon.coordinate_map_key,
                 bottom_recon.coordinate_manager)
            )
            return bottom_recon, recon, concat_string

    def compress(self, y_top: ME.SparseTensor) -> Tuple[bytes, ME.SparseTensor, ME.SparseTensor]:
        # Batch dimension of sparse tensor feature is supposed to
        # be added in minkowski_tensor_wrapped_fn(),
        # thus all the inputs of entropy models are supposed to
        # have batch size == 1.

        strided_list = [self.hyper_encoder_post_op(_) for _ in self.hyper_encoder(y_top)]
        assert len(strided_list) == self.hyper_coder_num + 1

        cm = y_top.coordinate_manager
        fea_strings = []
        coord_strings = []

        (fea_string,), coding_batch_shape, bottom_recon = \
            self.fea_prior_entropy_model.compress(strided_list[-1], return_dequantized=True)
        fea_strings.append(fea_string)
        recon = bottom_recon

        for idx in range(self.hyper_coder_num - 1, -1, -1):
            sub_hyper_decoder = self.hyper_decoder[idx]
            sub_decoder = self.decoder[idx]

            pre_indexes: ME.SparseTensor = sub_hyper_decoder(recon)
            indexes = self.hyper_decoder_post_op(pre_indexes)
            coord_mask = self.get_coord_mask((strided_list[idx].coordinate_map_key, cm), pre_indexes)
            (coord_string,), coord_mask_f_ = self.indexed_entropy_model.compress(coord_mask.F[None], indexes, True)
            coord_recon = coord_mask.C[coord_mask.F.to(torch.bool)[:, 0]]
            coord_recon_key, _ = cm.insert_and_map(
                coord_recon, pre_indexes.tensor_stride,
                pre_indexes.coordinate_map_key.get_key()[1] + 'pruned'
            )

            recon = sub_decoder(recon, coordinates=coord_recon_key)

            if self.skip_connection is True:
                # Permute features from encoders to fit in with order of features from decoders.
                permutation_kernel_map = cm.kernel_map(
                    strided_list[idx].coordinate_map_key,
                    coord_recon_key,
                    kernel_size=1)[0].to(torch.long)

                (fea_string,), coding_batch_shape, fea_skipped = \
                    self.fea_prior_entropy_model.compress(
                        ME.SparseTensor(
                            features=strided_list[idx].F[permutation_kernel_map[0]][permutation_kernel_map[1]],
                            coordinate_map_key=coord_recon_key,
                            coordinate_manager=cm
                        ),
                        return_dequantized=True
                    )

                if self.fusion_method == 'Add':
                    recon += fea_skipped

                elif self.fusion_method == 'Cat':
                    recon = ME.cat(recon, fea_skipped)

                else: raise NotImplementedError
                fea_strings.append(fea_string)

            coord_strings.append(coord_string)

        if self.skip_connection is True:
            concat_string = self.concat_strings(
                [[fea_strings[0]],
                 [self.concat_strings([fea_strings[1:], coord_strings],
                                      [self.fea_bytes_num_bytes, self.coord_bytes_num_bytes])]],
                [self.fea_bytes_num_bytes, self.coord_bytes_num_bytes]
            )
        else:
            concat_string = self.concat_strings(
                [[fea_strings[0]],
                 [self.concat_strings([coord_strings], [self.coord_bytes_num_bytes])]],
                [self.fea_bytes_num_bytes, self.coord_bytes_num_bytes]
            )
        return concat_string, bottom_recon, recon

    def decompress(self,
                   concat_string: bytes,
                   target_device: torch.device,
                   sparse_tensor_coords_tuple: Tuple[ME.CoordinateMapKey, ME.CoordinateManager]) \
            -> ME.SparseTensor:

        (bottom_fea_string,),  (other_string,) = self.split_strings(concat_string, 1, [4, 4])
        if self.skip_connection:
            fea_strings, coord_strings = self.split_strings(other_string, self.hyper_coder_num, [4, 4])
        else:
            coord_strings, = self.split_strings(other_string, self.hyper_coder_num, [4])
            fea_strings = None

        cm = sparse_tensor_coords_tuple[1]

        recon = self.fea_prior_entropy_model.decompress(
            [bottom_fea_string], torch.Size([1]), target_device,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )

        for idx in range(self.hyper_coder_num):
            sub_hyper_decoder = self.hyper_decoder[self.hyper_coder_num - 1 - idx]
            sub_decoder = self.decoder[self.hyper_coder_num - 1 - idx]
            coord_string = coord_strings[idx]

            pre_indexes = sub_hyper_decoder(recon)

            indexes = self.hyper_decoder_post_op(pre_indexes)
            coord_mask: ME.SparseTensor = self.indexed_entropy_model.decompress(
                [coord_string], indexes, target_device, True,
                sparse_tensor_coords_tuple=(pre_indexes.coordinate_map_key, cm)
            )
            coord_recon = coord_mask.C[coord_mask.F.to(torch.bool)[:, 0]]
            coord_recon_key, _ = cm.insert_and_map(
                coord_recon, pre_indexes.tensor_stride,
                pre_indexes.coordinate_map_key.get_key()[1] + 'pruned'
            )
            recon = sub_decoder(recon, coordinates=coord_recon_key)

            if self.skip_connection:
                fea_string = fea_strings[idx]
                fea_skipped = self.fea_prior_entropy_model.decompress(
                    [fea_string], torch.Size([1]), target_device,
                    sparse_tensor_coords_tuple=(coord_recon_key, cm)
                )

                if self.fusion_method == 'Add':
                    recon += fea_skipped

                elif self.fusion_method == 'Cat':
                    recon = ME.cat(recon, fea_skipped)

                else: raise NotImplementedError

        return recon

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
                 fea_prior_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 skip_connection: bool,
                 fusion_method: str,

                 hyper_encoder: nn.Module,
                 decoder: nn.ModuleList,
                 hyper_decoder: nn.ModuleList,

                 fea_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,
                 num_scales: int = 64,
                 scale_min: float = 0.11,
                 scale_max: float = 256,

                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min)) / (num_scales - 1)

        super(GeoLosslessScaleNoisyNormalEntropyModel, self).__init__(
            fea_prior_entropy_model, skip_connection, fusion_method,
            hyper_encoder, decoder, hyper_decoder,
            NoisyNormal, (num_scales,), {'loc': lambda _: 0,
                                         'scale': lambda i: torch.exp(offset + factor * i)},
            lambda x: x, lambda x: x, fea_bytes_num_bytes, coord_bytes_num_bytes,
            quantize_indexes, init_scale, tail_mass, range_coder_precision
        )

    def forward(self, y):
        y = minkowski_tensor_wrapped_op(y, torch.abs_)
        return super(GeoLosslessScaleNoisyNormalEntropyModel, self).forward(y)


class GeoLosslessNoisyDeepFactorizedEntropyModel(GeoLosslessEntropyModel):
    def __init__(self,
                 fea_prior_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 skip_connection: bool,
                 fusion_method: str,

                 hyper_encoder: nn.Module,
                 decoder: nn.ModuleList,
                 hyper_decoder: nn.ModuleList,

                 fea_bytes_num_bytes: int = 2,
                 coord_bytes_num_bytes: int = 2,

                 index_ranges: Tuple[int, ...] = (4,) * 9,
                 parameter_fns_type: str = 'split',
                 parameter_fns_factory: Callable[..., nn.Module] = None,
                 num_filters: Tuple[int, ...] = (1, 2, 1),
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
            fea_prior_entropy_model, skip_connection, fusion_method,
            hyper_encoder, decoder, hyper_decoder,
            NoisyDeepFactorized, index_ranges, parameter_fns,
            lambda x: x, indexes_view_fn, fea_bytes_num_bytes, coord_bytes_num_bytes,
            quantize_indexes, init_scale, tail_mass, range_coder_precision
        )

        for module_name, module in modules_to_add.items():
            setattr(self, module_name, module)
