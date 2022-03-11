from typing import List, Tuple, Union, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.distributions import Distribution

from ...continuous_batched import NoisyDeepFactorizedEntropyModel as PriorEntropyModel
from ...continuous_indexed import ContinuousIndexedEntropyModel, \
    noisy_deep_factorized_indexed_entropy_model_init, \
    noisy_scale_normal_indexed_entropy_model_init
from ...distributions.uniform_noise import NoisyNormal, NoisyDeepFactorized

from lib.torch_utils import \
    minkowski_tensor_wrapped_op, \
    get_minkowski_tensor_coords_tuple, \
    concat_loss_dicts


class EntropyModel(nn.Module):
    """
    Note:
        Variable y, input of forward(), compress() and output of decompress(),
        could be a torch.Tensor or a ME.SparseTensor.
        Only supports batch size == 1 during testing if y is a ME.SparseTensor.
    """

    def __init__(self,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module,

                 hyperprior_batch_shape: torch.Size,
                 coding_ndim: int,

                 prior_fn: Callable[..., Distribution],
                 index_ranges: Tuple[int, ...],
                 parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],

                 hyper_encoder_post_op: Callable = lambda x: x,
                 hyper_decoder_post_op: Callable = lambda x: x,
                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_shape_bytes: Tuple[int, ...] = (2,),
                 prior_bytes_num_bytes: int = 2,

                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16,
                 ):
        super(EntropyModel, self).__init__()
        self.hyper_encoder = hyper_encoder
        self.hyper_decoder = hyper_decoder
        self.hyper_encoder_post_op = hyper_encoder_post_op
        self.hyper_decoder_post_op = hyper_decoder_post_op
        self.prior_bytes_num_bytes = prior_bytes_num_bytes
        self.hyperprior_entropy_model = PriorEntropyModel(
            batch_shape=hyperprior_batch_shape,
            coding_ndim=coding_ndim,
            num_filters=hyperprior_num_filters,
            init_scale=hyperprior_init_scale,
            tail_mass=hyperprior_tail_mass,
            range_coder_precision=range_coder_precision,
            broadcast_shape_bytes=hyperprior_broadcast_shape_bytes
        )
        self.prior_entropy_model = ContinuousIndexedEntropyModel(
            prior_fn=prior_fn,
            index_ranges=index_ranges,
            parameter_fns=parameter_fns,
            coding_ndim=coding_ndim,
            indexes_bound_gradient=indexes_bound_gradient,
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def forward(self, y, return_aux_loss: bool = True):
        if self.training:
            z = self.hyper_encoder_post_op(self.hyper_encoder(y))
            z_tilde, hyperprior_loss_dict = self.hyperprior_entropy_model(z, return_aux_loss)
            indexes = self.hyper_decoder_post_op(self.hyper_decoder(z_tilde))
            y_tilde, prior_loss_dict = self.prior_entropy_model(y, indexes, return_aux_loss)
            loss_dict = concat_loss_dicts(prior_loss_dict, hyperprior_loss_dict, lambda k: 'hyper_' + k)
            return y_tilde, loss_dict

        else:
            concat_strings, coding_batch_shape, rounded_y = self.compress(y)
            sparse_tensor_coords_tuple = get_minkowski_tensor_coords_tuple(y)
            y_recon = self.decompress(
                concat_strings,
                coding_batch_shape,
                y.device,
                sparse_tensor_coords_tuple
            )
            return y_recon, concat_strings, coding_batch_shape

    def compress(self, y, return_dequantized: bool = False, estimate_bits: bool = False) \
            -> Union[Tuple[List[bytes], torch.Size, torch.Tensor],
                     Tuple[List[bytes], torch.Size, torch.Tensor, torch.Tensor]]:
        z = self.hyper_encoder_post_op(self.hyper_encoder(y))
        prior_strings, coding_batch_shape, z_recon, *estimated_prior_bits = \
            self.hyperprior_entropy_model.compress(
                z, return_dequantized=True, estimate_bits=estimate_bits
            )
        indexes = self.hyper_decoder_post_op(self.hyper_decoder(z_recon))
        strings, rounded_y_or_dequantized_y, *estimated_bits = \
            self.prior_entropy_model.compress(
                y, indexes, return_dequantized=return_dequantized, estimate_bits=estimate_bits
            )
        concat_strings = self.concat_strings(prior_strings, strings)
        if estimated_bits:
            return concat_strings, coding_batch_shape, rounded_y_or_dequantized_y, \
                   estimated_prior_bits[0] + estimated_bits[0]
        else:
            return concat_strings, coding_batch_shape, rounded_y_or_dequantized_y,

    def decompress(self,
                   concat_strings: List[bytes],
                   coding_batch_shape: torch.Size,
                   target_device: torch.device,
                   sparse_tensor_coords_tuple: Tuple = None) -> Any:
        prior_strings, strings = self.split_strings(concat_strings)
        z_recon = self.hyperprior_entropy_model.decompress(
            prior_strings, coding_batch_shape, target_device, False,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )
        pre_indexes = self.hyper_decoder(z_recon)
        sparse_tensor_coords_tuple = get_minkowski_tensor_coords_tuple(pre_indexes)
        indexes = self.hyper_decoder_post_op(pre_indexes)
        y_recon = self.prior_entropy_model.decompress(
            strings, indexes, target_device, False,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )
        return y_recon

    def concat_strings(self, prior_strings: List[bytes], strings: List[bytes]) -> List[bytes]:
        return [len(i).to_bytes(self.prior_bytes_num_bytes, 'little', signed=False) + i + j
                for i, j in zip(prior_strings, strings)]

    def split_strings(self, concat_strings: List[bytes]) -> Tuple[List[bytes], List[bytes]]:
        prior_strings = []
        strings = []
        for concat_s in concat_strings:
            prior_strings_len = int.from_bytes(
                concat_s[:self.prior_bytes_num_bytes],
                'little', signed=False
            )
            prior_strings.append(
                concat_s[self.prior_bytes_num_bytes:
                         self.prior_bytes_num_bytes + prior_strings_len]
            )
            strings.append(
                concat_s[self.prior_bytes_num_bytes + prior_strings_len:]
            )
        return prior_strings, strings


class ScaleNoisyNormalEntropyModel(EntropyModel):
    def __init__(self,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module,

                 hyperprior_batch_shape: torch.Size,
                 coding_ndim: int,

                 num_scales: int = 64,
                 scale_min: float = 0.11,
                 scale_max: float = 256,

                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_shape_bytes: Tuple[int, ...] = (2,),
                 prior_bytes_num_bytes: int = 2,

                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        parameter_fns = noisy_scale_normal_indexed_entropy_model_init(
            scale_min, scale_max, num_scales
        )
        super(ScaleNoisyNormalEntropyModel, self).__init__(
            hyper_encoder, hyper_decoder,
            hyperprior_batch_shape, coding_ndim,
            NoisyNormal, (num_scales,), parameter_fns,
            lambda x: x, lambda x: x,
            hyperprior_num_filters, hyperprior_init_scale, hyperprior_tail_mass,
            hyperprior_broadcast_shape_bytes, prior_bytes_num_bytes,
            indexes_bound_gradient, quantize_indexes,
            init_scale, tail_mass, range_coder_precision
        )

    def forward(self, y, return_aux_loss: bool = True):
        y = minkowski_tensor_wrapped_op(y, torch.abs)
        return super(ScaleNoisyNormalEntropyModel, self).forward(y, return_aux_loss)

    def compress(self, y, return_dequantized: bool = False, estimate_bits: bool = False) \
            -> Union[Tuple[List[bytes], torch.Size, torch.Tensor],
                     Tuple[List[bytes], torch.Size, torch.Tensor, torch.Tensor]]:
        y = minkowski_tensor_wrapped_op(y, torch.abs)
        return super(ScaleNoisyNormalEntropyModel, self).compress(y, return_dequantized, estimate_bits)


class NoisyDeepFactorizedEntropyModel(EntropyModel):
    def __init__(self,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module,

                 hyperprior_batch_shape: torch.Size,
                 coding_ndim: int,

                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_shape_bytes: Tuple[int, ...] = (2,),
                 prior_bytes_num_bytes: int = 2,

                 index_ranges: Tuple[int, ...] = (16, 16, 16, 16),
                 parameter_fns_type: str = 'transform',
                 parameter_fns_factory: Callable[..., nn.Module] = None,
                 num_filters: Tuple[int, ...] = (1, 3, 3, 3, 1),
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        parameter_fns, indexes_view_fn, modules_to_add = \
            noisy_deep_factorized_indexed_entropy_model_init(
                index_ranges, parameter_fns_type, parameter_fns_factory, num_filters
            )
        super(NoisyDeepFactorizedEntropyModel, self).__init__(
            hyper_encoder, hyper_decoder,
            hyperprior_batch_shape, coding_ndim,
            NoisyDeepFactorized, index_ranges, parameter_fns,
            lambda x: x, indexes_view_fn,
            hyperprior_num_filters, hyperprior_init_scale, hyperprior_tail_mass,
            hyperprior_broadcast_shape_bytes, prior_bytes_num_bytes,
            indexes_bound_gradient, quantize_indexes,
            init_scale, tail_mass, range_coder_precision
        )
        for module_name, module in modules_to_add.items():
            setattr(self, module_name, module)