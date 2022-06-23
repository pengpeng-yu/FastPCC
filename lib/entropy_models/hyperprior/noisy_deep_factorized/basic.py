from typing import List, Tuple, Union, Dict, Any, Callable
from functools import partial

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
                 hyperprior_init_scale: float = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_shape_bytes: Tuple[int, ...] = (2,),
                 prior_bytes_num_bytes: int = 2,

                 bottleneck_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 init_scale: float = 10,
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
            bottleneck_process=bottleneck_process,
            bottleneck_scaler=bottleneck_scaler,
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
            bottleneck_process=bottleneck_process,
            bottleneck_scaler=bottleneck_scaler,
            indexes_bound_gradient=indexes_bound_gradient,
            quantize_indexes=quantize_indexes,
            indexes_scaler=indexes_scaler,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def forward(self, y, is_first_forward: bool = True):
        if self.training:
            z = self.hyper_encoder_post_op(self.hyper_encoder(y))
            z_tilde, hyperprior_loss_dict = self.hyperprior_entropy_model(z, is_first_forward)
            indexes = self.hyper_decoder_post_op(self.hyper_decoder(z_tilde))
            y_tilde, prior_loss_dict = self.prior_entropy_model(y, indexes, is_first_forward)
            loss_dict = concat_loss_dicts(prior_loss_dict, hyperprior_loss_dict, lambda k: 'hyper_' + k)
            return y_tilde, loss_dict

        else:
            concat_bytes_list, coding_batch_shape, dequantized_y = self.compress(y)
            sparse_tensor_coords_tuple = get_minkowski_tensor_coords_tuple(y)
            y_recon = self.decompress(
                concat_bytes_list,
                coding_batch_shape,
                y.device,
                sparse_tensor_coords_tuple
            )
            return y_recon, concat_bytes_list, coding_batch_shape

    def compress(self, y, estimate_bits: bool = False) \
            -> Union[Tuple[List[bytes], torch.Size, torch.Tensor],
                     Tuple[List[bytes], torch.Size, torch.Tensor, torch.Tensor]]:
        z = self.hyper_encoder_post_op(self.hyper_encoder(y))
        prior_bytes_list, coding_batch_shape, z_recon, *estimated_prior_bits = \
            self.hyperprior_entropy_model.compress(
                z, estimate_bits=estimate_bits
            )
        indexes = self.hyper_decoder_post_op(self.hyper_decoder(z_recon))
        bytes_list, dequantized_y, *estimated_bits = \
            self.prior_entropy_model.compress(
                y, indexes, estimate_bits=estimate_bits
            )
        concat_bytes_list = self.concat_bytes_lists(prior_bytes_list, bytes_list)
        if estimated_bits:
            return concat_bytes_list, coding_batch_shape, dequantized_y, \
                   estimated_prior_bits[0] + estimated_bits[0]
        else:
            return concat_bytes_list, coding_batch_shape, dequantized_y,

    def decompress(self,
                   concat_bytes_list: List[bytes],
                   coding_batch_shape: torch.Size,
                   target_device: torch.device,
                   sparse_tensor_coords_tuple: Tuple = None) -> Any:
        prior_bytes_list, bytes_list = self.split_bytes_lists(concat_bytes_list)
        z_recon = self.hyperprior_entropy_model.decompress(
            prior_bytes_list, coding_batch_shape, target_device,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )
        pre_indexes = self.hyper_decoder(z_recon)
        sparse_tensor_coords_tuple = get_minkowski_tensor_coords_tuple(pre_indexes)
        indexes = self.hyper_decoder_post_op(pre_indexes)
        y_recon = self.prior_entropy_model.decompress(
            bytes_list, indexes, target_device,
            sparse_tensor_coords_tuple=sparse_tensor_coords_tuple
        )
        return y_recon

    def concat_bytes_lists(self, prior_bytes_list: List[bytes], bytes_list: List[bytes]) -> List[bytes]:
        return [len(i).to_bytes(self.prior_bytes_num_bytes, 'little', signed=False) + i + j
                for i, j in zip(prior_bytes_list, bytes_list)]

    def split_bytes_lists(self, concat_bytes_list: List[bytes]) -> Tuple[List[bytes], List[bytes]]:
        prior_bytes_list = []
        bytes_list = []
        for concat_b in concat_bytes_list:
            prior_bytes_len = int.from_bytes(
                concat_b[:self.prior_bytes_num_bytes],
                'little', signed=False
            )
            prior_bytes_list.append(
                concat_b[self.prior_bytes_num_bytes:
                         self.prior_bytes_num_bytes + prior_bytes_len]
            )
            bytes_list.append(
                concat_b[self.prior_bytes_num_bytes + prior_bytes_len:]
            )
        return prior_bytes_list, bytes_list


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
                 hyperprior_init_scale: float = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_shape_bytes: Tuple[int, ...] = (2,),
                 prior_bytes_num_bytes: int = 2,

                 bottleneck_process: str = 'noise',
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 init_scale: float = 10,
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
            bottleneck_process, 1,
            indexes_bound_gradient, quantize_indexes, indexes_scaler,
            init_scale, tail_mass, range_coder_precision
        )

    def forward(self, y, is_first_forward: bool = True):
        y = minkowski_tensor_wrapped_op(y, torch.abs)
        return super(ScaleNoisyNormalEntropyModel, self).forward(y, is_first_forward)

    def compress(self, y, estimate_bits: bool = False) \
            -> Union[Tuple[List[bytes], torch.Size, torch.Tensor],
                     Tuple[List[bytes], torch.Size, torch.Tensor, torch.Tensor]]:
        y = minkowski_tensor_wrapped_op(y, torch.abs)
        return super(ScaleNoisyNormalEntropyModel, self).compress(y, estimate_bits)


class NoisyDeepFactorizedEntropyModel(EntropyModel):
    def __init__(self,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module,

                 hyperprior_batch_shape: torch.Size,
                 coding_ndim: int,

                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: float = 10,
                 hyperprior_tail_mass: float = 2 ** -8,
                 hyperprior_broadcast_shape_bytes: Tuple[int, ...] = (2,),
                 prior_bytes_num_bytes: int = 2,

                 index_ranges: Tuple[int, ...] = (16, 16, 16, 16),
                 parameter_fns_type: str = 'transform',
                 parameter_fns_factory: Callable[..., nn.Module] = None,
                 num_filters: Tuple[int, ...] = (1, 3, 3, 3, 1),

                 bottleneck_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 init_scale: float = 10,
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
            partial(NoisyDeepFactorized, noise_width=1 / bottleneck_scaler),
            index_ranges, parameter_fns, lambda x: x, indexes_view_fn,
            hyperprior_num_filters, hyperprior_init_scale, hyperprior_tail_mass,
            hyperprior_broadcast_shape_bytes, prior_bytes_num_bytes,
            bottleneck_process, bottleneck_scaler,
            indexes_bound_gradient, quantize_indexes, indexes_scaler,
            init_scale, tail_mass, range_coder_precision
        )
        for module_name, module in modules_to_add.items():
            setattr(self, module_name, module)

    def _apply(self, fn):
        super(NoisyDeepFactorizedEntropyModel, self)._apply(fn)
        self.prior_entropy_model.update_prior()
