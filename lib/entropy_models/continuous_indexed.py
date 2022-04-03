from typing import List, Tuple, Union, Dict, Any, Callable
import math

import torch
import torch.nn as nn
import torch.distributions
from torch.distributions import Distribution

from .distributions.uniform_noise import NoisyDeepFactorized
from .continuous_base import ContinuousEntropyModelBase
from .utils import lower_bound, upper_bound, quantization_offset, grad_scaler

from lib.torch_utils import minkowski_tensor_wrapped_fn, minkowski_tensor_wrapped_op


class ContinuousIndexedEntropyModel(ContinuousEntropyModelBase):
    def __init__(self,
                 prior_fn: Callable[..., Distribution],
                 index_ranges: Tuple[int, ...],
                 parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],
                 coding_ndim: int,
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 lower_bound: Union[int, torch.Tensor] = 0,
                 upper_bound: Union[int, torch.Tensor] = -1,
                 range_coder_precision: int = 16,
                 overflow_coding: bool = True):
        """
        The prior of ContinuousIndexedEntropyModel object is rebuilt
        in each forwarding during training.

        batch_shape of prior is determined by index_ranges and parameter_fns.
        `indexes` must have the same shape as the bottleneck tensor,
        with an additional dimension at innermost axis if len(index_ranges) > 1.
        The values of the `k`th channel must be in the range
        `[0, index_ranges[k])`.

        parameter_fns should be derivable.

        Priors returned by prior_fn should have no member variable that is
        a learnable param.
        """
        if len(index_ranges) == 1:
            self.additional_indexes_dim = False
        else:
            self.additional_indexes_dim = True
        self.prior_fn = prior_fn
        self.parameter_fns = parameter_fns
        self.index_ranges = index_ranges
        self.indexes_bound_gradient = indexes_bound_gradient
        self.quantize_indexes = quantize_indexes
        range_coding_prior_indexes = self.make_range_coding_prior()
        with torch.no_grad():
            prior = self.make_prior(range_coding_prior_indexes)
        super(ContinuousIndexedEntropyModel, self).__init__(
            prior=prior,
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            range_coder_precision=range_coder_precision,
            overflow_coding=overflow_coding
        )
        # TODO: mark this buffer as ignored in DDP broadcasting.
        self.register_buffer('range_coding_prior_indexes',
                             range_coding_prior_indexes, persistent=False)

    def make_prior(self, indexes: torch.Tensor) -> Distribution:
        if self.quantize_indexes:
            if indexes.requires_grad:
                assert self.training
                indexes += (indexes.detach().round() - indexes.detach())
            else:
                torch.round_(indexes)
        parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
        return self.prior_fn(**parameters)

    @torch.no_grad()
    def update_prior(self):
        self.prior.update_base(self.make_prior(self.range_coding_prior_indexes))

    def make_range_coding_prior(self) -> torch.Tensor:
        """
        Make shared priors for generating cdf table.
        """
        if not self.additional_indexes_dim:
            indexes = torch.arange(self.index_ranges[0])
        else:
            indexes = [torch.arange(r) for r in self.index_ranges]
            indexes = torch.meshgrid(indexes)
            indexes = torch.stack(indexes, dim=-1)
        indexes = indexes.to(torch.float)
        return indexes

    def normalize_indexes(self, indexes: torch.Tensor):
        """
        Return indexes within bounds.
        """
        indexes = lower_bound(indexes, 0, self.indexes_bound_gradient)
        if not self.additional_indexes_dim:
            bounds = torch.tensor([self.index_ranges[0] - 1],
                                  dtype=torch.int32, device=indexes.device)
        else:
            bounds = torch.tensor([r - 1 for r in self.index_ranges],
                                  dtype=torch.int32, device=indexes.device)
            bounds_shape = [1] * (indexes.ndim - 1) + [len(self.index_ranges)]
            bounds = bounds.reshape(bounds_shape)
        indexes = upper_bound(indexes, bounds, self.indexes_bound_gradient)
        return indexes

    @minkowski_tensor_wrapped_fn({1: 0, 2: None})
    def forward(self, x: torch.Tensor, indexes: torch.Tensor,
                return_aux_loss: bool = True,
                additive_uniform_noise: bool = True,
                x_grad_scaler_for_bits_loss: float = 1.0) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, List[bytes]]]:
        """
        x: torch.Tensor containing the data to be compressed.
        indexes: torch.Tensor that determines prior distribution of x.
        """
        indexes = self.normalize_indexes(indexes)
        prior = self.make_prior(indexes)
        if self.training:
            self.update_prior()
            if additive_uniform_noise is True:
                x_perturbed = self.perturb(x)
            else:
                x_perturbed = x
            log_probs = prior.log_prob(grad_scaler(x_perturbed, x_grad_scaler_for_bits_loss))
            loss_dict = {'bits_loss': log_probs.sum() / (-math.log(2))}
            if return_aux_loss:
                loss_dict['aux_loss'] = self.prior.aux_loss()
            return x_perturbed, loss_dict

        else:
            strings, _ = self.compress(x, indexes)
            decompressed = self.decompress(strings, indexes, x.device)
            return decompressed, strings

    @torch.no_grad()
    def flatten_indexes(self, indexes):
        """
        Return flat int32 indexes for cached cdf table.
        """
        indexes = indexes.round()
        if not self.additional_indexes_dim:
            indexes = indexes.to(torch.int32)
            return indexes
        else:
            strides = torch.cumprod(
                torch.tensor((1, *self.index_ranges[:0:-1]),
                             device=indexes.device,
                             dtype=torch.float), dim=0, dtype=torch.float)
            strides = torch.flip(strides, dims=[0])
            return torch.tensordot(indexes, strides, [[-1], [0]]).to(torch.int32)

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({1: 1, 2: None})
    def compress(self, x: torch.Tensor,
                 indexes: torch.Tensor,
                 skip_quantization: bool = False,
                 return_dequantized: bool = False,
                 estimate_bits: bool = False) \
            -> Union[Tuple[List[bytes], torch.Tensor],
                     Tuple[List[bytes], torch.Tensor, torch.Tensor]]:
        """
        x.shape == batch_shape + coding_unit_shape
        prior_batch_shape == self.index_ranges
        prior_event_shape == torch.Size([])

        if additional_indexes_dim is True:
            indexes.shape == x.shape + (len(self.index_ranges),)
        else: indexes.shape == x.shape

        flat_indexes.shape == x.shape
        flat_indexes map value in x to cached cdf index.
        """
        input_shape = x.shape
        coding_unit_shape = input_shape[-self.coding_ndim:]

        indexes = self.normalize_indexes(indexes)
        flat_indexes = self.flatten_indexes(indexes)
        assert flat_indexes.shape == input_shape

        # collapse batch dimensions and coding_unit dimensions
        flat_indexes = flat_indexes.reshape(-1, coding_unit_shape.numel())
        prior = self.make_prior(indexes)
        if skip_quantization is True:
            if return_dequantized:
                rounded_x_or_dequantized_x = self.dequantize(x, offset=quantization_offset(prior))
            else:
                rounded_x_or_dequantized_x = x
            quantized_x = x.to(torch.int32)
        else:
            quantized_x, rounded_x_or_dequantized_x = self.quantize(
                x, offset=quantization_offset(prior), return_dequantized=return_dequantized
            )
        collapsed_x = quantized_x.reshape(-1, coding_unit_shape.numel())

        strings = self.prior.range_coder.encode_with_indexes(
            collapsed_x.tolist(),
            flat_indexes.tolist()
        )

        if estimate_bits:
            estimated_bits = prior.log_prob(quantized_x).sum() / (-math.log(2))
            return strings, rounded_x_or_dequantized_x, estimated_bits
        else:
            return strings, rounded_x_or_dequantized_x

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({'<del>sparse_tensor_coords_tuple': 0, 2: None})
    def decompress(self, strings: List[bytes],
                   indexes: torch.Tensor,
                   target_device: torch.device,
                   skip_dequantization: bool = False):
        indexes = self.normalize_indexes(indexes)
        flat_indexes = self.flatten_indexes(indexes)

        input_shape = flat_indexes.shape
        coding_unit_shape = input_shape[-self.coding_ndim:]
        flat_indexes = flat_indexes.reshape(-1, coding_unit_shape.numel())

        symbols = self.prior.range_coder.decode_with_indexes(
            strings, flat_indexes.tolist()
        )
        symbols = torch.tensor(symbols, device=target_device)
        if skip_dequantization:
            symbols = symbols.to(torch.float)
        else:
            symbols = self.dequantize(symbols, offset=quantization_offset(self.make_prior(indexes)))
        symbols = symbols.reshape(input_shape)
        return symbols

    def train(self, mode: bool = True):
        """
        Use model.train() to invalidate cached cdf table.
        Use model.eval() to update the prior function and the cdf table.
        """
        if mode is False and self.prior.requires_updating_cdf_table:
            self.update_prior()
        return super(ContinuousIndexedEntropyModel, self).train(mode=mode)


def noisy_scale_normal_indexed_entropy_model_init(scale_min: float, scale_max: float, num_scales: int) -> \
        Dict[str, Callable[[torch.Tensor], Any]]:
    offset = math.log(scale_min)
    factor = (math.log(scale_max) - math.log(scale_min)) / (num_scales - 1)
    parameter_fns = {
        'loc': lambda _: 0,
        'scale': lambda i: torch.exp(offset + factor * i)
    }
    return parameter_fns


def noisy_deep_factorized_indexed_entropy_model_init(
        index_ranges: Tuple[int, ...],
        parameter_fns_type: str,
        parameter_fns_factory: Callable[..., nn.Module],
        num_filters: Tuple[int, ...]
) -> Tuple[Dict[str, Callable[[torch.Tensor], Any]], Callable, Dict[str, nn.Module]]:
    assert len(num_filters) >= 3 and num_filters[0] == num_filters[-1] == 1
    assert parameter_fns_type in ('split', 'transform')
    if not len(index_ranges) > 1:
        raise NotImplementedError
    index_channels = len(index_ranges)

    def indexes_view_fn(x):
        return minkowski_tensor_wrapped_op(
            x,
            lambda x: x.view(
                *x.shape[:-1],
                x.shape[-1] // index_channels,
                index_channels
            ),
            needs_recover=False,
            add_batch_dim=True
        )

    weights_param_numel = [num_filters[i] * num_filters[i + 1] for i in range(len(num_filters) - 1)]
    biases_param_numel = num_filters[1:]
    factors_param_numel = num_filters[1:-1]

    if parameter_fns_type == 'split':
        weights_param_cum_numel = torch.cumsum(torch.tensor([0, *weights_param_numel]), dim=0)
        biases_param_cum_numel = \
            torch.cumsum(torch.tensor([0, *biases_param_numel]), dim=0) + weights_param_cum_numel[-1]
        factors_param_cum_numel = \
            torch.cumsum(torch.tensor([0, *factors_param_numel]), dim=0) + biases_param_cum_numel[-1]
        assert index_channels == factors_param_cum_numel[-1]
        parameter_fns = {
            'batch_shape': lambda i: i.shape[:-1],
            'weights': lambda i: [i[..., weights_param_cum_numel[_]: weights_param_cum_numel[_ + 1]].view
                                  (-1, num_filters[_ + 1], num_filters[_]) / 3 - 0.5
                                  for _ in range(len(weights_param_numel))],

            'biases': lambda i: [i[..., biases_param_cum_numel[_]: biases_param_cum_numel[_ + 1]].view
                                 (-1, biases_param_numel[_], 1) / 3 - 0.5
                                 for _ in range(len(biases_param_numel))],

            'factors': lambda i: [i[..., factors_param_cum_numel[_]: factors_param_cum_numel[_ + 1]].view
                                  (-1, factors_param_numel[_], 1) / 3 - 0.5
                                  for _ in range(len(factors_param_numel))],
        }
        return parameter_fns, indexes_view_fn, {}

    elif parameter_fns_type == 'transform':
        prior_indexes_weights_transforms = nn.ModuleList(
            [parameter_fns_factory(index_channels, out_channels)
             for out_channels in weights_param_numel]
        )
        prior_indexes_biases_transforms = nn.ModuleList(
            [parameter_fns_factory(index_channels, out_channels)
             for out_channels in biases_param_numel]
        )
        prior_indexes_factors_transforms = nn.ModuleList(
            [parameter_fns_factory(index_channels, out_channels)
             for out_channels in factors_param_numel]
        )
        parameter_fns = {
            'batch_shape': lambda i: i.shape[:-1],
            'weights': lambda i: [transform(i).view(-1, num_filters[_ + 1], num_filters[_])
                                  for _, transform in enumerate(prior_indexes_weights_transforms)],

            'biases': lambda i: [transform(i).view(-1, biases_param_numel[_], 1)
                                 for _, transform in enumerate(prior_indexes_biases_transforms)],

            'factors': lambda i: [transform(i).view(-1, factors_param_numel[_], 1)
                                  for _, transform in enumerate(prior_indexes_factors_transforms)],
        }
        return parameter_fns, indexes_view_fn, {
            'prior_indexes_weights_transforms': prior_indexes_weights_transforms,
            'prior_indexes_biases_transforms': prior_indexes_biases_transforms,
            'prior_indexes_factors_transforms': prior_indexes_factors_transforms}

    else:
        raise NotImplementedError


class ContinuousNoisyDeepFactorizedIndexedEntropyModel(ContinuousIndexedEntropyModel):
    def __init__(self,
                 index_ranges: Tuple[int, ...],
                 coding_ndim: int,
                 parameter_fns_type: str = 'transform',
                 parameter_fns_factory: Callable[..., nn.Module] = None,
                 num_filters: Tuple[int, ...] = (1, 3, 3, 3, 1),
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 lower_bound: Union[int, torch.Tensor] = 0,
                 upper_bound: Union[int, torch.Tensor] = -1,
                 range_coder_precision: int = 16,
                 overflow_coding: bool = True):
        parameter_fns, self.indexes_view_fn, modules_to_add = \
            noisy_deep_factorized_indexed_entropy_model_init(
                index_ranges, parameter_fns_type, parameter_fns_factory, num_filters
            )
        super(ContinuousNoisyDeepFactorizedIndexedEntropyModel, self).__init__(
            NoisyDeepFactorized, index_ranges, parameter_fns,
            coding_ndim, indexes_bound_gradient, quantize_indexes,
            init_scale, tail_mass, lower_bound, upper_bound,
            range_coder_precision, overflow_coding
        )
        for module_name, module in modules_to_add.items():
            setattr(self, module_name, module)

    @minkowski_tensor_wrapped_fn({1: 0, 2: None})
    def forward(self, x: torch.Tensor, indexes: torch.Tensor,
                return_aux_loss: bool = True,
                additive_uniform_noise: bool = True) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, List[bytes]]]:
        return super(ContinuousNoisyDeepFactorizedIndexedEntropyModel, self).forward(
            x, self.indexes_view_fn(indexes), return_aux_loss, additive_uniform_noise
        )

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({1: 1, 2: None})
    def compress(self, x: torch.Tensor,
                 indexes: torch.Tensor,
                 skip_quantization: bool = False,
                 return_dequantized: bool = False,
                 estimate_bits: bool = False) \
            -> Union[Tuple[List[bytes], torch.Tensor],
                     Tuple[List[bytes], torch.Tensor, torch.Tensor]]:
        return super(ContinuousNoisyDeepFactorizedIndexedEntropyModel, self).compress(
            x, self.indexes_view_fn(indexes), skip_quantization, return_dequantized, estimate_bits
        )

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({'<del>sparse_tensor_coords_tuple': 0, 2: None})
    def decompress(self, strings: List[bytes],
                   indexes: torch.Tensor,
                   target_device: torch.device,
                   skip_dequantization: bool = False):
        return super(ContinuousNoisyDeepFactorizedIndexedEntropyModel, self).decompress(
            strings, self.indexes_view_fn(indexes), target_device, skip_dequantization
        )

    def _apply(self, fn):
        super(ContinuousIndexedEntropyModel, self)._apply(fn)
        self.update_prior()


class LocationScaleIndexedEntropyModel(ContinuousIndexedEntropyModel):
    def __init__(self,
                 prior_fn: Callable[..., Distribution],
                 num_scales: int,
                 scale_fn: Dict[str, Callable[..., Union[int, float, torch.Tensor]]],
                 coding_ndim: int,
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16):
        super(LocationScaleIndexedEntropyModel, self).__init__(
            prior_fn=prior_fn,
            index_ranges=(num_scales,),
            parameter_fns={'loc': lambda _: 0, 'scale': scale_fn},
            coding_ndim=coding_ndim,
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def forward(self, x: torch.Tensor, scale_indexes: torch.Tensor, loc=None,
                return_aux_loss: bool = True,
                additive_uniform_noise: bool = True) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, List[bytes]]]:
        if loc is not None:
            x = x - loc
        values, *ret = super(LocationScaleIndexedEntropyModel, self).forward(
            x, scale_indexes, return_aux_loss, additive_uniform_noise)
        if loc is not None:
            values = values + loc
        return (values, *ret)

    @torch.no_grad()
    def compress(self, x, indexes, loc=None):
        if loc is not None:
            x -= loc
        return super(LocationScaleIndexedEntropyModel, self).compress(x, indexes)

    @torch.no_grad()
    def decompress(self, strings, indexes, loc=None):
        values = super(LocationScaleIndexedEntropyModel, self).decompress(
            strings, indexes)
        if loc is not None:
            values += loc
        return values
