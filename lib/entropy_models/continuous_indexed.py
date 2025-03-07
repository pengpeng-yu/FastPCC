from typing import List, Tuple, Union, Dict, Any, Callable
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributions
from torch.distributions import Distribution

from .continuous_base import ContinuousEntropyModelBase
from .utils import lower_bound, upper_bound, grad_scaler

from lib.minkowski_sparse_conv_layers import minkowski_tensor_wrapped_op, minkowski_tensor_wrapped_fn


class ContinuousIndexedEntropyModel(ContinuousEntropyModelBase):
    def __init__(self,
                 prior_fn: Callable[..., Distribution],
                 index_ranges: Tuple[int, ...],
                 parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],
                 coding_ndim: int,
                 bottleneck_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 quantize_bottleneck_in_eval: bool = True,
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 indexes_offset: float = 0,
                 lower_bound: Union[int, torch.Tensor] = -64,
                 upper_bound: Union[int, torch.Tensor] = 64,
                 batch_shape: torch.Size = torch.Size([1]),
                 overflow_coding: bool = True):
        """
        batch_shape of prior is determined by index_ranges and parameter_fns.
        `indexes` must have the same shape as the bottleneck tensor,
        with an additional dimension at innermost axis if len(index_ranges) > 1.
        The values of the `k`th channel must be in the range
        `[0, index_ranges[k])`.

        parameter_fns should be derivable.

        Priors returned by prior_fn should have no member variable that is
        a learnable param.
        """
        self.additional_indexes_dim = len(index_ranges) != 1
        self.prior_fn = prior_fn
        self.parameter_fns = parameter_fns
        self.bottleneck_scaler = bottleneck_scaler
        self.quantize_bottleneck_in_eval = quantize_bottleneck_in_eval
        self.index_ranges = index_ranges
        self.indexes_bound_gradient = indexes_bound_gradient
        self.quantize_indexes = quantize_indexes
        self.indexes_scaler = indexes_scaler
        self.indexes_offset = indexes_offset
        range_coding_prior_indexes = self.make_range_coding_prior_indexes()
        with torch.no_grad():
            prior = self.make_prior(range_coding_prior_indexes)
        super(ContinuousIndexedEntropyModel, self).__init__(
            prior=prior,
            coding_ndim=coding_ndim,
            bottleneck_process=bottleneck_process,
            bottleneck_scaler=bottleneck_scaler,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            batch_shape=batch_shape,
            overflow_coding=overflow_coding
        )
        # TODO: mark this buffer as ignored in DDP broadcasting.
        self.register_buffer('range_coding_prior_indexes',
                             range_coding_prior_indexes, persistent=False)

    def make_prior(self, indexes: torch.Tensor) -> Distribution:
        if indexes.requires_grad:
            assert self.training
            if self.quantize_indexes:
                indexes = indexes + (indexes.detach().round() - indexes.detach())
        else:
            indexes = indexes.round()
        if self.indexes_scaler != 0:
            indexes = indexes / self.indexes_scaler
        else:
            indexes = (indexes / torch.tensor(
                [r - 1 for r in self.index_ranges], dtype=indexes.dtype, device=indexes.device
            ) - 0.5) * 2
        parameters = {k: f(indexes - self.indexes_offset) for k, f in self.parameter_fns.items()}
        return self.prior_fn(**parameters)

    @torch.no_grad()
    def update_prior(self):
        self.prior.update_base(self.make_prior(self.range_coding_prior_indexes))

    def make_range_coding_prior_indexes(self) -> torch.Tensor:
        """
        Make shared indexes for generating cdf table.
        """
        if not self.additional_indexes_dim:
            indexes = torch.arange(self.index_ranges[0])
        else:
            indexes = [torch.arange(r) for r in self.index_ranges]
            indexes = torch.meshgrid(*indexes, indexing='ij')
            indexes = torch.stack(indexes, dim=-1)
        indexes = indexes.to(torch.float)
        return indexes

    def bound_indexes(self, indexes: torch.Tensor):
        """
        Return indexes within bounds.
        """
        indexes = indexes + self.indexes_offset
        if self.indexes_scaler != 0:
            indexes = indexes * self.indexes_scaler
        else:
            indexes = (indexes / 2 + 0.5) * torch.tensor(
                [r - 1 for r in self.index_ranges], dtype=indexes.dtype, device=indexes.device
            )
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
                is_first_forward: bool = True,
                x_grad_scaler_for_bits_loss: float = 1.0) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, List[bytes]]]:
        """
        x: torch.Tensor containing the data to be compressed.
        indexes: torch.Tensor that determines prior distribution of x.
        """
        if self.bottleneck_scaler != 1:
            x = x * self.bottleneck_scaler
        if self.training:
            indexes = self.bound_indexes(indexes)
            prior = self.make_prior(indexes)
            if is_first_forward:
                self.update_prior()
            processed_x = self.process(x)
            if self.bottleneck_scaler != 1:
                processed_x = processed_x / self.bottleneck_scaler
            log_probs = prior.log_prob(grad_scaler(processed_x, x_grad_scaler_for_bits_loss))
            loss_dict = {'bits_loss': log_probs.sum() / (-math.log(2))}
            return processed_x, loss_dict

        else:
            bytes_list, _ = self.compress(x, indexes)
            decompressed = self.decompress(bytes_list, indexes, x.device)
            return decompressed, bytes_list

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
        if self.bottleneck_scaler != 1:
            x = x * self.bottleneck_scaler
        input_shape = x.shape
        coding_unit_shape = input_shape[-self.coding_ndim:]

        indexes = self.bound_indexes(indexes)
        flat_indexes = self.flatten_indexes(indexes)
        assert flat_indexes.shape == input_shape

        # collapse batch dimensions and coding_unit dimensions
        flat_indexes = flat_indexes.reshape(-1, coding_unit_shape.numel())
        if estimate_bits is True:
            prior = self.make_prior(indexes)
        if self.quantize_bottleneck_in_eval is True:
            quantized_x, dequantized_x = self.quantize(
                x, offset=0
            )
        else:
            dequantized_x = x
            quantized_x = x.to(torch.int32)
        collapsed_x = quantized_x.reshape(-1, coding_unit_shape.numel())

        bytes_list = self.prior.range_coder.encode_with_indexes(
            collapsed_x.cpu().numpy(),
            flat_indexes.cpu().numpy()
        )

        if self.bottleneck_scaler != 1:
            dequantized_x = dequantized_x / self.bottleneck_scaler
        if estimate_bits is True:
            estimated_bits = prior.log_prob(
                quantized_x / self.bottleneck_scaler
            ).sum() / (-math.log(2))
            return bytes_list, dequantized_x, estimated_bits
        else:
            return bytes_list, dequantized_x

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({'<del>sparse_tensor_coords_tuple': 0, 2: None})
    def decompress(self, bytes_list: List[bytes],
                   indexes: torch.Tensor,
                   target_device: torch.device):
        indexes = self.bound_indexes(indexes)
        flat_indexes = self.flatten_indexes(indexes)

        input_shape = flat_indexes.shape
        coding_unit_shape = input_shape[-self.coding_ndim:]
        flat_indexes = flat_indexes.reshape(-1, coding_unit_shape.numel()).cpu().numpy()

        symbols = np.empty_like(flat_indexes)
        self.prior.range_coder.decode_with_indexes(
            bytes_list, flat_indexes, symbols
        )
        symbols = torch.from_numpy(symbols).to(target_device)

        if self.quantize_bottleneck_in_eval is True:
            symbols = self.dequantize(symbols, offset=0)
        else:
            symbols = symbols.to(torch.float)
        symbols = symbols.reshape(input_shape)
        if self.bottleneck_scaler != 1:
            symbols /= self.bottleneck_scaler
        return symbols

    def train(self, mode: bool = True):
        """
        Use model.eval() to update the prior function.
        """
        if mode is False:
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
    assert len(num_filters) >= 2 and num_filters[0] == num_filters[-1] == 1
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
        assert index_channels == factors_param_cum_numel[-1], f'{index_channels} != {factors_param_cum_numel[-1]}'
        parameter_fns = {
            'batch_shape': lambda i: i.shape[:-1],
            'weights': lambda i: [i[..., weights_param_cum_numel[_]: weights_param_cum_numel[_ + 1]].view
                                  (-1, num_filters[_ + 1], num_filters[_])
                                  for _ in range(len(weights_param_numel))],

            'biases': lambda i: [i[..., biases_param_cum_numel[_]: biases_param_cum_numel[_ + 1]].view
                                 (-1, biases_param_numel[_], 1)
                                 for _ in range(len(biases_param_numel))],

            'factors': lambda i: [i[..., factors_param_cum_numel[_]: factors_param_cum_numel[_ + 1]].view
                                  (-1, factors_param_numel[_], 1)
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
