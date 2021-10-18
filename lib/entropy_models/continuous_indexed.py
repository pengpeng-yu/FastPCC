from typing import List, Tuple, Union, Optional, Dict, Any, Callable
import math

import torch
import torch.distributions
from torch.distributions import Distribution

from .continuous_base import ContinuousEntropyModelBase
from .utils import lower_bound, upper_bound, quantization_offset

from lib.torch_utils import minkowski_tensor_wrapped_fn


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
                 range_coder_precision: int = 16):
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
            range_coder_precision=range_coder_precision
        )

        self.register_buffer('range_coding_prior_indexes',
                             range_coding_prior_indexes, persistent=False)

    def make_prior(self, indexes: torch.Tensor) -> Distribution:
        if self.quantize_indexes:
            if indexes.requires_grad:
                assert self.training
                indexes += (indexes.detach().round() - indexes.detach())
            else:
                indexes.round_()
        parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
        return self.prior_fn(**parameters)

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
                additive_uniform_noise: bool = True) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, List[bytes]]]:
        """
        x: torch.Tensor containing the data to be compressed.
        indexes: torch.Tensor that determines prior distribution of x.
        """
        indexes = self.normalize_indexes(indexes)
        prior = self.make_prior(indexes)
        if self.training:
            with torch.no_grad():
                self.prior.update_base(self.make_prior(self.range_coding_prior_indexes))
            if additive_uniform_noise is True:
                x_perturbed = self.perturb(x)
            else:
                x_perturbed = x
            log_probs = prior.log_prob(x_perturbed)
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

        # collapse batch dimensions
        flat_indexes = flat_indexes.reshape(-1, *coding_unit_shape)
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
        collapsed_x = quantized_x.reshape(-1, *coding_unit_shape)

        strings = []
        for unit_idx in range(collapsed_x.shape[0]):
            strings.append(
                self.range_encoder.encode_with_indexes(
                    collapsed_x[unit_idx].reshape(-1).tolist(),
                    flat_indexes[unit_idx].reshape(-1).tolist(),
                    self.prior.cached_cdf_table_list,
                    self.prior.cached_cdf_length_list,
                    self.prior.cached_cdf_offset_list
                )
            )

        if estimate_bits:
            estimated_bits = prior.log_prob(quantized_x).sum() / (-math.log(2))
            return strings, rounded_x_or_dequantized_x, estimated_bits

        else:
            return strings, rounded_x_or_dequantized_x

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({'<del>sparse_tensor_coords_tuple': 0})
    def decompress(self, strings: List[bytes],
                   indexes: torch.Tensor,
                   target_device: torch.device,
                   skip_dequantization: bool = False):
        indexes = self.normalize_indexes(indexes)
        flat_indexes = self.flatten_indexes(indexes)

        input_shape = flat_indexes.shape
        coding_unit_shape = input_shape[-self.coding_ndim:]

        flat_indexes = flat_indexes.reshape(-1, *coding_unit_shape)

        symbols = []
        for s, i in zip(strings, flat_indexes):
            symbols.append(
                self.range_decoder.decode_with_indexes(
                    s, i.reshape(-1).tolist(),
                    self.prior.cached_cdf_table_list,
                    self.prior.cached_cdf_length_list,
                    self.prior.cached_cdf_offset_list
                )
            )
        symbols = torch.tensor(symbols, device=target_device)
        if skip_dequantization:
            symbols = symbols.to(torch.float)
        else:
            symbols = self.dequantize(symbols, offset=quantization_offset(self.make_prior(indexes)))

        symbols = symbols.reshape(input_shape)
        return symbols


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
