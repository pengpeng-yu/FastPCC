from typing import List, Tuple, Union, Optional, Dict, Any, Callable
import math

import numpy as np
import torch
import torch.distributions
from torch.distributions import Distribution

from .continuous_base import ContinuousEntropyModelBase
from .utils import lower_bound, upper_bound, quantization_offset

from lib.torch_utils import minkowski_tensor_wrapped


class ContinuousIndexedEntropyModel(ContinuousEntropyModelBase):
    def __init__(self,
                 prior_fn: Callable[[Any], Distribution],
                 index_ranges: List[int],
                 parameter_fns: Dict[str, Callable[[Any], Union[int, float, torch.Tensor]]],
                 coding_ndim: int,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16):
        """
        The prior of ContinuousIndexedEntropyModel object is rebuilt
        in each forwarding during training.

        batch_shape of prior is determined by index_ranges and parameter_fns.
        `indexes` must have the same shape as the bottleneck tensor,
        with an additional dimension at position `channel_axis`.
        The values of the `k`th channel must be in the range
        `[0, index_ranges[k])`.

        If len(index_ranges) > 1,
        indexes are supposed to have an additional dimension at axis -1,
        and additional_indexes_channel is supposed to be True.

        parameter_fns should be derivable.

        Priors returned by prior_fn should have no member variable that is
        a learnable param.
        """
        if len(index_ranges) == 1:
            self.additional_indexes_channel = False
        else:
            self.additional_indexes_channel = True

        self.prior_fn = prior_fn
        self.parameter_fns = parameter_fns
        self.index_ranges = index_ranges

        super(ContinuousIndexedEntropyModel, self).__init__(
            prior=self.make_range_coding_prior(),
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def make_prior(self, indexes):
        parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
        return self.prior_fn(**parameters)

    def make_range_coding_prior(self) -> Distribution:
        """
        Make shared priors for generating cdf table.
        """
        if not self.additional_indexes_channel:
            indexes = torch.arange(self.index_ranges[0])
        else:
            indexes = [torch.arange(r) for r in self.index_ranges]
            indexes = torch.meshgrid(indexes)
            indexes = torch.stack(indexes, dim=-1)
        return self.make_prior(indexes)

    def normalize_indexes(self, indexes: torch.Tensor):
        """
        Return indexes within bounds.
        """
        indexes = lower_bound(indexes, 0)
        if not self.additional_indexes_channel:
            bounds = torch.tensor([self.index_ranges[0] - 1],
                                  dtype=torch.int32, device=indexes.device)
        else:
            bounds = torch.tensor([r - 1 for r in self.index_ranges],
                                  dtype=torch.int32, device=indexes.device)
            bounds_shape = [1] * (indexes.ndim - 1) + [len(self.index_ranges)]
            bounds = bounds.reshape(bounds_shape)
        indexes = upper_bound(indexes, bounds)
        return indexes

    @minkowski_tensor_wrapped('1->0 2->None')
    def forward(self, x: torch.Tensor, indexes: torch.Tensor) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, Dict[str, torch.Tensor], List]]:
        """
        x: torch.Tensor containing the data to be compressed.
        indexes: torch.Tensor that determines prior distribution of x.
        """
        indexes = self.normalize_indexes(indexes)
        prior = self.make_prior(indexes)
        if self.training:
            x_perturbed = self.perturb(x)
            log_probs = prior.log_prob(x_perturbed)
            return x_perturbed, {'bits_loss': log_probs.sum() / (-math.log(2)),
                                 'aux_loss': self.prior.aux_loss()}
        else:
            quantized_x = self.quantize(x, offset=quantization_offset(prior))
            log_probs = prior.log_prob(quantized_x)

            strings = self.compress(quantized_x, indexes=indexes, quantized=True)
            decompressed = self.decompress(strings, indexes=indexes)
            decompressed = decompressed.to(quantized_x.device)

            return decompressed, {'bits_loss': log_probs.sum() / (-math.log(2))}, strings

    @torch.no_grad()
    def flatten_indexes(self, indexes):
        """
        Return flat quantized indexes for cached cdf table.
        """
        indexes = indexes.to(torch.int32)
        if not self.additional_indexes_channel:
            return indexes
        else:
            strides = torch.cumprod(
                torch.tensor([1] + self.index_ranges[:0:-1], device=indexes.device),
                dim=0)
            strides = torch.flip(strides, dims=[0])
            return torch.tensordot(indexes, strides, [[-1], [0]])

    @torch.no_grad()
    def compress(self, x, indexes, quantized: bool = False) -> List:
        """
        x.shape == batch_shape + coding_unit_shape
        prior_batch_shape == self.index_ranges
        prior_event_shape == torch.Size([])

        if additional_indexes_channel is True:
            indexes.shape == x.shape + (len(self.index_ranges),)
        else: indexes.shape == x.shape

        flat_indexes.shape == x.shape
        flat_indexes map value in x to cached cdf index.
        """
        input_shape = x.shape
        batch_shape = input_shape[:-self.coding_ndim]
        coding_unit_shape = input_shape[-self.coding_ndim:]

        indexes = self.normalize_indexes(indexes)
        flat_indexes = self.flatten_indexes(indexes)
        assert flat_indexes.shape == input_shape

        # collapse batch dimensions
        flat_indexes = flat_indexes.reshape(-1, *coding_unit_shape)
        if not quantized:
            x = self.quantize(x, offset=quantization_offset(self.make_prior(indexes)))
        x = x.reshape(-1, *coding_unit_shape)

        strings = []
        for unit_idx in range(x.shape[0]):
            strings.append(
                self.range_encoder.encode_with_indexes(
                    x[unit_idx].reshape(-1).tolist(),
                    flat_indexes[unit_idx].reshape(-1).tolist(),
                    self.prior.cached_cdf_table_list,
                    self.prior.cached_cdf_length_list,
                    self.prior.cached_cdf_offset_list
                )
            )
        strings = np.array(strings).reshape(batch_shape).tolist()
        return strings

    @torch.no_grad()
    def decompress(self, strings, indexes):
        strings = np.array(strings)
        strings = strings.reshape(-1)

        indexes = self.normalize_indexes(indexes)
        flat_indexes = self.flatten_indexes(indexes)

        input_shape = flat_indexes.shape
        batch_shape = input_shape[:-self.coding_ndim]
        coding_unit_shape = input_shape[-self.coding_ndim:]
        assert batch_shape == strings.shape

        flat_indexes = flat_indexes.reshape(-1, *coding_unit_shape)

        symbols = []
        for s, i in zip(strings, flat_indexes):
            symbols.append(
                self.range_decoder.decode_with_indexes(
                    s.item(), i.reshape(-1).tolist(),
                    self.prior.cached_cdf_table_list,
                    self.prior.cached_cdf_length_list,
                    self.prior.cached_cdf_offset_list
                )
            )
        symbols = torch.tensor(symbols)
        symbols = self.dequantize(symbols, offset=quantization_offset(self.make_prior(indexes)))
        symbols = symbols.reshape(input_shape)
        return symbols


class LocationScaleIndexedEntropyModel(ContinuousIndexedEntropyModel):
    def __init__(self,
                 prior_fn: Callable[[Any], Distribution],
                 num_scales: int,
                 scale_fn: Dict[str, Callable[[Any], Union[int, float, torch.Tensor]]],
                 coding_ndim: int,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16):
        super(LocationScaleIndexedEntropyModel, self).__init__(
            prior_fn=prior_fn,
            index_ranges=[num_scales],
            parameter_fns={'loc': lambda _: 0, 'scale': scale_fn},
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    @minkowski_tensor_wrapped('1->0 2->None')
    def forward(self, x: torch.Tensor, scale_indexes: torch.Tensor, loc=None) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, Dict[str, torch.Tensor], List]]:
        if loc is not None:
            x = x - loc
        values, *ret = super(LocationScaleIndexedEntropyModel, self).forward(
            x, indexes=scale_indexes)
        if loc is not None:
            values = values + loc
        return (values, *ret)

    @torch.no_grad()
    def compress(self, x, indexes, quantized: bool = False, loc=None) -> List:
        if loc is not None:
            x -= loc
        return super(LocationScaleIndexedEntropyModel, self).compress(
            x, indexes=indexes, quantized=quantized)

    @torch.no_grad()
    def decompress(self, strings, indexes, loc=None):
        values = super(LocationScaleIndexedEntropyModel, self).decompress(
            strings, indexes=indexes)
        if loc is not None:
            values += loc
        return values
