from typing import List, Tuple, Union, Optional, Dict, Any, Callable
import math

import numpy as np
import torch
import torch.distributions
from torch.distributions import Distribution

from .continuous_base import ContinuousEntropyModelBase
from .utils import lower_bound, upper_bound, quantization_offset


class ContinuousIndexedEntropyModel(ContinuousEntropyModelBase):
    def __init__(self,
                 prior_fn: Callable[[Any], Distribution],
                 index_ranges: List[int],
                 parameter_fns: Dict[str, Callable[[torch.Tensor],
                                                   Union[int, float, torch.Tensor]]],
                 coding_ndim: int,
                 additional_indexes_channel: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16):
        """
        The prior of ContinuousIndexedEntropyModel object is rebuilt during each forwarding.

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
        if len(index_ranges) == 1: assert not additional_indexes_channel
        else: assert additional_indexes_channel

        self.prior_fn = prior_fn
        self.parameter_fns = parameter_fns
        self.additional_indexes_channel = additional_indexes_channel
        self.index_ranges = index_ranges

        super(ContinuousIndexedEntropyModel, self).__init__(
            prior=self.make_range_coding_prior(),
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def make_prior(self, indexes):
        parameters = {k: f(indexes) for k, f in self.parameter_fns}
        return self.prior_fn(**parameters)

    def make_range_coding_prior(self):
        if self.additional_indexes_channel:
            indexes = torch.arange(self.index_ranges[0])
        else:
            indexes = [torch.arange(r) for r in self.index_ranges]
            indexes = torch.meshgrid(indexes)
            indexes = torch.stack(indexes, dim=-1)
        return self.make_prior(indexes)

    def normalize_indexes(self, indexes: torch.Tensor):
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
        else:
            # TODO: use self.prior during testing
            x_perturbed = self.quantize(x, offset=quantization_offset(prior))
        log_probs = prior.log_prob(x_perturbed)
        return x_perturbed, {'bits_loss': log_probs.sum() / (-math.log(2)),
                             'aux_loss': self.prior.aux_loss()}

    @torch.no_grad()
    def flatten_indexes(self, indexes):
        if not self.additional_indexes_channel:
            return indexes
        else:
            strides = torch.cumprod(
                torch.tensor([1] + self.index_ranges[:0:-1], device=indexes.device),
                dim=0)
            strides = torch.flip(strides, dims=[0])
            return torch.tensordot(indexes, strides, [[-1], [0]])

    @torch.no_grad()
    def compress(self, x, indexes):
        """
        x.shape == batch_shape + coding_unit_shape
        prior_batch_shape == *self.index_ranges
                (prior_event_shape == torch.Size([]))

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
        indexes = indexes.to(torch.int32)
        flat_indexes = self.flatten_indexes(indexes)
        assert flat_indexes.shape == input_shape

        # collapse batch dimensions
        flat_indexes = flat_indexes.reshape(-1, *coding_unit_shape)
        x = self.quantize(x, offset=quantization_offset(self.make_prior(indexes)))
        x = x.reshape(-1, *coding_unit_shape)

        strings = []
        for unit_idx in range(x.shape[0]):
            strings.append(self.range_encoder.encode_with_indexes(
                x[unit_idx].reshape[-1].tolist(),
                flat_indexes[unit_idx].tolist(),
                self.prior.cached_cdf_table.tolist(),
                self.prior.cached_cdf_length.tolist(),
                self.prior.cached_cdf_offset.tolist()
            ))
        strings = np.array(strings).reshape(batch_shape).tolist()
        return strings

    @torch.no_grad()
    def decompress(self, strings, indexes):
        strings = np.array(strings)
        input_shape = strings.shape
        strings = strings.reshape(-1)
        batch_shape = input_shape[:-self.coding_ndim]
        coding_unit_shape = input_shape[-self.coding_ndim:]

        indexes = self.normalize_indexes(indexes)
        flat_indexes = self.flatten_indexes(indexes)
        assert flat_indexes.shape == input_shape

        flat_indexes = flat_indexes.reshape(-1, *coding_unit_shape)

        symbols = []
        for s, i in zip(strings, flat_indexes):
            symbols.append(self.range_decoder.decode_with_indexes(
                s.item(),
                i,
                self.prior.cached_cdf_table.tolist(),
                self.prior.cached_cdf_length.tolist(),
                self.prior.cached_cdf_offset.tolist()
            ))
        symbols = torch.tensor(symbols)
        symbols = self.dequantize(symbols)
        symbols = symbols.reshape(batch_shape)
        return symbols
