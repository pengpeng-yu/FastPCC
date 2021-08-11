from typing import List, Tuple, Dict, Union, Sequence, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributions
from torch.distributions import Distribution

from .distributions.deep_factorized import DeepFactorized
from .distributions.uniform_noise import NoisyDeepFactorized
from .continuous_base import ContinuousEntropyModelBase

from lib.torch_utils import minkowski_tensor_wrapped


class ContinuousBatchedEntropyModel(ContinuousEntropyModelBase):
    def __init__(self,
                 prior: Distribution,
                 coding_ndim: int,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16):
        """
        Generally, prior object should have parameters on target device when being
        constructed, since functions of nn.Module like "to()" and "cuda()" have
        no effect on it.

        The innermost `self.coding_ndim` dimensions are treated as one coding unit,
        i.e. are compressed into one string each. Any additional dimensions to the
        left are treated as batch dimensions.

        The innermost dimensions of input tensor are supposed to be the same
        as self.prior_shape.
        """
        assert coding_ndim >= prior.batch_ndim
        super(ContinuousBatchedEntropyModel, self).__init__(
            prior=prior,
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision,
        )

    def build_indexes(self, broadcast_shape: torch.Size):
        indexes = torch.arange(self.prior.batch_numel, dtype=torch.int32)
        indexes = indexes.reshape(self.prior.batch_shape)
        indexes = indexes.repeat(*broadcast_shape,
                                 *[1] * len(self.prior.batch_shape))
        return indexes

    @minkowski_tensor_wrapped({1: 0})
    def forward(self, x) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, Dict[str, torch.Tensor], List]]:
        """
        x: `torch.Tensor` containing the data to be compressed. Must have at
        least `self.coding_ndim` dimensions, and the innermost dimensions must
        be broadcastable to `self.prior_shape`.
        """
        if self.training:
            x_perturbed = self.perturb(x)
            log_probs = self.prior.log_prob(x_perturbed)
            return x_perturbed, {'bits_loss': log_probs.sum() / (-math.log(2)),
                                 'aux_loss': self.prior.aux_loss()}
        else:
            quantized_x = self.quantize(x)
            log_probs = self.prior.log_prob(quantized_x)
            # TODO: check self.prior.log_prob(self.perturb(x)).sum() / (-math.log(2))

            strings, broadcast_shape = self.compress(quantized_x, quantized=True)
            decompressed = self.decompress(strings, broadcast_shape)
            decompressed = decompressed.to(quantized_x.device)

            return decompressed, {'bits_loss': log_probs.sum() / (-math.log(2))}, strings

    @torch.no_grad()
    def compress(self, x: torch.Tensor, quantized: bool = False) \
            -> Tuple[List, torch.Size]:
        """
        x.shape = batch_shape + coding_unit_shape
                = batch_shape + broadcast_shape + prior_batch_shape
                (prior_event_shape = torch.Size([]))
        """
        input_shape = x.shape
        batch_shape = input_shape[:-self.coding_ndim]
        coding_unit_shape = input_shape[-self.coding_ndim:]
        broadcast_shape = coding_unit_shape[:-len(self.prior.batch_shape)]

        if not quantized: x = self.quantize(x)
        x = x.reshape(-1, *coding_unit_shape)  # collapse batch dimensions
        indexes = self.build_indexes(broadcast_shape)  # shape: coding_unit_shape

        strings = []
        indexes = indexes.reshape(-1).tolist()
        for unit_idx in range(x.shape[0]):
            strings.append(
                self.range_encoder.encode_with_indexes(
                    x[unit_idx].reshape(-1).tolist(),
                    indexes,
                    self.prior.cached_cdf_table_list,
                    self.prior.cached_cdf_length_list,
                    self.prior.cached_cdf_offset_list
                )
            )
        strings = np.array(strings).reshape(batch_shape).tolist()
        return strings, broadcast_shape

    @torch.no_grad()
    def decompress(self, strings, broadcast_shape):
        indexes = self.build_indexes(broadcast_shape)

        indexes = indexes.reshape(-1).tolist()
        symbols = []
        strings = np.array(strings)
        batch_shape = strings.shape
        strings = strings.reshape(-1)
        for s in strings:
            symbols.append(
                self.range_decoder.decode_with_indexes(
                    s.item(), indexes,
                    self.prior.cached_cdf_table_list,
                    self.prior.cached_cdf_length_list,
                    self.prior.cached_cdf_offset_list
                )
            )
        symbols = torch.tensor(symbols)
        symbols = self.dequantize(symbols)
        symbols = symbols.reshape(batch_shape + broadcast_shape + self.prior.batch_shape)
        return symbols


class NoisyDeepFactorizedEntropyModel(ContinuousBatchedEntropyModel):
    def __init__(self,
                 batch_shape: torch.Size,
                 coding_ndim: int,
                 num_filters=(1, 3, 3, 3, 3, 1),
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16):

        prior_weights, prior_biases, prior_factors = \
            DeepFactorized.make_parameters(
                batch_numel=batch_shape.numel(),
                init_scale=init_scale,
                num_filters=num_filters)

        super(ContinuousBatchedEntropyModel, self).__init__(
            prior=NoisyDeepFactorized(
                batch_shape=batch_shape,
                weights=prior_weights,
                biases=prior_biases,
                factors=prior_factors),
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision,
        )

        # Keep references to ParameterList objects here to make them a part of state dict.
        self.prior_weights, self.prior_biases, self.prior_factors = \
            prior_weights, prior_biases, prior_factors
