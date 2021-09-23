import io
from typing import List, Tuple, Dict, Union, Sequence, Optional
import math
from functools import reduce

import torch
import torch.distributions
from torch.distributions import Distribution

from .distributions.deep_factorized import DeepFactorized
from .distributions.uniform_noise import NoisyDeepFactorized
from .continuous_base import ContinuousEntropyModelBase

from lib.torch_utils import minkowski_tensor_wrapped_fn


class ContinuousBatchedEntropyModel(ContinuousEntropyModelBase):
    def __init__(self,
                 prior: Distribution,
                 coding_ndim: int,
                 additive_uniform_noise: bool = True,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16,
                 broadcast_shape_bytes: Tuple[int, ...] = (2,)):
        """
        Generally, prior object should have parameters on target device when being
        constructed, since functions of nn.Module like "to()" or "cuda()" has
        no effect on it.

        The innermost `self.coding_ndim` dimensions are treated as one coding unit,
        i.e. are compressed into one string each. Any additional dimensions to the
        left are treated as batch dimensions.

        The innermost dimensions of input tensor are supposed to be the same
        as self.prior_shape.
        """
        super(ContinuousBatchedEntropyModel, self).__init__(
            prior=prior,
            coding_ndim=coding_ndim,
            additive_uniform_noise=additive_uniform_noise,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )
        assert coding_ndim >= self.prior.batch_ndim
        self.broadcast_shape_bytes = broadcast_shape_bytes

    def build_indexes(self, broadcast_shape: torch.Size):
        indexes = torch.arange(self.prior.batch_numel, dtype=torch.int32)
        indexes = indexes.reshape(self.prior.batch_shape)
        indexes = indexes.repeat(*broadcast_shape,
                                 *[1] * len(self.prior.batch_shape))
        return indexes

    @minkowski_tensor_wrapped_fn({1: 0})
    def forward(self, x: torch.Tensor) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, List[bytes], torch.Size]]:
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
            strings, batch_shape, _ = self.compress(x)
            decompressed = self.decompress(strings, batch_shape, x.device)

            return decompressed, strings, batch_shape

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({1: 2})
    def compress(self, x: torch.Tensor,
                 skip_quantization: bool = False,
                 return_dequantized: bool = False,
                 estimate_bits: bool = False) \
            -> Union[Tuple[List[bytes], torch.Size, torch.Tensor],
                     Tuple[List[bytes], torch.Size, torch.Tensor, torch.Tensor]]:
        """
        x.shape = batch_shape + coding_unit_shape
                = batch_shape + broadcast_shape + prior_batch_shape
                (prior_event_shape = torch.Size([]))
        """
        input_shape = x.shape
        batch_shape = input_shape[:-self.coding_ndim]
        coding_unit_shape = input_shape[-self.coding_ndim:]
        broadcast_shape = coding_unit_shape[:-len(self.prior.batch_shape)]

        if skip_quantization:
            if return_dequantized:
                rounded_x_or_dequantized_x = self.dequantize(x)
            else:
                rounded_x_or_dequantized_x = x
            quantized_x = x.to(torch.int32)
        else:
            quantized_x, rounded_x_or_dequantized_x = self.quantize(
                x, return_dequantized=return_dequantized
            )
        collapsed_x = quantized_x.reshape(-1, *coding_unit_shape)  # collapse batch dimensions
        indexes = self.build_indexes(broadcast_shape)  # shape: coding_unit_shape

        strings = []
        indexes = indexes.reshape(-1).tolist()
        for unit_idx in range(collapsed_x.shape[0]):
            strings.append(
                self.range_encoder.encode_with_indexes(
                    collapsed_x[unit_idx].reshape(-1).tolist(),
                    indexes,
                    self.prior.cached_cdf_table_list,
                    self.prior.cached_cdf_length_list,
                    self.prior.cached_cdf_offset_list
                )
            )

        # Log broadcast shape.
        assert len(self.broadcast_shape_bytes) == len(broadcast_shape)
        broadcast_shape_encoded = reduce(
            lambda i, j: i + j,
            (length.to_bytes(bytes_num, 'little', signed=False)
             for bytes_num, length in zip(self.broadcast_shape_bytes, broadcast_shape))
        )
        # All the samples in a batch share the same broadcast_shape.
        strings = [broadcast_shape_encoded + s for s in strings]

        if estimate_bits is True:
            estimated_bits = self.prior.log_prob(quantized_x).sum() / (-math.log(2))
            return strings, batch_shape, rounded_x_or_dequantized_x, estimated_bits

        else:
            return strings, batch_shape, rounded_x_or_dequantized_x

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({'<del>sparse_tensor_coords_tuple': 0})
    def decompress(self, strings: List[bytes],
                   batch_shape: torch.Size,
                   target_device: torch.device,
                   skip_dequantization: bool = False):
        broadcast_shape = []
        broadcast_shape_total_bytes = sum(self.broadcast_shape_bytes)
        broadcast_shape_string = strings[0][:broadcast_shape_total_bytes]
        with io.BytesIO(broadcast_shape_string) as bs:
            for bytes_num in self.broadcast_shape_bytes:
                broadcast_shape.append(int.from_bytes(bs.read(bytes_num), 'little', signed=False))

        broadcast_shape = torch.Size(broadcast_shape)

        indexes = self.build_indexes(broadcast_shape)
        indexes = indexes.reshape(-1).tolist()

        symbols = []
        for s in strings:
            assert s[:broadcast_shape_total_bytes] == broadcast_shape_string
            symbols.append(
                self.range_decoder.decode_with_indexes(
                    s[broadcast_shape_total_bytes:], indexes,
                    self.prior.cached_cdf_table_list,
                    self.prior.cached_cdf_length_list,
                    self.prior.cached_cdf_offset_list
                )
            )
        symbols = torch.tensor(symbols, device=target_device)
        if skip_dequantization:
            symbols = symbols.to(torch.float)
        else:
            symbols = self.dequantize(symbols)

        symbols = symbols.reshape(batch_shape + broadcast_shape + self.prior.batch_shape)
        return symbols


class NoisyDeepFactorizedEntropyModel(ContinuousBatchedEntropyModel):
    def __init__(self,
                 batch_shape: torch.Size,
                 coding_ndim: int,
                 num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 additive_uniform_noise: bool = True,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16,
                 broadcast_shape_bytes: Tuple[int, ...] = (2,)):

        prior_weights, prior_biases, prior_factors = \
            DeepFactorized.make_parameters(
                batch_numel=batch_shape.numel(),
                init_scale=init_scale,
                num_filters=num_filters)

        super(NoisyDeepFactorizedEntropyModel, self).__init__(
            prior=NoisyDeepFactorized(
                batch_shape=batch_shape,
                weights=prior_weights,
                biases=prior_biases,
                factors=prior_factors),
            additive_uniform_noise=additive_uniform_noise,
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision,
            broadcast_shape_bytes=broadcast_shape_bytes
        )

        # Keep references to ParameterList objects here to make them a part of state dict.
        self.prior_weights, self.prior_biases, self.prior_factors = \
            prior_weights, prior_biases, prior_factors
