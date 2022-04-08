import io
from typing import List, Tuple, Dict, Union, Sequence
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
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 lower_bound: Union[int, torch.Tensor] = 0,
                 upper_bound: Union[int, torch.Tensor] = -1,
                 range_coder_precision: int = 16,
                 overflow_coding: bool = True,
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
            init_scale=init_scale,
            tail_mass=tail_mass,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            range_coder_precision=range_coder_precision,
            overflow_coding=overflow_coding
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
    def forward(self, x: torch.Tensor,
                return_aux_loss: bool = True,
                additive_uniform_noise: bool = True) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, List[bytes], torch.Size]]:
        """
        x: `torch.Tensor` containing the data to be compressed. Must have at
        least `self.coding_ndim` dimensions, and the innermost dimensions must
        be broadcastable to `self.prior_shape`.
        """
        if self.training:
            if additive_uniform_noise is True:
                x_perturbed = self.perturb(x)
            else:
                x_perturbed = x
            log_probs = self.prior.log_prob(x_perturbed)
            loss_dict = {'bits_loss': log_probs.sum() / (-math.log(2))}
            if return_aux_loss:
                aux_loss = self.prior.aux_loss()
                if aux_loss is not None:
                    loss_dict['aux_loss'] = aux_loss
            return x_perturbed, loss_dict

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
        # collapse batch dimensions and coding_unit dimensions
        collapsed_x = quantized_x.reshape(-1, coding_unit_shape.numel())
        indexes = self.build_indexes(broadcast_shape).reshape(-1).tolist()  # shape: coding_unit_shape.numel()

        strings = []
        for unit_idx in range(collapsed_x.shape[0]):
            strings.append(
                self.prior.range_coder.encode_with_indexes(
                    [collapsed_x[unit_idx].tolist()],
                    [indexes]
                )[0]
            )

        # Log broadcast shape.
        assert len(self.broadcast_shape_bytes) == len(broadcast_shape)
        if sum(self.broadcast_shape_bytes) != 0:
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
        if sum(self.broadcast_shape_bytes) != 0:
            broadcast_shape = []
            broadcast_shape_total_bytes = sum(self.broadcast_shape_bytes)
            broadcast_shape_string = strings[0][:broadcast_shape_total_bytes]
            with io.BytesIO(broadcast_shape_string) as bs:
                for bytes_num in self.broadcast_shape_bytes:
                    broadcast_shape.append(int.from_bytes(bs.read(bytes_num), 'little', signed=False))
            broadcast_shape = torch.Size(broadcast_shape)
        else:
            broadcast_shape = torch.Size([1] * len(self.broadcast_shape_bytes))
            broadcast_shape_total_bytes = 0
            broadcast_shape_string = b''

        indexes = self.build_indexes(broadcast_shape).reshape(-1).tolist()

        symbols = []
        for s in strings:
            assert s[:broadcast_shape_total_bytes] == broadcast_shape_string
            symbols.append(
                self.prior.range_coder.decode_with_indexes(
                    [s[broadcast_shape_total_bytes:]], [indexes],
                )[0]
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
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 lower_bound: Union[int, torch.Tensor] = 0,
                 upper_bound: Union[int, torch.Tensor] = -1,
                 range_coder_precision: int = 16,
                 overflow_coding: bool = True,
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
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            range_coder_precision=range_coder_precision,
            overflow_coding=overflow_coding,
            broadcast_shape_bytes=broadcast_shape_bytes
        )
        # Keep references to ParameterList objects here to make them a part of state dict.
        self.prior_weights, self.prior_biases, self.prior_factors = \
            prior_weights, prior_biases, prior_factors
