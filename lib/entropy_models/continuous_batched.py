import io
from typing import List, Tuple, Dict, Union
import math

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
                 bottleneck_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 quantize_bottleneck_in_eval: bool = True,
                 init_scale: float = 10,
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
            bottleneck_process=bottleneck_process,
            bottleneck_scaler=bottleneck_scaler,
            init_scale=init_scale,
            tail_mass=tail_mass,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            range_coder_precision=range_coder_precision,
            overflow_coding=overflow_coding
        )
        assert coding_ndim >= self.prior.batch_ndim
        self.bottleneck_scaler = bottleneck_scaler
        self.quantize_bottleneck_in_eval = quantize_bottleneck_in_eval
        self.broadcast_shape_bytes = broadcast_shape_bytes

    def build_indexes(self, broadcast_shape: torch.Size):
        indexes = torch.arange(self.prior.batch_numel, dtype=torch.int32)
        indexes = indexes.reshape(self.prior.batch_shape)
        indexes = indexes.repeat(*broadcast_shape,
                                 *[1] * len(self.prior.batch_shape))
        return indexes

    @minkowski_tensor_wrapped_fn({1: 0})
    def forward(self, x: torch.Tensor,
                is_first_forward: bool = True) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, List[bytes], torch.Size]]:
        """
        x: `torch.Tensor` containing the data to be compressed. Must have at
        least `self.coding_ndim` dimensions, and the innermost dimensions must
        be broadcastable to `self.prior_shape`.
        """
        if self.bottleneck_scaler != 1:
            x = x * self.bottleneck_scaler
        if self.training:
            processed_x = self.process(x)
            if self.bottleneck_scaler != 1:
                processed_x = processed_x / self.bottleneck_scaler
            log_probs = self.prior.log_prob(processed_x)
            loss_dict = {'bits_loss': log_probs.sum() / (-math.log(2))}
            if is_first_forward:
                aux_loss = self.prior.aux_loss()
                if aux_loss is not None:
                    loss_dict['aux_loss'] = aux_loss
            return processed_x, loss_dict

        else:
            bytes_list, batch_shape, _ = self.compress(x)
            decompressed = self.decompress(bytes_list, batch_shape, x.device)

            return decompressed, bytes_list, batch_shape

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({1: 2})
    def compress(self, x: torch.Tensor, estimate_bits: bool = False) \
            -> Union[Tuple[List[bytes], torch.Size, torch.Tensor],
                     Tuple[List[bytes], torch.Size, torch.Tensor, torch.Tensor]]:
        """
        x.shape = batch_shape + coding_unit_shape
                = batch_shape + broadcast_shape + prior_batch_shape
                (prior_event_shape = torch.Size([]))
        """
        if self.bottleneck_scaler != 1:
            x = x * self.bottleneck_scaler
        input_shape = x.shape
        batch_shape = input_shape[:-self.coding_ndim]
        coding_unit_shape = input_shape[-self.coding_ndim:]
        broadcast_shape = coding_unit_shape[:-len(self.prior.batch_shape)]

        if self.quantize_bottleneck_in_eval is True:
            quantized_x, dequantized_x = self.quantize(x)
        else:
            dequantized_x = x
            quantized_x = x.to(torch.int32)
        # collapse batch dimensions and coding_unit dimensions
        collapsed_x = quantized_x.reshape(-1, coding_unit_shape.numel())
        indexes = self.build_indexes(broadcast_shape).reshape(-1).tolist()  # shape: coding_unit_shape.numel()

        bytes_list = []
        for unit_idx in range(collapsed_x.shape[0]):
            bytes_list.append(
                self.prior.range_coder.encode_with_indexes(
                    [collapsed_x[unit_idx].tolist()],
                    [indexes]
                )[0]
            )

        # Log broadcast shape.
        assert len(self.broadcast_shape_bytes) == len(broadcast_shape)
        if sum(self.broadcast_shape_bytes) != 0:
            broadcast_shape_encoded = b''.join((
                length.to_bytes(bytes_num, 'little', signed=False)
                for bytes_num, length in zip(self.broadcast_shape_bytes, broadcast_shape)
            ))
            # All the samples in a batch share the same broadcast_shape.
            bytes_list = [broadcast_shape_encoded + s for s in bytes_list]

        if self.bottleneck_scaler != 1:
            dequantized_x = dequantized_x / self.bottleneck_scaler
        if estimate_bits is True:
            estimated_bits = self.prior.log_prob(
                quantized_x / self.bottleneck_scaler
            ).sum() / (-math.log(2))
            return bytes_list, batch_shape, dequantized_x, estimated_bits
        else:
            return bytes_list, batch_shape, dequantized_x

    @torch.no_grad()
    @minkowski_tensor_wrapped_fn({'<del>sparse_tensor_coords_tuple': 0})
    def decompress(self, bytes_list: List[bytes],
                   batch_shape: torch.Size,
                   target_device: torch.device,
                   broadcast_shape: Tuple[int] = None):
        if sum(self.broadcast_shape_bytes) != 0:
            broadcast_shape = []
            broadcast_shape_total_bytes = sum(self.broadcast_shape_bytes)
            broadcast_shape_bytes = bytes_list[0][:broadcast_shape_total_bytes]
            with io.BytesIO(broadcast_shape_bytes) as bs:
                for bytes_num in self.broadcast_shape_bytes:
                    broadcast_shape.append(int.from_bytes(bs.read(bytes_num), 'little', signed=False))
            broadcast_shape = torch.Size(broadcast_shape)
        else:
            broadcast_shape = broadcast_shape or torch.Size([1] * len(self.broadcast_shape_bytes))
            broadcast_shape_total_bytes = 0
            broadcast_shape_bytes = b''

        indexes = self.build_indexes(broadcast_shape).reshape(-1).tolist()

        symbols = []
        for s in bytes_list:
            assert s[:broadcast_shape_total_bytes] == broadcast_shape_bytes
            symbols.append(
                self.prior.range_coder.decode_with_indexes(
                    [s[broadcast_shape_total_bytes:]], [indexes],
                )[0]
            )
        symbols = torch.tensor(symbols, device=target_device)
        if self.quantize_bottleneck_in_eval is True:
            symbols = self.dequantize(symbols)
        else:
            symbols = symbols.to(torch.float)
        symbols = symbols.reshape(batch_shape + broadcast_shape + self.prior.batch_shape)
        if self.bottleneck_scaler != 1:
            symbols /= self.bottleneck_scaler
        return symbols


class NoisyDeepFactorizedEntropyModel(ContinuousBatchedEntropyModel):
    def __init__(self,
                 batch_shape: torch.Size,
                 coding_ndim: int,
                 num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 bottleneck_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 quantize_bottleneck_in_eval: bool = True,
                 init_scale: float = 10,
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
                factors=prior_factors,
                noise_width=1 / bottleneck_scaler),
            coding_ndim=coding_ndim,
            bottleneck_process=bottleneck_process,
            bottleneck_scaler=bottleneck_scaler,
            quantize_bottleneck_in_eval=quantize_bottleneck_in_eval,
            init_scale=init_scale,
            tail_mass=tail_mass,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            range_coder_precision=range_coder_precision,
            overflow_coding=overflow_coding,
            broadcast_shape_bytes=broadcast_shape_bytes
        )
        self.prior_num_filter = num_filters
        # Keep references to ParameterList objects here to make them a part of state dict.
        self.prior_weights, self.prior_biases, self.prior_factors = \
            prior_weights, prior_biases, prior_factors

    def __repr__(self):
        return f'NoisyDeepFactorizedEntropyModel with\n' \
               f'   batch_shape={self.prior.batch_shape}\n' \
               f'   coding_ndim={self.coding_ndim}\n' \
               f'   num_filter={self.prior_num_filter}\n'
