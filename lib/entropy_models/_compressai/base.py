from typing import Tuple, List, Dict, Union

import torch
import torch.nn as nn

from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf


class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through_if = (x >= bound) | (grad_output < 0)
        return pass_through_if * grad_output, None


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.tensor([bound], dtype=torch.float))

    def forward(self, x):
        return LowerBoundFunction.apply(x, self.bound)


def pmf_to_quantized_cdf(pmf, precision=16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class EntropyModel(nn.Module):
    def __init__(self,
                 likelihood_bound: float = 1e-9,
                 entropy_coder: str = 'ans',
                 entropy_coder_precision: int = 16):
        super(EntropyModel, self).__init__()
        if entropy_coder == 'ans':
            from compressai import ans
            self.entropy_encoder = ans.RansEncoder()
            self.entropy_decoder = ans.RansDecoder()
        else:
            raise NotImplementedError

        if likelihood_bound > 0:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)
        else:
            self.likelihood_lower_bound = None
        self.entropy_coder_precision = entropy_coder_precision

    def forward(self, *args):
        raise NotImplementedError

    @staticmethod
    def check_vars(indexes, quantized_cdf, cdf_length, offset):
        assert quantized_cdf.ndim == 2
        assert cdf_length.ndim == offset.ndim == 1
        assert quantized_cdf.shape[0] == cdf_length.shape[0] == offset.shape[0]
        assert indexes.dtype == quantized_cdf.dtype == \
               cdf_length.dtype == offset.dtype == torch.int32

    def _cached_noise(self, shape: torch.Size, device):
        if not hasattr(self, "_noise"):
            setattr(self, "_noise", torch.empty(shape, dtype=torch.float, device=device))
        self._noise.resize_(shape)
        self._noise.uniform_(-0.5, 0.5)
        return self._noise

    def quantize(self, x: torch.Tensor, mode: str, means: torch.Tensor = None):
        if mode == 'noise':
            assert self.training
            noise = self._cached_noise(x.shape, x.device)
            return x + noise

        elif mode == 'symbols':
            assert not self.training
            x -= means
            x = torch.round(x).to(torch.int32)
            return x

        else: raise NotImplementedError

    @staticmethod
    def dequantize(x, means: torch.Tensor = None):
        if means is not None:
            x = x + means
        else:
            x = x.type(torch.float)
        return x

    def _pmf_to_cdf(self, pmf: torch.Tensor,
                    tail_mass: torch.Tensor,
                    pmf_length: torch.Tensor,
                    max_length: int):
        """
        :param pmf: (channels, max_length) float
        :param tail_mass: (channels, 1) float
        :param pmf_length: (channels, ) int32
        :param max_length: max length of pmf int32
        :return: quantized cdf (channels, max_length + 2)
        """
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.shape[0]] = _cdf
        return cdf

    def compress(self, x: torch.Tensor,
                 indexes: torch.Tensor,
                 quantized_cdf: torch.Tensor,
                 cdf_length: torch.Tensor,
                 offset: torch.Tensor,
                 means: torch.Tensor = None) -> List[bytes]:
        """
        x: tensor with shape (B, ...)
        indexes: CDF indexes of _quantized_cdf with the same shape as x
        """
        self.check_vars(indexes, quantized_cdf, cdf_length, offset)

        symbols = self.quantize(x, 'symbols', means)
        quantized_cdf = quantized_cdf.tolist()
        cdf_length = cdf_length.tolist()
        offset = offset.tolist()

        strings = []
        for i in range(x.shape[0]):
            rv = self.entropy_encoder.encode_with_indexes(
                symbols[i].reshape(-1).tolist(),
                indexes[i].reshape(-1).tolist(),
                quantized_cdf,
                cdf_length,
                offset)
            strings.append(rv)
        return strings

    def decompress(self, strings: List[bytes],
                   indexes: torch.Tensor,
                   quantized_cdf: torch.Tensor,
                   cdf_length: torch.Tensor,
                   offset: torch.Tensor,
                   means: torch.Tensor = None) -> torch.Tensor:
        self.check_vars(indexes, quantized_cdf, cdf_length, offset)

        outputs = quantized_cdf.new(indexes.size())

        quantized_cdf = quantized_cdf.tolist()
        cdf_length = cdf_length.tolist()
        offset = offset.tolist()

        for i, s in enumerate(strings):
            values = self.entropy_decoder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                quantized_cdf,
                cdf_length,
                offset,
            )
            outputs[i] = torch.tensor(values).view(outputs[i].size())
        outputs = self.dequantize(outputs, means)
        return outputs
