from typing import Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.entropy_models.compressai.base import EntropyModel
from lib.torch_utils import minkowski_tensor_wrapped


class FactorizedPrior(EntropyModel):
    def __init__(self, channels: int, chnl_dim: int,
                 tail_mass: float = 1e-9,
                 init_scale: float = 10,
                 filters: Tuple[int, ...] = (3, 3, 3, 3),
                 likelihood_bound: float = 1e-9,
                 entropy_coder: str = 'ans',
                 entropy_coder_precision: int = 16):
        super(FactorizedPrior, self).__init__(
            likelihood_bound=likelihood_bound,
            entropy_coder=entropy_coder,
            entropy_coder_precision=entropy_coder_precision)

        self.chnl_dim = chnl_dim
        self.channels = channels
        assert 0 < tail_mass < 1
        self.tail_mass = tail_mass
        self.init_scale = init_scale
        self.filters = filters

        filters = (1, ) + self.filters + (1, )
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        for i in range(len(self.filters) + 1):
            matrix = torch.full((self.channels, filters[i + 1], filters[i]),
                                fill_value=np.log(np.expm1(1 / scale / filters[i + 1])),
                                dtype=torch.float)
            self.register_parameter(f'_matrix{i}', nn.Parameter(matrix))

            bias = torch.empty((self.channels, filters[i + 1], 1),
                               dtype=torch.float)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f'_bias{i}', nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.zeros((self.channels, filters[i + 1], 1),
                                     dtype=torch.float)
                self.register_parameter(f'_factor{i}', nn.Parameter(factor))

        self.quantiles = nn.Parameter(
            torch.tensor(
                [-self.init_scale, 0, self.init_scale],
                dtype=torch.float
            ).repeat(self.channels, 1, 1))

        target = np.log(self.tail_mass / 2 / (1 - self.tail_mass / 2))
        self.register_buffer('target',
                             torch.tensor([target, 0, -target], dtype=torch.float))

        self.register_buffer('_offset', torch.tensor([], dtype=torch.int32))
        self.register_buffer('_cdf_length', torch.tensor([], dtype=torch.int32))
        self.register_buffer('_quantized_cdf', torch.tensor([], dtype=torch.int32))

    @torch.no_grad()
    def update(self, force=False) -> bool:
        assert not self.training
        if self._quantized_cdf.numel() > 0 and not force:
            return False

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        samples = torch.arange(max_length, device=pmf_start.device)

        samples = samples[None, :] + pmf_start[:, None, None]

        lower = self._logits_cumulative(samples - 0.5, stop_gradient=True)
        upper = self._logits_cumulative(samples + 0.5, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        # tail_mass ????
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def aux_loss(self):
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, x, stop_gradient):
        """
        :param x: tensor to evaluate cumulative logits with shape (channels, 1, ?)
        :param stop_gradient: stop gradients of _matrix, _bias and _factor
        :return: logits with the same shape as x
        """
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            x = torch.matmul(F.softplus(matrix), x)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            x += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                x += torch.tanh(factor) * torch.tanh(x)
        return x

    def _likelihood(self, x):
        """
        :param x: tensor to evaluate likelihood with shape (channels, 1, ?)
        :return: likelihood with the same shape as x
        """
        lower = self._logits_cumulative(x - 0.5, stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5, stop_gradient=False)

        # likelihood = torch.abs(torch.sigmoid(upper) - torch.sigmoid(lower))
        with torch.no_grad():
            sign = -torch.sign(lower + upper)
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        return likelihood

    @minkowski_tensor_wrapped({1: 0})
    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], List[bytes]]]:
        """
        :param x: tensor with shape: ... x C x ...
        :return: x with noise and likelihood with the same shape with x
        """
        if self.training:
            permute_order = np.arange(len(x.shape))
            permute_order[0], permute_order[self.chnl_dim] = \
                permute_order[self.chnl_dim], permute_order[0]
            inv_permute_order = np.arange(len(x.shape))[np.argsort(permute_order)]

            x = x.permute(*permute_order).contiguous()
            shape_chnl_first = x.shape
            x = x.view(self.channels, 1, -1)

            x_tilde = self.quantize(x, 'noise', self.quantiles[:, :, 1:2])
            likelihood = self._likelihood(x_tilde)
            if self.likelihood_lower_bound is not None:
                likelihood = self.likelihood_lower_bound(likelihood)

            x_tilde = x_tilde.view(*shape_chnl_first)
            x_tilde = x_tilde.permute(*inv_permute_order).contiguous()

            likelihood = likelihood.view(*shape_chnl_first)
            likelihood = likelihood.permute(*inv_permute_order).contiguous()

            return x_tilde, {'bits_loss': torch.log2(likelihood).sum(),
                             'aux_loss': self.aux_loss()}

        else:
            self.update()
            compressed_strings = self.compress(x)
            return self.decompress(compressed_strings, x.shape), compressed_strings

    def _build_indexes(self, shape):
        """
        :param shape: torch.Size[..., self.channels, ...]
        :return: tensor: [0 ~ self.channels - 1].repeat(..., 1, ...)
        """
        assert shape[self.chnl_dim] == self.channels
        repeat_shape = list(shape)
        repeat_shape[self.chnl_dim] = 1
        indexes = torch.arange(self.channels, dtype=torch.int32).repeat(*repeat_shape)
        return indexes

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> List[bytes]:
        assert not self.training
        self.update()
        # x shape: (B, ...)
        indexes = self._build_indexes(x.shape)

        view_shape = [1] * len(x.shape)
        view_shape[self.chnl_dim] = -1
        medians = self.quantiles[:, 0, 1].view(*view_shape)
        medians = medians.expand(x.shape[0], *[-1] * (len(x.shape) - 1))

        return super().compress(x, indexes, medians)

    @torch.no_grad()
    def decompress(self, strings: List[bytes], shape: torch.Tensor) -> torch.Tensor:
        assert not self.training
        self.update()
        assert shape[0] == len(strings)
        indexes = self._build_indexes(shape).to(self._quantized_cdf.device)

        view_shape = [1] * len(shape)
        view_shape[self.chnl_dim] = -1
        medians = self.quantiles[:, 0, 1].view(*view_shape)
        medians = medians.expand(shape[0], *[-1] * (len(shape) - 1))

        return super().decompress(strings, indexes, medians)
