import math
from typing import Union

import torch
from torch import nn as nn
try:
    import MinkowskiEngine as ME
except ImportError: ME = None

from .factorized_prior import FactorizedPrior
from .gaussian_conditional import GaussianConditional


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class FactorizedPriorGaussianConditional(nn.Module):
    def __init__(self, channels: int, chnl_dim: int, hyper_encoder: nn.Module, hyper_decoder: nn.Module):
        super(FactorizedPriorGaussianConditional, self).__init__()
        self.factorized_prior = FactorizedPrior(channels=channels, chnl_dim=chnl_dim)
        self.gaussian_conditional = GaussianConditional()
        self.hyper_encoder = hyper_encoder
        self.hyper_decoder = hyper_decoder

    def forward(self, y: Union[torch.Tensor, ME.SparseTensor]):
        if self.training:
            z = self.hyper_encoder(y)
            z_hat, hyper_loss_items = self.factorized_prior(z)
            gaussian_params = self.hyper_decoder(z_hat)
            if isinstance(y, torch.Tensor):
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
            elif isinstance(y, ME.SparseTensor):
                scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
                scales_hat, means_hat = scales_hat[None], means_hat[None]
            else: raise NotImplementedError
            y_hat, loss_items = self.gaussian_conditional(y, scales_hat, means=means_hat)

            hyper_loss_items['hyper_bits_loss'] = hyper_loss_items['bits_loss']
            del hyper_loss_items['bits_loss']
            loss_items.update(hyper_loss_items)
            return y_hat, loss_items

        else:
            self.update()
            z = self.hyper_encoder(y)
            z_hat, z_strings = self.factorized_prior(z)
            gaussian_params = self.hyper_decoder(z_hat)
            if isinstance(y, torch.Tensor):
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
            elif isinstance(y, ME.SparseTensor):
                scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
                scales_hat, means_hat = scales_hat[None], means_hat[None]
            else: raise NotImplementedError
            y_hat, y_strings = self.gaussian_conditional(y, scales_hat, means=means_hat)

            return y_hat, tuple(s1 + b' ' + s2 for s1, s2 in zip(y_strings, z_strings))

    def update(self, scale_table=None, force=True):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.factorized_prior.update(force=force)
        return updated
