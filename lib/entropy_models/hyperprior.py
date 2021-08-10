from typing import List, Union, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.distributions import Distribution

from .continuous_batched import NoisyDeepFactorizedEntropyModel
from .continuous_indexed import ContinuousIndexedEntropyModel


class NoisyDeepFactorizedHyperPriorEntropyModel(nn.Module):
    def __init__(self,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module,

                 hyperprior_batch_shape: torch.Size,
                 coding_ndim: int,

                 prior_fn: Callable[[Any], Distribution],
                 index_ranges: List[int],
                 parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],

                 hyperprior_num_filters=(1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,

                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16,
                 ):
        super(NoisyDeepFactorizedHyperPriorEntropyModel, self).__init__()

        self.hyper_encoder = hyper_encoder
        self.hyper_decoder = hyper_decoder

        self.hyperprior_entropy_model = NoisyDeepFactorizedEntropyModel(
            prior_batch_shape=hyperprior_batch_shape,
            coding_ndim=coding_ndim,
            num_filters=hyperprior_num_filters,
            init_scale=hyperprior_init_scale,
            tail_mass=hyperprior_tail_mass,
            range_coder_precision=range_coder_precision
        )

        self.prior_entropy_model = ContinuousIndexedEntropyModel(
            prior_fn=prior_fn,
            index_ranges=index_ranges,
            parameter_fns=parameter_fns,
            coding_ndim=coding_ndim,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def forward(self, y):
        z = self.hyper_encoder(y)
        z_tilde, hyperprior_loss_dict, *hyperprior_strings = self.hyperprior_entropy_model(z)
        indexes = self.hyper_decoder(z_tilde)
        y_tilde, prior_loss_dict, *strings = self.prior_entropy_model(y, indexes)

        loss_dict = prior_loss_dict
        loss_dict['hyper_bits_loss'] = hyperprior_loss_dict['bits_loss']

        if self.training:
            loss_dict['hyper_aux_loss'] = hyperprior_loss_dict['aux_loss']
            assert hyperprior_strings == strings == []
        else:
            strings = strings[0] + hyperprior_strings[0]  # TODO: split

        if self.training:
            return y_tilde, loss_dict
        else:
            return y_tilde, loss_dict, strings
