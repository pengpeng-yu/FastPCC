from typing import List, Tuple, Union, Dict, Any, Callable
import math

import torch
import torch.nn as nn
from torch.distributions import Distribution

from .continuous_batched import NoisyDeepFactorizedEntropyModel
from .continuous_indexed import ContinuousIndexedEntropyModel
from .distributions.uniform_noise import NoisyNormal, NoisyDeepFactorized

from lib.torch_utils import minkowski_tensor_wrapped


class NoisyDeepFactorizedHyperPriorEntropyModel(nn.Module):
    def __init__(self,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module,

                 hyperprior_batch_shape: torch.Size,
                 coding_ndim: int,

                 prior_fn: Callable[..., Distribution],
                 index_ranges: Tuple[int, ...],
                 parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],

                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,

                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16,
                 ):
        super(NoisyDeepFactorizedHyperPriorEntropyModel, self).__init__()

        self.hyper_encoder = hyper_encoder
        self.hyper_decoder = hyper_decoder

        self.hyperprior_entropy_model = NoisyDeepFactorizedEntropyModel(
            batch_shape=hyperprior_batch_shape,
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
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def prior_entropy_model_forward(self, y, indexes):
        return self.prior_entropy_model(y, indexes)

    def forward(self, y):
        z = self.hyper_encoder(y)
        z_tilde, hyperprior_loss_dict, *hyperprior_strings = self.hyperprior_entropy_model(z)
        indexes = self.hyper_decoder(z_tilde)
        y_tilde, prior_loss_dict, *strings = self.prior_entropy_model_forward(y, indexes)

        loss_dict = prior_loss_dict
        loss_dict['hyper_bits_loss'] = hyperprior_loss_dict['bits_loss']

        if self.training:
            loss_dict['hyper_aux_loss'] = hyperprior_loss_dict['aux_loss']
            assert hyperprior_strings == strings == []
        else:
            # TODO: split
            strings = [i + j for i, j in zip(hyperprior_strings[0], strings[0])]

        if self.training:
            return y_tilde, loss_dict
        else:
            return y_tilde, loss_dict, strings


class NoisyDeepFactorizedHyperPriorScaleNoisyNormalEntropyModel(NoisyDeepFactorizedHyperPriorEntropyModel):
    def __init__(self,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module,

                 hyperprior_batch_shape: torch.Size,
                 coding_ndim: int,

                 num_scales: int = 64,
                 scale_min: float = 0.11,
                 scale_max: float = 256,

                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,

                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16,
                 ):
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min)) / (num_scales - 1)

        super(NoisyDeepFactorizedHyperPriorScaleNoisyNormalEntropyModel, self).__init__(
            hyper_encoder=hyper_encoder,
            hyper_decoder=hyper_decoder,
            hyperprior_batch_shape=hyperprior_batch_shape,
            coding_ndim=coding_ndim,
            prior_fn=NoisyNormal,
            index_ranges=(num_scales,),
            parameter_fns={'loc': lambda _: 0,
                           'scale': lambda i: torch.exp(offset + factor * i)},
            hyperprior_num_filters=hyperprior_num_filters,
            hyperprior_init_scale=hyperprior_init_scale,
            hyperprior_tail_mass=hyperprior_tail_mass,
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    @minkowski_tensor_wrapped(extra_preparation={1: abs})
    def forward(self, y):
        return super(NoisyDeepFactorizedHyperPriorScaleNoisyNormalEntropyModel, self).forward(y)


class NoisyDeepFactorizedHyperPriorNoisyDeepFactorizedEntropyModel(NoisyDeepFactorizedHyperPriorEntropyModel):
    def __init__(self,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module,

                 hyperprior_batch_shape: torch.Size,
                 coding_ndim: int,

                 hyperprior_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),
                 hyperprior_init_scale: int = 10,
                 hyperprior_tail_mass: float = 2 ** -8,

                 index_ranges: Tuple[int, ...] = (4,) * 9,
                 parameter_fns_type: str = 'split',
                 parameter_fns_transform_fn: Callable = None,
                 num_filters: Tuple[int, ...] = (1, 2, 1),
                 quantize_indexes: bool = False,
                 init_scale: int = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16,
                 ):
        assert len(num_filters) >= 3 and num_filters[0] == num_filters[-1] == 1
        assert parameter_fns_type in ('split', 'transform')
        if not len(index_ranges) > 1:
            raise NotImplementedError
        index_channels = len(index_ranges)

        weights_param_numel = [num_filters[i] * num_filters[i+1] for i in range(len(num_filters) - 1)]
        biases_param_numel = num_filters[1:]
        factors_param_numel = num_filters[1:-1]

        if parameter_fns_type == 'split':

            weights_param_cum_numel = torch.cumsum(torch.tensor([0, *weights_param_numel]), dim=0)
            biases_param_cum_numel = \
                torch.cumsum(torch.tensor([0, *biases_param_numel]), dim=0) + weights_param_cum_numel[-1]
            factors_param_cum_numel = \
                torch.cumsum(torch.tensor([0, *factors_param_numel]), dim=0) + biases_param_cum_numel[-1]

            assert index_channels == factors_param_cum_numel[-1]

            parameter_fns = {
                'batch_shape': lambda i: i.shape[:-1],
                'weights': lambda i: [i[..., weights_param_cum_numel[_]: weights_param_cum_numel[_ + 1]].view
                                      (-1, num_filters[_ + 1], num_filters[_]) / 3 - 0.5
                                      for _ in range(len(weights_param_numel))],

                'biases': lambda i: [i[..., biases_param_cum_numel[_]: biases_param_cum_numel[_ + 1]].view
                                     (-1, biases_param_numel[_], 1) / 3 - 0.5
                                     for _ in range(len(biases_param_numel))],

                'factors': lambda i: [i[..., factors_param_cum_numel[_]: factors_param_cum_numel[_ + 1]].view
                                      (-1, factors_param_numel[_], 1) / 3 - 0.5
                                      for _ in range(len(factors_param_numel))],
            }

        elif parameter_fns_type == 'transform':

            prior_indexes_weights_mlp = nn.ModuleList(
                [parameter_fns_transform_fn(index_channels, out_channels)
                 for out_channels in weights_param_numel])
            prior_indexes_biases_mlp = nn.ModuleList(
                [parameter_fns_transform_fn(index_channels, out_channels)
                 for out_channels in biases_param_numel])
            prior_indexes_factors_mlp = nn.ModuleList(
                [parameter_fns_transform_fn(index_channels, out_channels)
                 for out_channels in factors_param_numel])

            parameter_fns = {
                'batch_shape': lambda i: i.shape[:-1],
                'weights': lambda i: [transform(i).view(-1, num_filters[_ + 1], num_filters[_])
                                      for _, transform in enumerate(prior_indexes_weights_mlp)],

                'biases': lambda i: [transform(i).view(-1, biases_param_numel[_], 1)
                                     for _, transform in enumerate(prior_indexes_biases_mlp)],

                'factors': lambda i: [transform(i).view(-1, factors_param_numel[_], 1)
                                      for _, transform in enumerate(prior_indexes_factors_mlp)],
            }

        else: raise NotImplementedError

        self.indexes_view_fn = lambda x: x.view(
            *x.shape[:-1],
            x.shape[-1] // index_channels,
            index_channels)

        super(NoisyDeepFactorizedHyperPriorNoisyDeepFactorizedEntropyModel, self).__init__(
            hyper_encoder=hyper_encoder,
            hyper_decoder=hyper_decoder,
            hyperprior_batch_shape=hyperprior_batch_shape,
            coding_ndim=coding_ndim,
            prior_fn=NoisyDeepFactorized,
            index_ranges=index_ranges,
            parameter_fns=parameter_fns,
            hyperprior_num_filters=hyperprior_num_filters,
            hyperprior_init_scale=hyperprior_init_scale,
            hyperprior_tail_mass=hyperprior_tail_mass,
            quantize_indexes=quantize_indexes,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

        if parameter_fns_type == 'transform':
            # noinspection PyUnboundLocalVariable
            self.prior_indexes_weights_mlp, self.prior_indexes_biases_mlp, self.prior_indexes_factors_mlp = \
                prior_indexes_weights_mlp, prior_indexes_biases_mlp, prior_indexes_factors_mlp

    def prior_entropy_model_forward(self, y, indexes):
        return self.prior_entropy_model(y, indexes, extra_preparation={2: self.indexes_view_fn})
