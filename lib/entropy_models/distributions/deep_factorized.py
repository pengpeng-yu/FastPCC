from typing import List, Union, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
from torch.distributions import Distribution


class DeepFactorized(Distribution):
    def __init__(self,
                 batch_shape: torch.Size,
                 weights: Union[List[torch.Tensor], nn.ParameterList] = None,
                 biases: Union[List[torch.Tensor], nn.ParameterList] = None,
                 factors: Union[List[torch.Tensor], nn.ParameterList] = None):
        super(DeepFactorized, self).__init__(batch_shape)
        self.weights = weights
        self.biases = biases
        self.factors = factors

    def logits_cdf(self, value: torch.Tensor, stop_gradient=False):
        value_shape = value.shape
        value = value.view(-1, 1, self.batch_shape.numel()).permute(2, 1, 0).contiguous()
        n_iter = len(self.weights)
        assert n_iter == len(self.biases) == len(self.factors) + 1
        for i in range(n_iter):
            weight = self.weights[i]
            if stop_gradient: weight = weight.detach()
            value = torch.matmul(F.softplus(weight), value)

            bias = self.biases[i]
            if stop_gradient: bias = bias.detach()
            value += bias

            if i < n_iter - 1:
                factor = self.factors[i]
                if stop_gradient: factor = factor.detach()
                value += torch.tanh(factor) * torch.tanh(value)
        return value.permute(2, 1, 0).contiguous().view(value_shape)

    def cdf(self, value):
        return torch.sigmoid(self.logits_cdf(value))

    def log_cdf(self, value):
        return F.logsigmoid(self.logits_cdf(value))

    def survival_function(self, value):
        return torch.sigmoid(-self.logits_cdf(value))

    def log_survival_function(self, value):
        return F.logsigmoid(-self.logits_cdf(value))

    @staticmethod
    def make_parameters(batch_numel: int, init_scale=10, num_filters=(1, 3, 3, 3, 3, 1)) \
            -> Tuple[nn.ParameterList, nn.ParameterList, nn.ParameterList]:
        assert num_filters[0] == 1 and num_filters[-1] == 1
        scale = init_scale ** (1 / (len(num_filters) + 1))
        weights = nn.ParameterList()
        biases = nn.ParameterList()
        factors = nn.ParameterList()
        for i in range(len(num_filters) - 1):
            init = np.log(np.expm1(1 / scale / num_filters[i + 1]))
            weights.append(nn.Parameter(
                torch.full((batch_numel, num_filters[i + 1], num_filters[i]),
                           fill_value=init)))

            biases.append(nn.Parameter(
                torch.empty((batch_numel, num_filters[i + 1], 1))))
            nn.init.uniform_(biases[-1], -0.5, 0.5)

            if i < len(num_filters) - 2:
                factors.append(nn.Parameter(
                    torch.zeros((batch_numel, num_filters[i + 1], 1))))

        # assert len(weights) == len(biases) == len(factors) + 1 == len(num_filters) - 1
        return weights, biases, factors
