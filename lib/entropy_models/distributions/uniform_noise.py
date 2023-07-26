from typing import List, Union

import torch
import torch.nn as nn
import torch.distributions
from torch.distributions import Distribution

from .deep_factorized import DeepFactorized
from . import special_math


def _logsum_expbig_minus_expsmall(big, small):
    """Numerically stable evaluation of `log(exp(big) - exp(small))`.

    This assumes `small <= big` and arguments that can be broadcast against each
    other.
    """
    return torch.log1p(-torch.exp(small - big)) + big


class UniformNoiseAdapter(Distribution):
    arg_constraints = {}

    def __init__(self, base: Distribution, noise_width: float = 1):
        super(UniformNoiseAdapter, self).__init__(base.batch_shape)
        self.base = base
        self.half_width = noise_width / 2
        assert hasattr(self.base, 'log_cdf') or hasattr(self.base, 'cdf')

    def log_prob(self, y):
        try:
            return self._log_prob_with_logsf_and_logcdf(y)
        except (NotImplementedError, AttributeError):
            return self._log_prob_with_logcdf(y)

    def _log_prob_with_logcdf(self, y):
        return _logsum_expbig_minus_expsmall(
            self.base.log_cdf(y + self.half_width), self.base.log_cdf(y - self.half_width))

    # noinspection PyTypeChecker
    def _log_prob_with_logsf_and_logcdf(self, y):
        """Compute log_prob(y) using log survival_function and cdf together."""
        # There are two options that would be equal if we had infinite precision:
        # Log[ sf(y - .5) - sf(y + .5) ]
        #   = Log[ exp{logsf(y - .5)} - exp{logsf(y + .5)} ]
        # Log[ cdf(y + .5) - cdf(y - .5) ]
        #   = Log[ exp{logcdf(y + .5)} - exp{logcdf(y - .5)} ]
        logsf_y_plus = self.base.log_survival_function(y + self.half_width)
        logsf_y_minus = self.base.log_survival_function(y - self.half_width)
        logcdf_y_plus = self.base.log_cdf(y + self.half_width)
        logcdf_y_minus = self.base.log_cdf(y - self.half_width)

        # Important:  Here we use select in a way such that no input is inf, this
        # prevents the troublesome case where the output of select can be finite,
        # but the output of grad(select) will be NaN.

        # In either case, we are doing Log[ exp{big} - exp{small} ]
        # We want to use the sf items precisely when we are on the right side of the
        # median, which occurs when logsf_y < logcdf_y.
        condition = logsf_y_plus < logcdf_y_plus
        big = torch.where(condition, logsf_y_minus, logcdf_y_plus)
        small = torch.where(condition, logsf_y_plus, logcdf_y_minus)
        return _logsum_expbig_minus_expsmall(big, small)

    def prob(self, value):
        try:
            return self._prob_with_sf_and_cdf(value)
        except (NotImplementedError, AttributeError):
            return self._prob_with_cdf(value)

    def _prob_with_cdf(self, y):
        return self.base.cdf(y + self.half_width) - self.base.cdf(y - self.half_width)

    def _prob_with_sf_and_cdf(self, y):
        # There are two options that would be equal if we had infinite precision:
        # sf(y - .5) - sf(y + .5)
        # cdf(y + .5) - cdf(y - .5)
        sf_y_plus = self.base.survival_function(y + self.half_width)
        sf_y_minus = self.base.survival_function(y - self.half_width)
        cdf_y_plus = self.base.cdf(y + self.half_width)
        cdf_y_minus = self.base.cdf(y - self.half_width)

        # sf_prob has greater precision if we're on the right side of the median.
        # noinspection PyTypeChecker
        return torch.where(sf_y_plus < cdf_y_plus,
                           sf_y_minus - sf_y_plus,
                           cdf_y_plus - cdf_y_minus)


class NoisyDeepFactorized(UniformNoiseAdapter):
    def __init__(self,
                 batch_shape: torch.Size,
                 weights: Union[List[torch.Tensor], nn.ParameterList] = None,
                 biases: Union[List[torch.Tensor], nn.ParameterList] = None,
                 factors: Union[List[torch.Tensor], nn.ParameterList] = None,
                 noise_width: float = 1):
        super(NoisyDeepFactorized, self).__init__(
            DeepFactorized(
                batch_shape=batch_shape,
                weights=weights,
                biases=biases,
                factors=factors),
            noise_width=noise_width
        )


class Normal(torch.distributions.Normal):
    def _z(self, x, scale=None):
        return (x - self.loc) / (self.scale if scale is None else scale)

    def log_cdf(self, x):
        return special_math.log_ndtr(self._z(x))

    def log_survival_function(self, x):
        return special_math.log_ndtr(-self._z(x))


class NoisyNormal(UniformNoiseAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(Normal(*args, **kwargs))


class NoisyMixtureSameFamily(torch.distributions.MixtureSameFamily):
    def __init__(self, mixture_distribution, components_distribution):
        super().__init__(
            mixture_distribution=mixture_distribution,
            component_distribution=UniformNoiseAdapter(components_distribution),
        )
        self.base = torch.distributions.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=components_distribution
        )
