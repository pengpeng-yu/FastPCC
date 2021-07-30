from typing import Tuple, List, Union, Dict

import scipy.stats
import torch

try:
    import MinkowskiEngine as ME
except ImportError: ME = None

from lib.entropy_models.compressai.base import LowerBound, EntropyModel
from lib.torch_utils import minkowski_tensor_wrapped


class GaussianConditional(EntropyModel):
    def __init__(self,
                 scale_table: Tuple = None,
                 scale_bound: float = 0.11,
                 tail_mass: float = 1e-9,
                 likelihood_bound: float = 1e-9,
                 entropy_coder: str = 'ans',
                 entropy_coder_precision: int = 16):
        super(GaussianConditional, self).__init__(
            likelihood_bound=likelihood_bound,
            entropy_coder=entropy_coder,
            entropy_coder_precision=entropy_coder_precision)

        # used to precompute cdf of various scales which can be directly indexed when compressing
        if scale_table is not None:
            assert len(scale_table) >= 1
            assert scale_table == sorted(scale_table)
            assert all(s > 0 for s in scale_table)

        self.tail_mass = tail_mass

        if scale_bound is None and scale_table is not None:
            self.lower_bound_scale = LowerBound(self.scale_table[0])
        elif scale_bound > 0:
            self.lower_bound_scale = LowerBound(scale_bound)
        else:
            raise NotImplementedError

        self.register_buffer('scale_table',
                             torch.tensor(scale_table, dtype=torch.float)
                             if scale_table is not None else torch.tensor([], dtype=torch.float))

        self.register_buffer('scale_bound',
                             torch.tensor([scale_bound], dtype=torch.float)
                             if scale_bound is not None else torch.tensor([], dtype=torch.float))

        self.register_buffer('_offset', torch.tensor([], dtype=torch.int32))
        self.register_buffer('_cdf_length', torch.tensor([], dtype=torch.int32))
        self.register_buffer('_quantized_cdf', torch.tensor([], dtype=torch.int32))

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    @staticmethod
    def _standardized_cumulative(inputs: torch.Tensor) -> torch.Tensor:
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc((-(2 ** -0.5)) * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditional model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self._quantized_cdf.device  # pylint: disable=E0203
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, x, scales, means=None):
        if means is not None:
            x = x - means

        scales = self.lower_bound_scale(scales)

        x = torch.abs(x)
        upper = self._standardized_cumulative((0.5 - x) / scales)
        lower = self._standardized_cumulative((-0.5 - x) / scales)
        likelihood = upper - lower

        return likelihood

    @minkowski_tensor_wrapped('10')
    def forward(self,
                x: torch.Tensor,
                scales: torch.Tensor,
                means: torch.Tensor = None) \
            -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], List[bytes]]]:
        if self.training:
            x = self.quantize(x, 'noise', means)
            likelihood = self._likelihood(x, scales, means)

            if self.likelihood_lower_bound is not None:
                likelihood = self.likelihood_lower_bound(likelihood)
            return x, {'bits_loss': torch.log2(likelihood).sum()}

        else:
            indexes = self.build_indexes(scales)
            x_strings = self.compress(x, indexes, means=means)
            x_hat = self.decompress(x_strings, indexes, means=means)

            return x_hat, x_strings

    def build_indexes(self, scales):
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes


class _GaussianConditional(EntropyModel):
    def __init__(self,
                 scale_bound: float = 0.11,
                 tail_mass: float = 1e-9,
                 likelihood_bound: float = 1e-9,
                 entropy_coder: str = 'ans',
                 entropy_coder_precision: int = 16):
        super(_GaussianConditional, self).__init__(
            likelihood_bound=likelihood_bound,
            entropy_coder=entropy_coder,
            entropy_coder_precision=entropy_coder_precision)

        self.tail_mass = tail_mass

        self.lower_bound_scale = LowerBound(scale_bound)
        self.register_buffer('scale_bound',
                             torch.tensor([scale_bound], dtype=torch.float)
                             if scale_bound is not None else torch.tensor([], dtype=torch.float))

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    @staticmethod
    def _standardized_cumulative(inputs: torch.Tensor) -> torch.Tensor:
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc((-(2 ** -0.5)) * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update(self, scales):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(scales * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = scales.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, x, scales, means=None):
        if means is not None:
            x = x - means

        scales = self.lower_bound_scale(scales)

        x = torch.abs(x)
        upper = self._standardized_cumulative((0.5 - x) / scales)
        lower = self._standardized_cumulative((-0.5 - x) / scales)
        likelihood = upper - lower

        return likelihood

    @minkowski_tensor_wrapped('10')
    def forward(self,
                x: torch.Tensor,
                scales: torch.Tensor,
                means: torch.Tensor = None) \
            -> Tuple[torch.Tensor, Union[List[bytes], torch.Tensor]]:
        if self.training:
            x = self.quantize(x, 'noise', means)
            likelihood = self._likelihood(x, scales, means)

            if self.likelihood_lower_bound is not None:
                likelihood = self.likelihood_lower_bound(likelihood)
            return x, likelihood

        else:
            self.update(scales)
            indexes = self.build_indexes(scales)
            x_strings = self.compress(x, indexes, means=means)
            x_hat = self.decompress(x_strings, indexes, means=means)

            return x_hat, x_strings

    def build_indexes(self, scales):
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes

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
