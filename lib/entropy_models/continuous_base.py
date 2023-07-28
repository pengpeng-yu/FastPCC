from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.distributions
from torch.distributions import Distribution

from .utils import quantization_offset
from .rans_coder import IndexedRansCoder


class DistributionQuantizedCDFTable(nn.Module):
    """
    Provide function that can generate flat quantized CDF table
    used by range coder.
    """
    def __init__(self,
                 base: Distribution,
                 lower_bound: Union[int, torch.Tensor],
                 upper_bound: Union[int, torch.Tensor],
                 coding_batch_size: int,
                 overflow_coding: bool,
                 bottleneck_scaler: int
                 ):
        super(DistributionQuantizedCDFTable, self).__init__()
        self.base = base
        self.coding_batch_size = coding_batch_size
        self.overflow_coding = overflow_coding
        self.bottleneck_scaler = bottleneck_scaler
        assert lower_bound < upper_bound

        self.register_buffer(
            'lower_bound',
            lower_bound if isinstance(lower_bound, torch.Tensor)
            else torch.tensor(lower_bound, dtype=torch.int32).expand(self.batch_shape),
            persistent=False
        )
        self.register_buffer(
            'upper_bound',
            upper_bound if isinstance(upper_bound, torch.Tensor)
            else torch.tensor(upper_bound, dtype=torch.int32).expand(self.batch_shape),
            persistent=False
        )

        self.cdf_list: List[List[int]] = [[]]
        self.cdf_offset_list: List[int] = []
        self.requires_updating_cdf_table: bool = True
        self.range_coder = IndexedRansCoder(self.overflow_coding, self.coding_batch_size)

        if len(base.event_shape) != 0:
            raise NotImplementedError

    def update_base(self, new_base: Distribution):
        assert type(new_base) is type(self.base)
        assert new_base.batch_shape == self.base.batch_shape
        assert new_base.event_shape == self.base.event_shape
        self.base = new_base

    def mean(self):
        return self.base.mean()

    @property
    def batch_shape(self):
        return self.base.batch_shape

    @property
    def event_shape(self):
        return torch.Size([])

    @property
    def batch_numel(self):
        return self.base.batch_shape.numel()

    @property
    def batch_ndim(self):
        return len(self.base.batch_shape)

    def prob(self, value):
        if hasattr(self.base, 'prob'):
            return self.base.prob(value)
        else:
            raise NotImplementedError

    def log_prob(self, value):
        return self.base.log_prob(value)

    @torch.no_grad()
    def build_quantized_cdf_table(self):
        offset = quantization_offset(self.base)

        minima = (self.lower_bound - offset) * self.bottleneck_scaler
        maxima = (self.upper_bound.max() - offset) * self.bottleneck_scaler
        if torch.is_floating_point(minima):
            minima = torch.floor(minima).to(torch.int32)
        pmf_start = minima + offset
        max_length = maxima.to(torch.int32).item() - minima.max().item() + 1

        samples = torch.arange(max_length, device=pmf_start.device, dtype=torch.float)
        samples = samples.reshape(max_length, *[1] * len(self.base.batch_shape))
        samples = samples + pmf_start[None, ...]  # broadcast
        del pmf_start

        try:
            pmf = self.prob(samples / self.bottleneck_scaler)
        except NotImplementedError:
            pmf = torch.exp(self.log_prob(samples / self.bottleneck_scaler))
        del samples

        # Collapse batch dimensions of distribution.
        pmf = pmf.reshape(max_length, -1)
        minima = minima.reshape(-1)

        if self.overflow_coding:
            if not torch.all(pmf[0] == 0):
                print(f'Warning: Possible overflow at lower bound. Max PM: {pmf[0].max().item()}')
            if not torch.all(pmf[-1] == 0):
                print(f'Warning: Possible overflow at upper bound. Max PM: {pmf[-1].max().item()}')

        pmf = pmf.T.tolist()
        minima = minima.tolist()
        self.cdf_list, self.cdf_offset_list = self.range_coder.init_with_pmfs(pmf, minima)
        self.requires_updating_cdf_table = False

    def get_extra_state(self):
        return self.cdf_list, self.cdf_offset_list, self.requires_updating_cdf_table

    def set_extra_state(self, state):
        if state[2]:
            print('Warning: cached cdf table in state dict requires updating.\n'
                  'You need to call model.eval() to build it after loading state dict '
                  'before any inference.')
        else:
            self.cdf_list, self.cdf_offset_list, self.requires_updating_cdf_table = state
            self.range_coder.init_with_quantized_cdfs(self.cdf_list, self.cdf_offset_list)

    def train(self, mode: bool = True):
        """
        Use model.train() to invalidate cached cdf table.
        Use model.eval() to call build_quantized_cdf_table().
        """
        if mode is True:
            self.requires_updating_cdf_table = True
        else:
            if self.requires_updating_cdf_table:
                self.build_quantized_cdf_table()
        return super(DistributionQuantizedCDFTable, self).train(mode=mode)


class ContinuousEntropyModelBase(nn.Module):
    def __init__(self,
                 prior: Distribution,
                 coding_ndim: int,
                 bottleneck_process: str,
                 bottleneck_scaler: int,
                 lower_bound: Union[int, torch.Tensor],
                 upper_bound: Union[int, torch.Tensor],
                 batch_shape: torch.Size,
                 overflow_coding: bool):
        super(ContinuousEntropyModelBase, self).__init__()
        # "self.prior" is supposed to be able to generate
        # flat quantized CDF table used by range coder.
        self.prior: DistributionQuantizedCDFTable = DistributionQuantizedCDFTable(
            base=prior,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            coding_batch_size=batch_shape.numel(),
            overflow_coding=overflow_coding,
            bottleneck_scaler=bottleneck_scaler
        )
        self.quantize_bottleneck = 'quantization' in bottleneck_process
        if self.quantize_bottleneck:
            bottleneck_process = bottleneck_process.replace('quantization', '', 1)
        self.perturb_bottleneck = 'noise' in bottleneck_process
        if self.perturb_bottleneck:
            bottleneck_process = bottleneck_process.replace('noise', '', 1)
        assert bottleneck_process in (',', '_', ' ', '+', ''), \
            f'Unexpected bottleneck_process: {bottleneck_process}'
        self.coding_ndim = coding_ndim

    def perturb(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_noise"):
            setattr(self, "_noise", torch.empty(x.shape, dtype=torch.float, device=x.device))
        self._noise.resize_(x.shape)
        self._noise.uniform_(-0.5, 0.5)
        x = x + self._noise
        return x

    def process(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_bottleneck is True:
            x = x + (x.detach().round() - x.detach())
        if self.perturb_bottleneck is True:
            x = self.perturb(x)
        return x

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, offset=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset is None: offset = quantization_offset(self.prior.base)
        x -= offset
        torch.round_(x)
        quantized_x = x.to(torch.int32)
        x += offset
        return quantized_x, x

    @torch.no_grad()
    def dequantize(self, x: torch.Tensor, offset=None) -> torch.Tensor:
        if offset is None: offset = quantization_offset(self.prior.base)
        if isinstance(offset, torch.Tensor) and x.device != offset.device:
            x = x.to(offset.device)
        x += offset
        return x.to(torch.float)

    def forward(self, *args, **kwargs) \
            -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                     Tuple[torch.Tensor, Dict[str, torch.Tensor], List]]:
        raise NotImplementedError

    def compress(self, *args, **kwargs):
        raise NotImplementedError

    def decompress(self, *args, **kwargs):
        raise NotImplementedError
