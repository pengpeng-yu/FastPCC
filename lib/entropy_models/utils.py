from typing import Union

import torch
import torch.autograd
from torch.distributions import Distribution


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


def lower_bound(x: torch.Tensor, bound: Union[int, float, torch.Tensor],
                gradient: str = "identity_if_towards") \
        -> torch.Tensor:
    if not isinstance(bound, torch.Tensor):
        bound = torch.tensor([bound], dtype=x.dtype, device=x.device)
    if gradient == "identity_if_towards":
        return LowerBoundFunction.apply(x, bound)
    elif gradient == "disconnected":
        return torch.maximum(x, bound)
    else:
        raise NotImplementedError


class UpperBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.min(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through_if = (x <= bound) | (grad_output > 0)
        return pass_through_if * grad_output, None


def upper_bound(x: torch.Tensor, bound: Union[int, float, torch.Tensor],
                gradient: str = "identity_if_towards") \
        -> torch.Tensor:
    if not isinstance(bound, torch.Tensor):
        bound = torch.tensor([bound], dtype=x.dtype, device=x.device)
    if gradient == "identity_if_towards":
        return UpperBoundFunction.apply(x, bound)
    elif gradient == "disconnected":
        return torch.minimum(x, bound)
    else:
        raise NotImplementedError


@torch.no_grad()
def quantization_offset(distribution: Distribution):
    if isinstance(distribution, torch.distributions.MixtureSameFamily):  # TODO
        pass

    else:
        assert isinstance(distribution, Distribution)
        try:
            offset = distribution.mean()
            return offset - torch.round(offset)
        except NotImplementedError:
            return 0
