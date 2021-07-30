from typing import Union

import torch
import torch.autograd


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


def lower_bound(x: torch.Tensor, bound: Union[int, float, torch.Tensor]) \
        -> torch.Tensor:
    return LowerBoundFunction.apply(x, bound)


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


def upper_bound(x: torch.Tensor, bound: Union[int, float, torch.Tensor]) \
        -> torch.Tensor:
    return UpperBoundFunction.apply(x, bound)


@torch.no_grad()
def quantization_offset(distribution):
    try:
        offset = distribution.mean()
        return offset - torch.round(offset)
    except NotImplementedError:
        return 0
