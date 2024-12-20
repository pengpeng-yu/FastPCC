from typing import Union

import torch
import torch.autograd


class GradScalerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.save_for_backward(factor)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        factor, = ctx.saved_tensors
        return grad_output * factor, None


def grad_scaler(x: torch.Tensor, scaler: Union[float, torch.Tensor]) -> torch.Tensor:
    if scaler == 1.0:
        return x
    else:
        if not isinstance(scaler, torch.Tensor):
            scaler = torch.tensor([scaler], dtype=x.dtype, device=x.device)
        return GradScalerFunction.apply(x.clone(), scaler)


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
