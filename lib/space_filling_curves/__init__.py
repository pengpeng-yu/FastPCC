from typing import Union

import numpy as np
import torch

from .src import space_filling_curves_ext

VALID_AXIS_ORDERS_3D = ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')


def _split_by_3(a: Union[np.ndarray, torch.Tensor]):
    if isinstance(a, np.ndarray):
        x = a.T.astype(np.uint64, order='C', copy=True)
    else:
        assert isinstance(a, torch.Tensor)
        x = a.T.to(torch.int64, memory_format=torch.contiguous_format, copy=True)
    x |= x << 32
    x &= 0x1f00000000ffff
    x |= x << 16
    x &= 0x1f0000ff0000ff
    x |= x << 8
    x &= 0x100f00f00f00f00f
    x |= x << 4
    x &= 0x10c30c30c30c30c3
    x |= x << 2
    x &= 0x1249249249249249
    return x  # 3 x N


def morton_encode_magicbits(
        xyz: Union[np.ndarray, torch.Tensor],
        axis_order: str = 'xyz',
        inverse: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
        assert (0 <= xyz <= 0x1fffff).all()
    """
    axis_order = axis_order.lower()
    if inverse:
        axis_order = axis_order[::-1]
    assert axis_order in VALID_AXIS_ORDERS_3D, axis_order

    if isinstance(xyz, torch.Tensor) and xyz.dtype == torch.int32 and xyz.is_cuda:
        return space_filling_curves_ext.morton3d_encode_magicbits(xyz, axis_order)

    assert xyz.ndim == 2 and xyz.shape[1] == 3
    axis_to_index = {'x': 0, 'y': 1, 'z': 2}
    i0, i1, i2 = [axis_to_index[axis] for axis in axis_order]
    a = _split_by_3(xyz)
    a[i1] <<= 1
    a[i2] <<= 2
    a[i0] |= a[i1]
    a[i0] |= a[i2]
    return a[i0]


def hilbert3d_encode_lut(
        xyz: Union[np.ndarray, torch.Tensor],
        bits=21,
        axis_order: str = 'xyz',
        inverse: bool = False):
    """
        assert (0 <= xyz <= 0x1fffff).all()
    """
    axis_order = axis_order.lower()
    if inverse:
        axis_order = axis_order[::-1]
    assert axis_order in VALID_AXIS_ORDERS_3D, axis_order

    if isinstance(xyz, torch.Tensor) and xyz.dtype == torch.int32 and xyz.is_cuda:
        return space_filling_curves_ext.hilbert3d_encode_lut(xyz, bits, axis_order)
    else:
        raise NotImplementedError
