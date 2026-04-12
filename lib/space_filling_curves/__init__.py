import sys
import os
import os.path as osp
from glob import glob
from typing import Union

import numpy as np
import torch
from torch.utils.cpp_extension import load


VALID_AXIS_ORDERS_3D = ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')


def _load_ext():
    if sys.platform == 'win32':
        cxx_args = ['/O2']
        nvcc_args = ['-O3']
    elif sys.platform == 'linux':
        cxx_args = ['-O3']
        nvcc_args = ['-O3']
    else:
        raise NotImplementedError(sys.platform)

    current_file_dir = osp.dirname(osp.abspath(__file__))
    build_directory = osp.join(current_file_dir, 'build')
    os.makedirs(build_directory, exist_ok=True)
    src_dir = osp.join(current_file_dir, 'src')
    src_files = [osp.abspath(_) for _ in glob(osp.join(src_dir, '**/*.cu'), recursive=True)]

    space_filling_curves_ext = load(
        name='space_filling_curves_ext',
        extra_include_paths=[current_file_dir],
        sources=src_files,
        extra_cflags=cxx_args,
        extra_cuda_cflags=nvcc_args,
        build_directory=build_directory,
        verbose=True
    )
    return space_filling_curves_ext


space_filling_curves_ext = _load_ext()


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
