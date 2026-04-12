import os
import os.path as osp
import sys
from typing import Tuple

import torch
from torch.utils.cpp_extension import load


def _load_ext():
    root = osp.dirname(osp.abspath(__file__))
    build_dir = osp.join(root, 'build')
    os.makedirs(build_dir, exist_ok=True)
    if sys.platform == 'win32':
        cxx_args = ['/O2']
        nvcc_args = ['-O3']
    elif sys.platform == 'linux':
        cxx_args = ['-O3']
        nvcc_args = ['-O3']
    else:
        raise NotImplementedError(sys.platform)
    return load(
        name='knn3d_ext',
        sources=[
            osp.join(root, 'src', 'binding.cpp'),
            osp.join(root, 'src', 'knn3d.cu'),
        ],
        extra_include_paths=[osp.join(root, 'src')],
        extra_cflags=cxx_args,
        extra_cuda_cflags=nvcc_args,
        build_directory=build_dir,
        verbose=False,
    )


knn3d_ext = _load_ext()


def knn3d(
        p1: torch.Tensor,
        p2: torch.Tensor,
        K: int,
        version: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return knn3d_ext.knn3d(p1, p2, int(K), int(version))
