import sys
import os
import os.path as osp
from glob import glob

from torch.utils.cpp_extension import load


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
    src_dir = current_file_dir
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
