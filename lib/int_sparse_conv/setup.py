import sys
import os
import os.path as osp
from glob import glob
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_target_sm_arch():
    arch_marco_dict = {
        '7.0': 'TARGET_SM70',
        '7.2': 'TARGET_SM72',
        '7.5': 'TARGET_SM75',
        '8.0': 'TARGET_SM80',
        '8.6': 'TARGET_SM86',
        '8.9': 'TARGET_SM89',
        '9.0': 'TARGET_SM90',
        '9.0a': 'TARGET_SM90A',
        '10.0': 'TARGET_SM100',
        '10.1': 'TARGET_SM101',
        '10.3': 'TARGET_SM103',
        '12.0': 'TARGET_SM120',
    }

    env_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    arch_list = []
    if env_arch_list is None:
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            supported_sm = [int(arch.split('_')[1])
                            for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
            max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
            capability = min(max_supported_sm, capability)
            arch = f'{capability[0]}.{capability[1]}'
            if arch not in arch_list:
                arch_list.append(arch)
    else:
        for arch in env_arch_list.replace(' ', ';').split(';'):
            arch_list.append(arch.split('+', 1)[0])
    return [(arch_marco_dict[_], 1) for _ in arch_list if _ in arch_marco_dict]


def find_path_and_setup():
    current_file_dir = osp.dirname(osp.abspath(__file__))
    if 'CUTLASS_HOME' in os.environ:
        cutlass_home = os.environ['CUTLASS_HOME']
    else:
        cutlass_home = osp.join(osp.dirname(osp.dirname(osp.dirname(current_file_dir))), 'cutlass')
        if not osp.isdir(cutlass_home):
            cutlass_home = osp.join(current_file_dir, 'cutlass')
            if not osp.isdir(cutlass_home):
                raise FileNotFoundError

    cutlass_include = osp.abspath(osp.join(cutlass_home, 'include'))
    src_dir = osp.abspath(osp.join(current_file_dir, 'src'))
    src_files = [osp.abspath(_) for _ in glob(osp.join(src_dir, '**/*.cu'), recursive=True)]

    setup(
        name='int_sparse_conv_ext',
        ext_modules=[
            CUDAExtension(
                name='int_sparse_conv_ext',
                sources=src_files,
                include_dirs=[cutlass_include, src_dir],
                define_macros=[('CUTLASS_DEBUG_TRACE_LEVEL', 0)] + get_target_sm_arch(),
                extra_compile_args={
                    'cxx': ['-march=native', '-fno-strict-aliasing', '-O3']
                            # '-Wall', '-Wextra', '-Wconversion',]
                            if sys.platform != 'win32' else
                            ['/O2'],  # '/W3'
                    'nvcc': ['-O3', '--expt-relaxed-constexpr',
                             '-ftemplate-backtrace-limit=0']},
            )],
        cmdclass={'build_ext': BuildExtension},
        options={
            "build_ext": {
                "build_temp": osp.join(current_file_dir, 'build', 'temp'),
                "build_lib": osp.join(current_file_dir, 'build'),
            }
        }
    )


find_path_and_setup()
