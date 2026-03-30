import sys
import os
import os.path as osp
import re
from glob import glob
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_target_sm_arch():
    arch_macro_dict = {
        '7.0': 'TARGET_SM70',
        '7.2': 'TARGET_SM72',
        '7.5': 'TARGET_SM75',
        '8.0': 'TARGET_SM80',
        '8.6': 'TARGET_SM86',
        '8.9': 'TARGET_SM89',
        '9.0': 'TARGET_SM90',
        '9.0a': 'TARGET_SM90A',
        '10.0': 'TARGET_SM100',
        '10.0a': 'TARGET_SM100A',
        '10.0f': 'TARGET_SM100F',
        '10.1': 'TARGET_SM101',
        '10.1a': 'TARGET_SM101A',
        '10.1f': 'TARGET_SM101F',
        '10.3': 'TARGET_SM103',
        '10.3a': 'TARGET_SM103A',
        '10.3f': 'TARGET_SM103F',
        '12.0': 'TARGET_SM120',
        '12.0a': 'TARGET_SM120A',
        '12.0f': 'TARGET_SM120F',
    }

    env_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    arch_list = []

    if env_arch_list is None:
        supported_sm = [int(arch.split('_')[1])
                        for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
        max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)

        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            capability = min(max_supported_sm, capability)
            arch = f'{capability[0]}.{capability[1]}'
            if arch not in arch_list:
                arch_list.append(arch)

    else:
        for arch in env_arch_list.replace(' ', ';').split(';'):
            match = re.search(
                r'(?:^|_)(?:(\d+)\.(\d)|(\d+)(\d))([af]?)(?:\+|$)',
                arch.strip().lower(),
            )
            if match is None: continue

            major_dot, minor_dot, major_compact, minor_compact, suffix = match.groups()
            major = major_dot or major_compact
            minor = minor_dot or minor_compact
            arch = f'{int(major)}.{minor}{suffix}'
            if arch not in arch_list:
                arch_list.append(arch)

    return [(arch_macro_dict[_], 1) for _ in arch_list if _ in arch_macro_dict]


def find_path_and_setup():
    current_file_dir = osp.dirname(osp.abspath(__file__))
    if 'CUTLASS_HOME' in os.environ:
        cutlass_home = os.environ['CUTLASS_HOME']
    else:
        cutlass_home = osp.join(osp.dirname(osp.dirname(osp.dirname(current_file_dir))), 'cutlass')
        if not osp.isdir(cutlass_home):
            cutlass_home = osp.join(current_file_dir, 'cutlass')
            if not osp.isdir(cutlass_home):
                raise FileNotFoundError(cutlass_home)

    cutlass_include = osp.abspath(osp.join(cutlass_home, 'include'))
    src_dir = osp.abspath(osp.join(current_file_dir, 'src'))
    src_files = [osp.abspath(_) for _ in glob(osp.join(src_dir, '**/*.cu'), recursive=True)]

    if sys.platform == 'win32':
        cxx_args = ['/O2', '/std:c++17', '/Zc:__cplusplus']  # '/W3'
        nvcc_args = [
            '-O3',
            '--expt-relaxed-constexpr',
            '-Xcompiler', '/O2',
            '-Xcompiler', '/std:c++17',
            '-Xcompiler', '/Zc:__cplusplus',
        ]
    elif sys.platform == 'linux':
        cxx_args = ['-O3', '-std=c++17', '-fno-strict-aliasing']  # '-Wall', '-Wextra', '-Wconversion',]
        nvcc_args = ['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                     '-ftemplate-backtrace-limit=0']
    else:
        raise NotImplementedError(sys.platform)

    setup(
        name='int_sparse_conv_ext',
        ext_modules=[
            CUDAExtension(
                name='int_sparse_conv_ext',
                sources=src_files,
                include_dirs=[cutlass_include, src_dir],
                define_macros=[('CUTLASS_DEBUG_TRACE_LEVEL', 0)] + get_target_sm_arch(),
                extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args},
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
