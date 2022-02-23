import os

import torch
from torch.utils.cpp_extension import load


def _test(initialized_rans_coder, seqs, indexes):
    encoded = initialized_rans_coder.encode_with_indexes(seqs, indexes)
    decoded = initialized_rans_coder.decode_with_indexes(encoded, indexes)
    assert torch.all(torch.tensor(seqs, dtype=torch.int) == torch.tensor(decoded, dtype=torch.int))


def _load_and_test():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_file_dir)
    build_directory = os.path.join(current_file_dir, 'build')
    os.makedirs(build_directory, exist_ok=True)
    rans_ext_cpp = load(
        name='rans_ext_cpp',
        extra_include_paths=[current_file_dir],
        sources=[os.path.join(current_file_dir, 'indexed_rans_coder.cpp')],
        extra_cflags=['-Wall', '-Wextra', '-Wconversion', '-O3', ],  # '-DNDEBUG'
        build_directory=build_directory,
        verbose=True
    )

    torch.manual_seed(0)
    float_pmfs = torch.rand(3, 4, dtype=torch.float64) / 4
    quantized_cdfs = rans_ext_cpp.batched_pmf_to_quantized_cdf(float_pmfs.tolist(), 16)

    rans_coder = rans_ext_cpp.IndexedRansCoder(16)
    rans_coder.init_with_pmfs(float_pmfs.tolist(), [-2, -2, -2])
    _test(rans_coder, [[-2049, -2049, 2049, 2049]], [[0, 1, 2, 1]])
    _test(rans_coder, [[-2, -1, 0, 10]], [[0, 1, 2, 1]])

    rans_coder.init_with_quantized_cdfs(quantized_cdfs, [-3, -3, -3])
    _test(rans_coder, [[-2049, -2049, 2049, 2049]], [[0, 1, 2, 2]])
    _test(rans_coder, [[-2, -1, 0, 10]], [[0, 1, 2, 2]])

    return rans_ext_cpp.batched_pmf_to_quantized_cdf, rans_ext_cpp.IndexedRansCoder


try:
    batched_pmf_to_quantized_cdf, IndexedRansCoder = _load_and_test()
except Exception as e:
    print('Error when loading RANS cpp ext')
    raise e
