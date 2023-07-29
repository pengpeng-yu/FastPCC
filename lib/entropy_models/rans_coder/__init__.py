import os
from typing import List

import numpy as np
from torch.utils.cpp_extension import load


def _test(coder, symbol_array: List[List[int]], index_array: List[List[int]] = None):
    symbol_array = np.array(symbol_array, dtype=np.int32)
    decoded_array = np.empty_like(symbol_array)
    if index_array is not None:
        index_array = np.array(index_array, dtype=np.int32)
        encoded_list = coder.encode_with_indexes(symbol_array, index_array)
        coder.decode_with_indexes(encoded_list, index_array, decoded_array)
    else:
        encoded_list = coder.encode(symbol_array)
        coder.decode(encoded_list, decoded_array)
    assert np.all(symbol_array == decoded_array)


def _bin_test(BinCoder):
    coder = BinCoder(2, 100)
    symbol_array = np.random.randint(0, 2, (2, 100), dtype=bool)
    prob_array = np.clip(np.round(np.random.rand(2, 100) * (1 << 16)), 1, (1 << 16) - 1)
    encoded_list = coder.encode(symbol_array, prob_array)
    decoded_array = np.empty_like(symbol_array)
    coder.decode(encoded_list, prob_array, decoded_array)
    assert np.all(symbol_array == decoded_array)


def _load_and_test():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_file_dir)
    build_directory = os.path.join(current_file_dir, 'build')
    os.makedirs(build_directory, exist_ok=True)
    rans_ext_cpp = load(
        name='rans_ext_cpp',
        extra_include_paths=[current_file_dir],
        sources=[os.path.join(
            current_file_dir, 'rans_coder_wrapper.cpp'
        )],
        extra_cflags=['-fopenmp', '-Wall', '-Wextra', '-Wconversion', '-O3'],
        # '-DNDEBUG'
        build_directory=build_directory,
        verbose=True
    )
    batched_pmf_to_quantized_cdf = rans_ext_cpp.batched_pmf_to_quantized_cdf
    Coder = rans_ext_cpp.IndexedRansCoder
    BinCoder = rans_ext_cpp.BinaryRansCoder
    np.random.seed(0)

    coder = Coder(True, 2, 100)
    float_pmfs = np.random.rand(3, 4).astype(np.float64) / 4
    offsets = np.array([-2, -2, -2], dtype=np.int32)
    coder.init_with_pmfs(float_pmfs, offsets)
    _test(coder, [[-2049, -2049], [2049, 2049]],
          [[0, 1], [2, 1]])
    _test(coder, [[-2, -1], [0, 10]],
          [[0, 1], [2, 1]])

    float_pmfs = np.random.rand(3, 4).astype(np.float64) / 4
    offsets = np.array([-2, -2, -2], dtype=np.int32)
    quantized_cdfs = batched_pmf_to_quantized_cdf(
        float_pmfs, offsets, True
    )
    coder.init_with_quantized_cdfs(quantized_cdfs, offsets)
    _test(coder, [[-2049, -2049], [2049, 2049]],
          [[0, 1], [2, 1]])
    _test(coder, [[-2, -1], [0, 10]],
          [[0, 1], [2, 1]])

    float_pmfs = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [2 ** -17, 1, 0, 0]], dtype=np.float64)
    offsets = np.array([0] * len(float_pmfs), dtype=np.int32)
    coder.init_with_pmfs(float_pmfs, offsets)
    quantized_cdfs = coder.get_cdfs()
    assert np.all(quantized_cdfs == np.array([[0, 1, 65536], *([[0, 65535, 65536]] * 3)], dtype=np.uint32))
    assert np.all(offsets == np.array([4, 0, 3, 1], np.int32))
    _test(coder, [[-2, -1], [0, 10]],
          [[0, 1], [2, 2]])

    coder = Coder(True, 8, 100)
    float_pmfs = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [2 ** -17, 1, 0, 0]], dtype=np.float64)
    offsets = np.array([0] * len(float_pmfs), dtype=np.int32)
    coder.init_with_pmfs(float_pmfs, offsets)
    _test(coder, [[0], [1], [0], [1], [0], [1], [3], [3]],
          [[0], [0], [1], [1], [2], [2], [3], [3]])

    coder = Coder(False, 4, 100)
    float_pmfs = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.float64)
    offsets = np.array([0] * len(float_pmfs), dtype=np.int32)
    coder.init_with_pmfs(float_pmfs, offsets)
    _test(coder, [[0, 1, 1, 0]] * 4)

    _bin_test(BinCoder)

    return batched_pmf_to_quantized_cdf, Coder, BinCoder


try:
    batched_pmf_to_quantized_cdf, IndexedRansCoder, BinaryRansCoder = _load_and_test()
except Exception as e:
    print('Error when loading RANS cpp ext')
    raise e
