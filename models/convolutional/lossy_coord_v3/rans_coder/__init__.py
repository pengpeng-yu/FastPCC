import os
import os.path as osp

import numpy as np
from torch.utils.cpp_extension import load


def _load_and_test():
    current_file_dir = osp.dirname(osp.abspath(__file__))
    build_directory = osp.join(current_file_dir, 'build')
    os.makedirs(build_directory, exist_ok=True)
    simple_rans_ext_cpp = load(
        name='simple_rans_ext_cpp',
        extra_include_paths=[current_file_dir, osp.join(os.getcwd(), 'lib', 'entropy_models', 'rans_coder')],
        sources=[osp.join(
            current_file_dir, 'simple_rans_wrapper.cpp'
        )],
        extra_cflags=['-Wall', '-Wextra', '-Wconversion', '-march=native', '-O3'],
        # '-DNDEBUG'
        build_directory=build_directory,
        verbose=True
    )
    RansEncoder = simple_rans_ext_cpp.RansEncoder
    RansDecoder = simple_rans_ext_cpp.RansDecoder

    quan_cdf = np.array([[1, 2, 3, 4, 65535],
                         [1, 2, 3, 5, 65535],
                         [2, 3, 4, 6, 65535],
                         [2, 3, 4, 7, 65535],
                         [1, 2, 3, 8, 65535],
                         [1, 2, 3, 9, 65535]], dtype=np.uint16)
    quan_cdf2 = np.array([[1, 2, 4000, 5000, 65535],
                          [2, 3, 3000, 6000, 65535],
                          [3, 4, 3000, 7000, 65535],
                          [4, 5, 1000, 8000, 65535],
                          [5, 6, 5000, 9000, 65535],
                          [6, 7, 6000, 10000, 65535]], dtype=np.uint16)
    org = np.array([2, 4, 1, 1, 2, 3, 0, 2, 4, 2, 1, 1], dtype=np.uint16)

    encoder = RansEncoder(8 * 1024 * 1024)  # cached bytes 8MB
    encoder.encode(quan_cdf2, org[6:12])
    encoder.encode(quan_cdf, org[:6])
    s = encoder.flush()

    decoder = simple_rans_ext_cpp.RansDecoder()
    decoder.flush(s)  # important to keep the reference to encoded bytes before decoding is done
    decoded = np.zeros((12,), dtype=np.uint16)
    decoder.decode(quan_cdf, decoded[:6])
    decoder.decode(quan_cdf2, decoded[6:12])
    assert (decoded == org).all()

    encoder.flush()
    encoder.encode(quan_cdf, org[:6])
    encoder.encode(quan_cdf2, org[6:12])
    s = encoder.flush()

    decoder.flush(s)  # important to keep the reference to encoded bytes before decoding is done
    decoded = np.zeros((12,), dtype=np.uint16)
    decoder.decode(quan_cdf2, decoded[6:12])
    decoder.decode(quan_cdf, decoded[:6])
    assert (decoded == org).all()

    return RansEncoder, RansDecoder


try:
    RansEncoder, RansDecoder = _load_and_test()
except Exception as e:
    print('Error when loading RANS cpp ext')
    raise e
