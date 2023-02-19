import numpy as np


def _split_by_3(a: np.ndarray):
    x = np.full(a.shape, fill_value=0x1fffff, dtype=np.uint64)
    x &= a
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
    return x


def morton_encode_magicbits(xyz: np.ndarray) -> np.ndarray:
    assert xyz.dtype == np.uint32
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    a = _split_by_3(xyz)
    a[:, 1] <<= 1
    a[:, 2] <<= 2
    a[:, 0] |= a[:, 1]
    a[:, 0] |= a[:, 2]
    return a[:, 0]
