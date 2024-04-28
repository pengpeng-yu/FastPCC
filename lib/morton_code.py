from typing import Union

import numpy as np
import torch


def _split_by_3(a: Union[np.ndarray, torch.Tensor]):
    if isinstance(a, np.ndarray):  # assert np.all(0 <= a <= 0x1fffff)
        x = a.astype(np.uint64)
    else:
        assert isinstance(a, torch.Tensor)
        x = a.to(torch.int64, copy=True)
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


def morton_encode_magicbits(xyz: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    a = _split_by_3(xyz)
    a[:, 1] <<= 1
    a[:, 2] <<= 2
    a[:, 0] |= a[:, 1]
    a[:, 0] |= a[:, 2]
    return a[:, 0]
