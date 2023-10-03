from dataclasses import dataclass
from typing import Tuple
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    sample_method: str = 'uniform'
    sample_rate: float = 0.25
    neighbor_num: int = 8
    channels: Tuple[int, ...] = (3, 16, 32, 64)
    neighbor_feature_channels: Tuple[int, ...] = (16, 16, 16, 16)
    compressed_channels: int = 32

    res_em_index_ranges: Tuple[int, ...] = (16, 16, 16, 16)

    reconstruct_loss_factor: float = 1e5
    bpp_loss_factor: float = 1.
