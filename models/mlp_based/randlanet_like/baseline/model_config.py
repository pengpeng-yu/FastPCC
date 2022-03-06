from dataclasses import dataclass
from typing import Tuple
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    sample_method: str = 'uniform'
    sample_rate: float = 0.25
    neighbor_num: int = 8
    encoder_channels: Tuple[int, ...] = (16, 32, 64)
    encoder_neighbor_feature_channels: Tuple[int, ...] = (8, 16, 32, 32)
    compressed_channels: int = 32
    decoder_channels: Tuple[int, ...] = (64, 32, 16)
    decoder_neighbor_feature_channels: Tuple[int, ...] = (32, 16, 8)

    bottleneck_scaler: int = 2 ** 0
    res_em_index_ranges: Tuple[int, ...] = (16, 16, 16, 16)

    reconstruct_loss_factor: float = 1e5
    bpp_loss_factor: float = 1.

    # only for test phase:
    chamfer_dist_test_phase: bool = False
    mpeg_pcc_error_command: str = 'pc_error_d'
    mpeg_pcc_error_threads: int = 16
