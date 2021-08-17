from dataclasses import dataclass
from typing import Tuple
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 8192
    sample_method: str = 'uniform'
    sample_rate: float = 0.5
    neighbor_num: int = 16
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024)
    compressed_channels: int = 1024
    decoder_channels: Tuple[int, ...] = (512, 256, 96)

    bpp_loss_factor: float = 1
    reconstruct_loss_factor: float = 1e5
