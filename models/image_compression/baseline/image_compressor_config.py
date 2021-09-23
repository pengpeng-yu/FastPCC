from dataclasses import dataclass
from typing import Tuple

from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    use_batch_norm: bool = False
    activation: str = 'relu'
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 128)
    decoder_channels: Tuple[int, ...] = (128, 128, 64, 32)

    hyper_encoder_channels: Tuple[int, ...] = (128, 128)
    hyper_decoder_channels: Tuple[int, ...] = (128, 256)
    prior_indexes_range: Tuple[int, ...] = (16, 16)

    bpp_loss_factor: float = 1.0
    reconstruct_loss_factor: float = 300.0
