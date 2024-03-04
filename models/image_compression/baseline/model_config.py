from dataclasses import dataclass
from typing import Tuple

from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    use_batch_norm: bool = False
    activation: str = 'prelu'
    channels: Tuple[int, ...] = (32, 64, 128, 128)
    bottleneck_scaler: int = 2

    bpp_loss_factor: float = 1.0
    recon_loss_factor: float = 16384.0
