from dataclasses import dataclass
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    res_block_type: str = 'InceptionResNet'
    compressed_channels: int = 8
    bottleneck_scaler: int = 2 ** 7
    bpp_loss_factor: float = 0.001
    aux_loss_factor: float = 10.0

    resolution: int = 128  # only used for computing chamfer distances, should depends on dataset config.

