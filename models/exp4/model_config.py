from dataclasses import dataclass
from lib.config import SimpleConfig

@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 32768
    input_points_dim: int = 3
    sample_method: str = 'uniform'
    neighbor_num: int = 8

    bottleneck_scaler: int = 2 ** 7

    bpp_loss_factor: float = 0.001
    aux_loss_factor: float = 10.0
