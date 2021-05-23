from dataclasses import dataclass
from lib.config import SimpleConfig

@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 4096
    input_points_dim: int = 3
    sample_method: str = 'fps'
    neighbor_num: int = 8

    bottleneck_scaler: int = 2 ** 7

    bpp_loss_factor: float = 1e-4
    aux_loss_factor: float = 100.0
