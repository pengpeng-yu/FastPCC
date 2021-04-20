from dataclasses import dataclass
from lib.config import SimpleConfig

@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 8192
    input_points_dim: int = 3
    sample_method: str = 'uniform'
    neighbor_num: int = 16

    bpp_loss_factor: float = 1e-4
    aux_loss_factor: float = 1e2
