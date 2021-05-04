from dataclasses import dataclass
from lib.config import SimpleConfig

@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 8192
    input_points_dim: int = 3
    neighbor_num: int = 16

    quantize_loss_factor: float = 2.0
    balance_loss_factor: float = 2.0
