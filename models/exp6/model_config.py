from dataclasses import dataclass
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 4096
    input_points_dim: int = 3
    sample_method: str = 'uniform'
    neighbor_num: int = 16

    quantize_loss_factor: float = 0.5

