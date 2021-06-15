from dataclasses import dataclass
from lib.config import SimpleConfig

@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 4096
    neighbor_num: int = 8
    classes_num: int = 40
    anchor_points: int = 3
