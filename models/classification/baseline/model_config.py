from dataclasses import dataclass
from lib.config import SimpleConfig

@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 1024
    neighbor_num: int = 16
    classes_num: int = 40
    anchor_points: int = 6