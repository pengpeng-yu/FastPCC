from dataclasses import dataclass
from lib.config import SimpleConfig

@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 8192
    neighbor_num: int = 4
    classes_num: int = 40