from dataclasses import dataclass
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    res_block_type: str = 'InceptionResNet'
    compressed_channels: int = 8
    bottleneck_scaler: int = 2 ** 7
