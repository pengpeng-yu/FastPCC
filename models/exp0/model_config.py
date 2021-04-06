from dataclasses import dataclass
from lib.config import SimpleConfig

@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 8192
    input_points_dim: int = 3
    first_mlp_dim: int = 8

    encoder_blocks_num: int = 4
    chnl_upscale_per_block: int = 4
    sample_method: str = 'uniform'
    neighbor_num: int = 16
    dowansacle_per_block: int = 4

    encoded_points_dim: int = 1024

    decoder_blocks_num: int = 4
    chnl_downscale_per_block: int = 4
    upsacle_per_block: int = 4

    bpp_loss_factor: float = 1e-3
    aux_loss_factor: float = 100.0
