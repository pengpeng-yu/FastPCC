from dataclasses import dataclass
from typing import Tuple

from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    # Basic network structure
    minkowski_algorithm: str = 'DEFAULT'
    basic_block_type: str = 'ResBlock'
    conv_region_type: str = 'HYPER_CUBE'
    basic_block_num: int = 1
    use_batch_norm: bool = False
    activation: str = 'relu'

    # Basic compression settings
    compressed_channels: int = 8
    bottleneck_process: str = 'noise'
    bottleneck_scaler: int = 1
    prior_indexes_scaler: float = 0.0
    prior_indexes_range: Tuple[int, ...] = (1024, 1024)
    parameter_fns_mlp_num: int = 2
    quantize_indexes: bool = False  # during training

    # Normal part of network
    encoder_channels: Tuple[int, ...] = (8, 32)
    decoder_channels: int = 8

    # Recurrent part (lossless) of network
    skip_encoding_fea: int = -1
    recurrent_part_channels: int = 32

    # Lossless compression settings
    lossless_coord_indexes_range: Tuple[int, ...] = (1024, 1024)
    lossless_fea_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1)

    # Loss items
    bpp_loss_factor: float = 0.2
    adaptive_pruning: bool = True
    adaptive_pruning_num_scaler: float = 1.0
    coord_recon_loss_factor: float = 1.0
    color_recon_loss_factor: float = 1.0
    warmup_steps: int = 0
    warmup_bpp_loss_factor: float = 0.2
    linear_warmup: bool = False

    mpeg_pcc_error_command: str = 'pc_error_d'
