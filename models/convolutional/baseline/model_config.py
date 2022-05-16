from dataclasses import dataclass
from typing import Tuple, Union

from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    # Basic network structure
    input_feature_type: str = 'Occupation'  # Occupation, Color
    minkowski_algorithm: str = 'DEFAULT'  # DEFAULT, SPEED_OPTIMIZED, MEMORY_EFFICIENT
    basic_block_type: str = 'InceptionResBlock'  # ResBlock or InceptionResBlock
    conv_region_type: str = 'HYPER_CUBE'  # HYPER_CUBE or HYPER_CROSS
    basic_block_num: int = 3
    use_batch_norm: bool = False
    activation: str = 'relu'
    first_conv_kernel_size: int = 3

    # Basic compression settings
    compressed_channels: int = 8
    bottleneck_process: str = 'noise'
    prior_indexes_scaler: float = 1.0
    prior_indexes_post_scaler: float = 1.0  # only valid for deep factorized conditional em.
    prior_indexes_range: Tuple[int, ...] = (64, )
    quantize_indexes: bool = False  # during training

    # Normal part of network
    encoder_channels: Tuple[int, ...] = (16, 32, 64, 32)
    decoder_channels: Tuple[int, ...] = (64, 32, 16)
    mpeg_gpcc_command: str = 'tmc3'

    hyperprior: str = 'None'
    hyper_compressed_channels: int = 8
    hyper_encoder_channels: Tuple[int, ...] = (16, 16, 16, 16)
    hyper_decoder_channels: Tuple[int, ...] = (16, 16, 16, 16)

    # Recurrent part (lossless) of network
    recurrent_part_enabled: bool = False
    recurrent_part_channels: int = 64

    # Lossless compression settings
    lossless_coord_enabled: bool = False
    lossless_color_enabled: bool = False
    lossless_coord_indexes_range: Tuple[int, ...] = (8, 8, 8, 8)
    lossless_hybrid_hyper_decoder_fea: bool = False
    lossless_part_upper_fea_grad_scaler: float = 1.0

    # Loss items
    bpp_loss_factor: float = 0.2
    coord_recon_loss_type: str = 'BCE'  # BCE or Dist or Focal
    dist_upper_bound = 2.0
    adaptive_pruning: bool = True
    adaptive_pruning_num_scaler: float = 1.0
    coord_recon_loss_factor: float = 1.0
    color_recon_loss_type: str = 'MSE_YUVBT709'
    color_recon_loss_factor: float = 1.0
    warmup_steps: int = 0
    warmup_bpp_loss_factor: float = 0.2

    # Only for test phase:
    chamfer_dist_test_phase: bool = False
    mpeg_pcc_error_command: str = 'pc_error_d'
    mpeg_pcc_error_threads: int = 8
