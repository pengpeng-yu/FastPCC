from dataclasses import dataclass
from typing import Tuple

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
    bottleneck_scaler: int = 1
    prior_indexes_scaler: float = 1.0
    prior_indexes_range: Tuple[int, ...] = (64, )
    fea_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 1)
    parameter_fns_mlp_num: int = 2
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
    lossless_fea_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1)

    # For geo lossless based EM and residual-aided GenerativeUpsample
    hybrid_hyper_decoder_fea: bool = False
    upper_fea_grad_scaler: float = 1.0

    # For residual-aided GenerativeUpsample
    coord_lossy_residuals: bool = False
    decoder_aware_residuals: bool = False

    # Geo lossless based EM settings
    # Valid if recurrent_part_enabled or lossless_coord_enabled
    skip_encoding_fea: Tuple[int, ...] = ()

    # Loss items
    bpp_loss_factor: float = 0.2
    coord_recon_loss_type: str = 'BCE'  # BCE or Dist
    coord_recon_p2points_weighted_bce: bool = False
    dist_upper_bound = 2.0
    adaptive_pruning: bool = True
    adaptive_pruning_num_scaler: float = 1.0
    coord_recon_loss_factor: float = 1.0
    color_recon_loss_type: str = 'MSE_YUVBT709'
    color_recon_loss_factor: float = 1.0
    warmup_steps: int = 0
    warmup_bpp_loss_factor: float = 0.2
    linear_warmup: bool = False

    # Only for test phase:
    mpeg_pcc_error_command: str = 'pc_error_d'
    mpeg_pcc_error_processes: int = 16
