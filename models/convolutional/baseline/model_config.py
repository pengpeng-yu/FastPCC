from dataclasses import dataclass
from typing import Tuple, Union

from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    input_feature_type: str = 'Occupation'
    minkowski_algorithm: str = 'DEFAULT'  # DEFAULT, SPEED_OPTIMIZED, MEMORY_EFFICIENT
    basic_block_type: str = 'InceptionResBlock'  # ResBlock or InceptionResBlock
    conv_region_type: str = 'HYPER_CUBE'  # HYPER_CUBE or HYPER_CROSS
    basic_block_num: int = 3
    use_batch_norm: bool = False
    activation: str = 'relu'
    conv_trans_near_pruning: bool = False
    encoder_channels: Tuple[int, ...] = (16, 32, 64, 32)
    compressed_channels: int = 8
    decoder_channels: Tuple[int, ...] = (64, 32, 16)
    encoder_scaler: float = 1.0
    mpeg_gpcc_command: str = 'tmc3'

    hyperprior: str = 'None'
    hyper_compressed_channels: int = 8
    hyper_encoder_channels: Tuple[int, ...] = (16, 16, 16, 16)
    hyper_decoder_channels: Tuple[int, ...] = (16, 16, 16, 16)
    prior_indexes_range: Tuple[int, ...] = (64, )
    hyper_encoder_scaler: float = 1.0
    prior_indexes_scaler: float = 1.0

    lossless_compression_based: bool = False
    lossless_coder_channels: int = 64
    lossless_coord_indexes_range: Tuple[int, ...] = (8, 8, 8, 8)
    lossless_detach_higher_fea: bool = False
    lossless_hybrid_hyper_decoder_fea: bool = False

    reconstruct_loss_type: str = 'BCE'  # BCE or Dist or Focal
    dist_upper_bound = 2.0
    adaptive_pruning: bool = True
    adaptive_pruning_num_scaler: float = 1.0
    bpp_loss_factor: float = 0.2
    reconstruct_loss_factor: float = 1.0
    warmup_steps: int = 0
    warmup_bpp_loss_factor: float = 0.2

    # only for test phase:
    chamfer_dist_test_phase: bool = False
    mpeg_pcc_error_command: str = 'pc_error_d'
    mpeg_pcc_error_threads: int = 8
