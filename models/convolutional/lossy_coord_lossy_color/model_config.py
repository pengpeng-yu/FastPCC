from dataclasses import dataclass
from typing import Tuple

from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    # Basic network structure
    minkowski_algorithm: str = 'DEFAULT'
    conv_region_type: str = 'HYPER_CUBE'
    activation: str = 'relu'

    # Basic compression settings
    compressed_channels: Tuple[int, ...] = 1
    bottleneck_process: str = 'noise'
    bottleneck_scaler: int = 1

    # Normal part of network
    encoder_channels: Tuple[int, ...] = (8, 32)
    decoder_channels: int = 8
    adaptive_pruning: bool = True
    adaptive_pruning_scaler_train: float = 1.0
    adaptive_pruning_scaler_test: float = 1.0

    # Geo lossless part of network
    geo_lossl_if_sample: Tuple[int, ...] = (1, 1)
    geo_lossl_channels: Tuple[int, ...] = (128, 128, 1)

    # Loss items
    bits_loss_factor: float = 0.2
    coord_recon_loss_factor: float = 1.0
    color_recon_loss_factor: float = 1.0
    warmup_fea_loss_steps: int = 1
    warmup_color_loss_steps: int = 1
    warmup_fea_loss_factor: float = 0.2
    warmup_color_loss_factor: float = 1.0
    linear_warmup: bool = False

    mpeg_pcc_error_command: str = 'pc_error_d'

    def check_local_value(self):
        if len(self.compressed_channels) == 1:
            self.compressed_channels *= len(self.geo_lossl_channels)
