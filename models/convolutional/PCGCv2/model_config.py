from dataclasses import dataclass
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    res_block_type: str = 'InceptionResNet'
    input_feature_type: str = 'Occupation'  # Occupation or Coordinates
    compressed_channels: int = 8
    reconstruct_loss_type: str = 'BCE'  # BCE or Dist
    dist_upper_bound = 2.0
    adaptive_pruning: bool = True
    bpp_loss_factor: float = 0.3
    reconstruct_loss_factor: float = 1.0
    hyper_bpp_balance_factor: float = 1.8

    # only for test phase:
    chamfer_dist_test_phase: bool = False
    mpeg_pcc_error_command: str = 'pc_error_d'
    mpeg_pcc_error_threads: int = 16
