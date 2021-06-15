from dataclasses import dataclass
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    input_points_num: int = 1024
    input_points_dim: int = 3
    sample_method: str = 'uniform'
    neighbor_num: int = 8

    bottleneck_scaler: int = 2 ** 7

    bpp_loss_factor: float = 0.001
    aux_loss_factor: float = 10.0

    # only for test phase:
    chamfer_dist_test_phase: bool = False
    mpeg_pcc_error_command: str = 'pc_error_d'
    mpeg_pcc_error_threads: int = 16
