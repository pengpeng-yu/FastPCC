from dataclasses import dataclass
from lib.config import SimpleConfig


@dataclass
class ModelConfig(SimpleConfig):
    res_block_type: str = 'InceptionResNet'
    compressed_channels: int = 8
    bottleneck_scaler: int = 2 ** 7
    bpp_loss_factor: float = 0.1
    aux_loss_factor: float = 10.0

    # only for test phase:
    chamfer_dist_test_phase: bool = False
    mpeg_pcc_error_command: str = 'pc_error_d'
    mpeg_pcc_error_threads: int = 4
