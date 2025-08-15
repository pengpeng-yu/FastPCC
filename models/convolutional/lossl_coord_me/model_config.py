from dataclasses import dataclass
from typing import Tuple

from lib.config import SimpleConfig


@dataclass
class Config(SimpleConfig):
    minkowski_algorithm: str = 'DEFAULT'
    conv_region_type: str = 'HYPER_CUBE'
    channels: int = 256
    max_stride_wo_recurrent: int = 2048
    max_stride: int = 8192
    fea_stride: int = 16
    use_more_ch_for_multi_step_pred: bool = True

    skip_top_scales_num: int = 0  # Test phase
    cal_avs_pc_evalue: bool = False
