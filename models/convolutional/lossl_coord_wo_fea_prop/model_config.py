from dataclasses import dataclass
from typing import Tuple

from lib.config import SimpleConfig


@dataclass
class Config(SimpleConfig):
    torchsparse_dataflow: str = 'ImplicitGEMM'  # ImplicitGEMM GatherScatter FetchOnDemand CodedCSR
    channels: int = 256
    max_stride_wo_recurrent: int = 2048
    max_stride: int = 8192
    fea_stride: int = 16

    skip_top_scales_num: int = 0  # Test phase
