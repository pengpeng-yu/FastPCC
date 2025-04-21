from dataclasses import dataclass
from typing import Tuple

from lib.config import SimpleConfig


@dataclass
class Config(SimpleConfig):
    channels: int = 256
    max_stride_wo_recurrent: int = 2048
    max_stride: int = 4096
    fea_stride: int = 16
    torchsparse_dataflow: str = 'ImplicitGEMM'  # ImplicitGEMM GatherScatter FetchOnDemand CodedCSR

    skip_top_scales_num: int = 0  # Test phase
