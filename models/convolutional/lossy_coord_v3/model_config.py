from dataclasses import dataclass
from typing import Tuple

from lib.config import SimpleConfig


@dataclass
class Config(SimpleConfig):
    channels: int = 128
    num_latents: Tuple[int, ...] = (0, 0, 2)  # (stride2, stride4, stride8, ...)
    lossl_geo_upsample: Tuple[int, ...] = (0, 0, 0)  # (stride2->1, stride4->2, stride8->4, ...)  1: lossl, 0: lossy
    max_stride: int = 64
    torchsparse_dataflow: str = 'ImplicitGEMM'  # ImplicitGEMM GatherScatter FetchOnDemand CodedCSR

    # Loss items
    coord_recon_loss_factor: float = 1.0
    warmup_steps: int = 0

    skip_top_scales_num: int = 0  # Test phase
