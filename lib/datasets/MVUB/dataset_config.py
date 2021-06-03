from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig(SimpleConfig):
    # Files list can be generated automatically
    root: str = 'datasets/MVUB'
    filelist_path: str = 'list.txt'

    with_color: bool = False
    with_file_path: bool = True

    ori_resolution: int = 512
    resolution: int = 512  # target resolution

