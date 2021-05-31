from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig(SimpleConfig):
    # when   coordinates are in [0, 1],
    # Files list can be generated automatically
    root: str = 'datasets/MVUB'
    filelist_path: str = 'list.txt'

    with_color: bool = False

