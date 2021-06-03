from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig(SimpleConfig):
    # Files list can be generated automatically
    root: str = 'datasets/8iVFBv2'
    filelist_path: str = 'list.txt'

    with_color: bool = False
    with_file_path: bool = True

    resolution: int = 1024

