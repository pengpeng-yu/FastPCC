from lib.simple_config import SimpleConfig
from dataclasses import dataclass


@dataclass
class DatasetConfig(SimpleConfig):
    root: str = 'datasets/PCGCv2Data'
    train_filelist_path: str = 'train_list.txt'
    test_filelist_path: str = 'test_list.txt'

    train_split_ratio: float = 1.0

    resolution: int = 128
