from lib.simple_config import SimpleConfig
from dataclasses import dataclass


@dataclass
class DatasetConfig(SimpleConfig):
    root: str = 'datasets/modelnet40_normal_resampled'
    train_filelist_path: str = 'train_list.txt'
    test_filelist_path: str = 'test_list.txt'
    input_points_num: int = 8192
    sample_method: str = 'uniform'
    with_normal_channel: bool = False
