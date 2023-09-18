from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig(SimpleConfig):
    root: str = 'datasets/KITTI/sequences'
    train_filelist_path: str = 'train_list.txt'
    test_filelist_path: str = 'val_list.txt'
    train_subset_index: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    test_subset_index: Tuple[int, ...] = (11,)
    coord_scaler: float = 1000.0
    ply_cache_dtype: str = '<u2'
