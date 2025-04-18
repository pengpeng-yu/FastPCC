from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class DatasetConfig(SimpleConfig):
    root: str = 'datasets/KITTI/sequences'
    train_filelist_path: str = 'train_list.txt'
    test_filelist_path: str = 'test_list.txt'
    train_subset_index: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    test_subset_index: Tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    list_sampling_interval: int = 1
    random_flip: bool = False
    morton_sort: bool = False
    morton_sort_inverse: bool = False
    resolution: Union[int, float] = 4096
    flag_sparsepcgc: bool = False
