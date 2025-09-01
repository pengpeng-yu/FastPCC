from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class DatasetConfig(SimpleConfig):
    # Files list can be generated automatically
    root: Union[str, Tuple[str, ...]] = ('datasets/MVUB', 'datasets/8iVFBv2', 'datasets/Owlii')
    filelist_path: Union[str, Tuple[str, ...]] = 'list.txt'
    file_path_pattern: Union[str, Tuple[str, ...]] = '**/*.ply'  # works if filelist does not exist
    list_sampling_interval: int = 1

    kd_tree_partition_max_points_num: Union[int, Tuple[int, ...]] = 0
    coord_scaler: Union[float, Tuple[float, ...]] = 1.0
    random_batch_coord_scaler_log2: Tuple[int, ...] = (0,)
    with_color: bool = False
    with_reflectance: bool = False
    random_flip: bool = False
    morton_sort: bool = False
    morton_sort_inverse: bool = False

    resolution: Union[int, Tuple[int, ...]] = (512, 1024, 2048)
