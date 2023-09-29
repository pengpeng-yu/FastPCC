from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class DatasetConfig(SimpleConfig):
    # Files list can be generated automatically using all.csv.
    root: str = 'datasets/ShapeNet/ShapeNetCore.v2'
    shapenet_all_csv: str = 'all.csv'
    train_filelist_path: str = 'all_list_obj.txt'
    test_filelist_path: str = 'test_list_obj.txt'
    train_divisions: Union[str, Tuple[str, ...]] = ('train', 'val', 'test')
    test_divisions: Union[str, Tuple[str, ...]] = 'test'
    generate_cache: bool = True

    # '.obj' or '.solid.binvox' or '.surface.binvox' or ['.solid.binvox', '.surface.binvox']
    data_format: Union[str, Tuple[str, ...]] = '.obj'

    # For '.obj' files.
    mesh_sample_points_num: int = 2500000
    mesh_sample_point_method: str = 'uniform'
    mesh_sample_point_resolution: int = 256
    ply_cache_dtype: str = '<u2'

    random_rotation: bool = True
    random_offset: Union[int, Tuple[int, ...]] = 0
    kd_tree_partition_max_points_num: int = 0

    resolution: int = 128
