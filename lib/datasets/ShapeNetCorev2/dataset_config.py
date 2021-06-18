from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class DatasetConfig(SimpleConfig):
    # Files list can be generated automatically using all.csv.
    root: str = 'datasets/ShapeNet/ShapeNetCore.v2'
    shapenet_all_csv: str = 'all.csv'
    train_filelist_path: str = 'train_list.txt'
    val_filelist_path: str = 'val_list.txt'  # not used for now.
    test_filelist_path: str = 'test_list.txt'

    # '.obj' or '.solid.binvox' or '.surface.binvox' or ['.solid.binvox', '.surface.binvox']
    data_format: Union[str, Tuple[str]] = '.surface.binvox'

    mesh_sample_points_num: int = 0
    mesh_sample_point_method: str = 'uniform'
    random_rotation: bool = True

    resolution: int = 128
    with_file_path: bool = False
    with_resolution: bool = False
    with_ori_resolution: bool = False
