from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig(SimpleConfig):
    # relative to dataset root.
    # when using modelnet40_normal_resampled(txt file), coordinates are left untouched,
    # when using original modelnet(off file) with resolution = 0, coordinates are in [0, 1],
    # when using original modelnet(off file) with resolution!= 0, coordinates are in [0, resolution).
    # Files list can be generated automatically when using original modelnet dataset.
    root: str = 'datasets/modelnet40_normal_resampled'
    classes_names: str = 'modelnet40_shape_names.txt'
    train_filelist_path: str = 'train_list.txt'
    test_filelist_path: str = 'test_list.txt'

    with_classes: bool = False
    random_rotation: bool = False

    # works only when using txt file like modelnet40_normal_resampled
    sample_method: str = 'uniform'
    with_normal_channel: bool = False

    # works when using txt file or resolution == 0
    input_points_num: int = 8192

    # for resampling mesh. This works if OFF files are given in files list.
    # 'barycentric' or 'poisson_disk' or 'uniform'
    mesh_sample_point_method: str = 'uniform'

    # for sparse tensor. 0 means no quantization
    resolution: int = 0
