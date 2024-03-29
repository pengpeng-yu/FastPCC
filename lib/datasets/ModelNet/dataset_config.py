from lib.simple_config import SimpleConfig
from dataclasses import dataclass


@dataclass
class DatasetConfig(SimpleConfig):
    # Files list can be generated automatically when using original modelnet dataset.
    root: str = 'datasets/modelnet40_normal_resampled'
    classes_names: str = 'modelnet40_shape_names.txt'
    train_filelist_path: str = 'train_list.txt'
    test_filelist_path: str = 'test_list.txt'

    with_class: bool = False
    random_rotation: bool = False

    # works only when using txt file like modelnet40_normal_resampled
    sample_method: str = 'uniform'

    # works when using points as model input, this is the amount of input points
    # when using voxels as model input, this is the amount of points before vocalization
    input_points_num: int = 8192

    # Sampling points from mesh.
    # This works if OFF files are given in files list.
    # 'barycentric' or 'poisson_disk' or 'uniform' (using open3d)
    mesh_sample_point_method: str = 'uniform'

    # for ME sparse tensor. 0 means no quantization (use points as model input).
    resolution: int = 0
