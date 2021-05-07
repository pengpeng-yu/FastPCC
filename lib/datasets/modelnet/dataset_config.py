from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig(SimpleConfig):
    root: str = 'datasets/modelnet40_normal_resampled'
    train_filelist_path: str = 'train_list.txt'
    test_filelist_path: str = 'test_list.txt'
    input_points_num: int = 8192
    sample_method: str = 'uniform'
    with_normal_channel: bool = False
    with_classes: bool = False
    classes_names: str = 'modelnet40_shape_names.txt'
    random_rotation: bool = False

    # precompute the neighborhood-based feature using cpu
    # not implemented
    # precompute: str = None  # 'RandLANeighborFea' or 'RotationInvariantDistFea' or None
    # neighbor_num: int = 16
    # anchors_points: int = 4  # for RotationInvariantDistFea
    # model_sample_method: str = 'uniform'  # only uniform supported fro now. Input points are supposed to be shuffled.
    # model_sample_rates: Tuple[float] = (1.0, 0.5, 0.25, 0.125)