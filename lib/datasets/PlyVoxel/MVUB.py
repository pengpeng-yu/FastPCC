from lib.datasets.PlyVoxel.Base import Dataset
from lib.datasets.PlyVoxel.Base import Config as BaseDatasetConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Config(BaseDatasetConfig):
    root: Union[str, Tuple[str, ...]] = 'datasets/MVUB'

    ori_resolution: Union[int, Tuple[int, ...]] = 512
    resolution: Union[int, Tuple[int, ...]] = 512
