from lib.datasets.PlyVoxel.Base import Dataset
from lib.datasets.PlyVoxel.Base import Config as BaseDatasetConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Config(BaseDatasetConfig):
    root: Union[str, Tuple[str, ...]] = ('datasets/Owlii', 'datasets/8iVFBv2', 'datasets/MVUB')
    resolution: Union[int, Tuple[int, ...]] = (2048, 1024, 512)
