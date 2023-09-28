from lib.datasets.PlyVoxel.Base import Dataset
from lib.datasets.PlyVoxel.Base import Config as BaseDatasetConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Config(BaseDatasetConfig):
    root: Union[str, Tuple[str, ...]] = (
        'datasets/MPEG_GPCC_CTC/Dense', 'datasets/MPEG_GPCC_CTC/Dense'
    )
    filelist_path: Union[str, Tuple[str, ...]] = (
        'Dense_4096.txt', 'Dense_16384.txt'
    )
    resolution: Union[int, Tuple[int, ...]] = (4096, 16384)
