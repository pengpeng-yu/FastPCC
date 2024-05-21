from lib.datasets.PlyVoxel.Base import Dataset
from lib.datasets.PlyVoxel.Base import Config as BaseDatasetConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Config(BaseDatasetConfig):
    root: Union[str, Tuple[str, ...]] = (
        'datasets/MPEG_GPCC_CTC/Solid', 'datasets/MPEG_GPCC_CTC/Solid', 'datasets/MPEG_GPCC_CTC/Solid'
    )
    filelist_path: Union[str, Tuple[str, ...]] = (
        'Solid_4096.txt', 'Solid_2048.txt', 'Solid_1024.txt'
    )
    resolution: Union[int, Tuple[int, ...]] = (4096, 2048, 1024)
