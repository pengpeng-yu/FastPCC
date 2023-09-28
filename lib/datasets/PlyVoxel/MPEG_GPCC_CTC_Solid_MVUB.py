from lib.datasets.PlyVoxel.Base import Dataset
from lib.datasets.PlyVoxel.Base import Config as BaseDatasetConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Config(BaseDatasetConfig):
    root: Union[str, Tuple[str, ...]] = (
        'datasets/MPEG_GPCC_CTC/Solid', 'datasets/MPEG_GPCC_CTC/Solid', 'datasets/MPEG_GPCC_CTC/Solid',
        'datasets/MVUB'
    )
    filelist_path: Union[str, Tuple[str, ...]] = (
        'Solid_1024.txt', 'Solid_2048.txt', 'Solid_4096.txt',
        'list.txt'
    )
    resolution: Union[int, Tuple[int, ...]] = (1024, 2048, 4096, 512)
