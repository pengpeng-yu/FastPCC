from lib.datasets.PlyVoxel.Base import Dataset
from lib.datasets.PlyVoxel.Base import Config as BaseDatasetConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Config(BaseDatasetConfig):
    root: Union[str, Tuple[str, ...]] = (
        'datasets/MPEG_GPCC_CTC/Dense',
        'datasets/MPEG_GPCC_CTC/Solid', 'datasets/MPEG_GPCC_CTC/Solid', 'datasets/MPEG_GPCC_CTC/Solid',
        'datasets/MVUB'
    )
    filelist_path: Union[str, Tuple[str, ...]] = (
        'Dense_16384.txt',
        'Solid_4096.txt', 'Solid_2048.txt', 'Solid_1024.txt',
        'list.txt'
    )
    resolution: Union[int, Tuple[int, ...]] = (16384, 4096, 2048, 1024, 512)
