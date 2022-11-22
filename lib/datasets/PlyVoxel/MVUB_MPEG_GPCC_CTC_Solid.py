from lib.datasets.PlyVoxel.Base import Dataset
from lib.datasets.PlyVoxel.Base import Config as BaseDatasetConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Config(BaseDatasetConfig):
    root: Union[str, Tuple[str, ...]] = (
        'datasets/MVUB',
        'datasets/MPEG_GPCC_CTC/Solid', 'datasets/MPEG_GPCC_CTC/Solid', 'datasets/MPEG_GPCC_CTC/Solid'
    )
    filelist_path: Union[str, Tuple[str, ...]] = (
        'list.txt',
        'Solid_1024.txt', 'Solid_2048.txt', 'Solid_4096.txt'
    )

    ori_resolution: Union[int, Tuple[int, ...]] = (512, 1024, 2048, 4096)
    resolution: Union[int, Tuple[int, ...]] = (512, 1024, 2048, 4096)
