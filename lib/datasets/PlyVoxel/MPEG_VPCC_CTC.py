from lib.datasets.PlyVoxel.Base import Dataset
from lib.datasets.PlyVoxel.Base import Config as BaseDatasetConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Config(BaseDatasetConfig):
    root: Union[str, Tuple[str, ...]] = ('datasets/Owlii', 'datasets/8iVFBv2')
    filelist_path: Union[str, Tuple[str, ...]] = (
        'list_basketball_player_dancer_all.txt',
        'list_all.txt'
    )

    ori_resolution: Union[int, Tuple[int, ...]] = (2048, 1024)
    resolution: Union[int, Tuple[int, ...]] = (2048, 1024)
