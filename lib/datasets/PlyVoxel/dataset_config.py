from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class DatasetConfig(SimpleConfig):
    # Files list can be generated automatically
    root: Union[str, Tuple[str, ...]] = ('datasets/MVUB', 'datasets/8iVFBv2', 'datasets/Owlii')
    filelist_path: Union[str, Tuple[str, ...]] = 'list.txt'
    file_path_pattern: Union[str, Tuple[str, ...]] = '**/*.ply'  # works if filelist does not exist

    with_color: bool = False
    with_normal: bool = False
    with_file_path: bool = True

    ori_resolution: Union[int, Tuple[int, ...]] = (512, 1024, 2048)  # depends on the datasets themselves
    resolution: Union[int, Tuple[int, ...]] = (512, 1024, 2048)  # target resolution
    with_resolution: bool = False
    with_ori_resolution: bool = False
