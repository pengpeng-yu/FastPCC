from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class DatasetConfig(SimpleConfig):
    root: Union[str, Tuple[str, ...]] = 'datasets/COCO'
    filelist_path: Union[str, Tuple[str, ...]] = 'train2017_list.txt'
    file_path_pattern: Union[str, Tuple[str, ...]] = 'train2017/*.jpg'

    channels_order: str = 'BGR'  # BGR or RGB
    target_shape_for_training: Tuple[int, ...] = (256, 256)
    random_h_flip: bool = False
    stride_for_test: int = 8
