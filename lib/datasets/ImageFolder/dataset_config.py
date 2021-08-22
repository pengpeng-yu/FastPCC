from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class DatasetConfig(SimpleConfig):
    root: Union[str, Tuple[str, ...]] = 'datasets/coco2017/images'
    filelist_path: Union[str, Tuple[str, ...]] = 'train2017_list.txt'
    file_path_pattern: Union[str, Tuple[str, ...]] = 'train2017/*.jpg'  # works if filelist does not exist

    normalization_scaler: int = 255  # Divide pixels value by scaler. 0 for no normalization.
    channels_order: str = 'BGR'  # BGR or RGB
    target_shapes: Tuple[int, ...] = (1280, 1280)
    resize_strategy: str = 'Expand'  # Shrink, Expand, Retain or Adapt
