from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple
import importlib

@dataclass
class TrainConfig(SimpleConfig):
    rundir_name: str = 'train_<autoindex>'
    device: str = '0'  # 0 or 0,1,2 or cpu
    more_reproducible: bool = False
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 4
    epochs: int = 100

    optimizer: str = 'sgd'
    learning_rate: float = 0.05
    weight_decay: float = 0.0
    aux_weight_decay: float = 0.0
    lr_step_size: int = 25
    lr_step_gamma: float = 0.3

    resume_from_ckpt: str = ''
    resume_items: Tuple[str] = ('start_epoch', 'state_dict')
    resume_tensorboard: bool = False

    log_frequency: int = 10  # (steps) used for both logging and tensorboard
    ckpt_frequency: int = 2  # (epochs)
    test_frequency: int = 0  # (epochs) 0 means no test in training phase


@dataclass
class TestConfig(SimpleConfig):
    rundir_name: str = 'test_<autoindex>'
    device: str = 'cuda'  # 'cpu' or 'cuda'(only single gpu supported)
    batch_size: int = 1
    num_workers: int = 4
    weights_from_ckpt: str = ''


@dataclass
class DatasetConfig(SimpleConfig):
    class_name: str = 'ModelNetDataset'
    root: str = 'dataset/modelnet40_normal_resampled'
    train_filelist_path: str = 'train_list.txt'
    test_filelist_path: str = 'test_list.txt'
    input_points_num: int = 8192
    sample_method: str = 'uniform'
    with_normal_channel: bool = False


@dataclass
class Config(SimpleConfig):
    model_path: str = 'models.exp0'  # require model_path.ModelConfig and model_path.PointCompressor exist
    model: SimpleConfig = None
    train: TrainConfig = TrainConfig()
    test: TestConfig = TestConfig()
    dataset: DatasetConfig = DatasetConfig()

    def __post_init__(self):
        try:
            self.model = importlib.import_module(self.model_path).ModelConfig()
        except Exception as e:
            raise ImportError(*e.args)
        self.check()

    def check_value(self):
        for item in self.train.resume_items:
            assert item in ('start_epoch', 'state_dict', 'optimizer_state_dict', 'scheduler_state_dict')
        if 'scheduler_state_dict' in self.train.resume_items:
            assert 'start_epoch' in self.train.resume_items
        if hasattr(self.model, 'input_points_num') and hasattr(self.dataset, 'input_points_num'):
            assert self.model.input_points_num == self.dataset.input_points_num
        if self.train.resume_tensorboard:
            assert self.train.resume_from_ckpt != ''
        assert self.train.ckpt_frequency > 0
