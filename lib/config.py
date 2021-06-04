import os
import yaml

from lib.simple_config import SimpleConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainConfig(SimpleConfig):
    rundir_name: str = 'train_<autoindex>'
    device: str = '0'  # 0 or 0,1,2 or cpu
    more_reproducible: bool = False
    amp: bool = True
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 4
    epochs: int = 100

    optimizer: str = 'sgd'
    learning_rate: float = 0.05
    momentum: float = 0.9
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

    def merge_setattr(self, key, value):
        if key == 'resume_items':
            if 'all' in value:
                value = ('start_epoch', 'state_dict', 'optimizer_state_dict', 'scheduler_state_dict')
        super().merge_setattr(key, value)

    def check_local_value(self):
        if 'scheduler_state_dict' in self.resume_items:
            assert 'start_epoch' in self.resume_items
        all_resume_items = ('start_epoch', 'state_dict', 'optimizer_state_dict', 'scheduler_state_dict')
        for item in self.resume_items:
            assert item in all_resume_items
        if self.resume_tensorboard:
            assert self.resume_from_ckpt != ''
        assert self.ckpt_frequency > 0


@dataclass
class TestConfig(SimpleConfig):
    rundir_name: str = 'test_<autoindex>'
    device: str = 'cuda:3'  # 'cpu' or 'cuda'(only single gpu supported)
    batch_size: int = 8
    num_workers: int = 4
    weights_from_ckpt: str = ''
    log_frequency: int = 50  # (steps) used for logging
    save_results: bool = False  # save outputs of model in runs/rundir_name/results


@dataclass
class Config(SimpleConfig):
    model_path: str = 'models.exp1'  # model_path.Config and model_path.Model are required
    model: SimpleConfig = None
    train: TrainConfig = TrainConfig()
    test: TestConfig = TestConfig()
    dataset_path: str = 'lib.datasets.ModelNet'  # dataset_path.Dataset and dataset_path.Config are required
    dataset: SimpleConfig = None

    def __post_init__(self):
        self.auto_import()
        self.check()

    def check_local_value(self):
        if hasattr(self.model, 'input_points_num') and hasattr(self.dataset, 'input_points_num'):
            assert self.model.input_points_num == self.dataset.input_points_num

        if hasattr(self.model, 'resolution') and hasattr(self.dataset, 'resolution'):
            if self.model.resolution != self.dataset.resolution:
                print(f'Warning: model.resolution({self.model.resolution}) != '
                      f'dataset.resolution({self.dataset.resolution}) !')

    def merge_with_yaml(self, yaml_path):
        yaml_dict = yaml.safe_load(open(yaml_path))
        if 'model_path' in yaml_dict:
            if not yaml_dict['model_path'].rsplit('.', 1)[-1] in os.path.splitext(os.path.basename(yaml_path))[0]:
                print(f'Warning: loading configuration from {os.path.basename(yaml_path)} '
                      f'with model_path == {yaml_dict["model_path"]}. Is this correct?')
        return self.merge_with_dict(yaml_dict)