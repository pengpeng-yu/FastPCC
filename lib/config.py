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

    optimizer: str = 'SGD'
    aux_optimizer: str = 'SGD'
    learning_rate: float = 0.05
    aux_learning_rate: float = 0.05
    momentum: float = 0.9
    aux_momentum: float = 0.9
    weight_decay: float = 0.0
    aux_weight_decay: float = 0.0
    max_grad_norm: float = 0.0  # 0.0 to close. aux_param excluded.

    scheduler: str = 'Step'  # Step or OneCycle
    # StepLR
    lr_step_size: int = 25
    lr_step_gamma: float = 0.3
    # OneCycleLR
    lr_pct_start: float = 0.3
    lr_init_div_factor: float = 10.
    lr_final_div_factor: float = 1000.

    resume_from_ckpt: str = ''
    resume_items: Tuple[str, ...] = ('state_dict',)
    resume_tensorboard: bool = False

    log_frequency: int = 10  # (steps) used for both logging and tensorboard
    ckpt_frequency: int = 2  # (epochs)
    test_frequency: int = 0  # (epochs) 0 means no test in training phase

    dataset_path: str = 'lib.datasets.ModelNet'  # dataset_path.Dataset and dataset_path.Config are required
    dataset: SimpleConfig = None

    def __post_init__(self):
        self.local_auto_import()

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
    device: str = '3'  # 0 or 1 or cpu (only single gpu supported)
    batch_size: int = 8
    num_workers: int = 4
    weights_from_ckpt: str = ''
    log_frequency: int = 50  # (steps) used for logging
    save_results: bool = False  # save outputs of model in runs/rundir_name/results

    # you can keep this empty to use the same dataloader class with the one in training during testing
    # this feature is defined in test.py
    dataset_path: str = ''
    dataset: SimpleConfig = SimpleConfig()

    def __post_init__(self):
        if self.dataset_path != '':
            self.local_auto_import()


@dataclass
class Config(SimpleConfig):
    model_path: str = 'models.convolutional.baseline'  # model_path.Config and model_path.Model are required
    model: SimpleConfig = None
    train: TrainConfig = TrainConfig()
    test: TestConfig = TestConfig()

    def __post_init__(self):
        self.local_auto_import()
        self.check()

    def check_local_value(self):
        if hasattr(self.model, 'input_points_num') and hasattr(self.train.dataset, 'input_points_num'):
            assert self.model.input_points_num == self.train.dataset.input_points_num
            if hasattr(self.test.dataset, 'input_points_num'):
                assert self.model.input_points_num == self.test.dataset.input_points_num
