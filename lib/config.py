from lib.simple_config import SimpleConfig
from dataclasses import dataclass, field
from typing import Tuple, Union

int_or_seq = Union[int, Tuple[int, ...]]
float_or_seq = Union[float, Tuple[float, ...]]
str_or_seq = Union[str, Tuple[str, ...]]


@dataclass
class TrainConfig(SimpleConfig):
    rundir_name: str = 'train_<autoindex>'
    device: Union[int, str] = '0'  # 0 or 0,1,2 or cpu
    find_unused_parameters: bool = False
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    epochs: int = 100
    pin_memory: bool = True
    bucket_cap_mb: Union[int, float] = 0

    ema: bool = False
    ema_decay: float = 0.9999
    ema_warmup: bool = False
    ema_warmup_gamma: float = 1.0
    ema_warmup_power: float = 3/4
    ema_foreach: bool = True
    amp: bool = False

    optimizer: str_or_seq = ('SGD', 'SGD')
    learning_rate: float_or_seq = 0.05
    momentum: float_or_seq = 0.9
    weight_decay: float_or_seq = 0.0
    max_grad_norm: float_or_seq = 0.0
    grad_acc_steps: int = 1

    scheduler: str_or_seq = 'Step'
    lr_step_size: int_or_seq = 25
    lr_step_gamma: float_or_seq = 0.3

    from_ckpt: str = ''
    resume_items: Tuple[str, ...] = ('state_dict',)

    tensorboard_port: int = 6006
    log_frequency: int = 20  # (steps) used for both logging and tensorboard
    ckpt_frequency: int = 2  # (epochs)
    test_frequency: int = 0  # (epochs) 0 means no test in training phase
    cuda_empty_cache_frequency: int = 0  # (steps)

    dataset_module_path: str = ''  # Classes dataset_module_path.Dataset and dataset_module_path.Config are required
    dataset: SimpleConfig = None

    def __post_init__(self):
        if self.dataset_module_path != '':
            self.local_auto_import()

    def merge_setattr(self, key, value):
        if key == 'resume_items':
            if 'all' in value:
                value = ('state_dict', 'optimizer_state_dict', 'scheduler_state_dict')
        super().merge_setattr(key, value)

    def check_local_value(self):
        all_resume_items = ('state_dict', 'optimizer_state_dict', 'scheduler_state_dict')
        for item in self.resume_items:
            assert item in all_resume_items
        assert self.ckpt_frequency > 0
        if isinstance(self.optimizer, str):
            self.optimizer = (self.optimizer,)

        for key in ['learning_rate',
                    'momentum',
                    'weight_decay',
                    'max_grad_norm',
                    'scheduler',
                    'lr_step_size',
                    'lr_step_gamma']:
            if isinstance(getattr(self, key), tuple) or isinstance(getattr(self, key), list):
                assert len(getattr(self, key)) == len(self.optimizer), \
                    f'length of cfg.{key} is not consistent with length of cfg.optimizer\n' \
                    f'cfg.{key}: {getattr(self, key)}\n' \
                    f'self.optimizer: {self.optimizer}\n'
            else:
                setattr(self, key, (getattr(self, key),) * len(self.optimizer))


@dataclass
class TestConfig(SimpleConfig):
    rundir_name: str = 'test_<autoindex>'
    device: Union[int, str] = '0'  # 0 or 1 or cpu (only single gpu supported)
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    from_ckpt: str = ''
    log_frequency: int = 1  # (steps) used for logging

    dataset_module_path: str = ''
    dataset: SimpleConfig = None

    def __post_init__(self):
        if self.dataset_module_path != '':
            self.local_auto_import()


@dataclass
class Config(SimpleConfig):
    model_module_path: str = ''  # Classes model_module_path.Config and model_module_path.Model are required
    model: SimpleConfig = None

    float32_matmul_precision: str = 'high'  # or highest or medium
    more_reproducible: bool = False

    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)

    def __post_init__(self):
        if self.model_module_path != '':
            self.local_auto_import()
