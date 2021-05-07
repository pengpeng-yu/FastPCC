from .model_config import ModelConfig as Config
from .model import PointCompressor as Model

__doc__ = 'randla-net-like points cloud compression net' \
          'custom bottleneck' \
          'pooling to one points' \
          'use conv1d to reduce computations of mlp'
