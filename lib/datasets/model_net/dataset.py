import os
import numpy as np
import torch
import torch.utils.data
from .dataset_config import DatasetConfig


class ModelNetDataset(torch.utils.data.Dataset):
        def __init__(self, cfg: DatasetConfig, is_training):
            super(ModelNetDataset, self).__init__()
            self.cfg = cfg
            if is_training:
                filelist_path = os.path.join(cfg.root, cfg.train_filelist_path)
            else:
                filelist_path = os.path.join(cfg.root, cfg.test_filelist_path)
            with open(filelist_path) as f:
                self.file_list = [os.path.join(cfg.root, i.strip()) for i in f.readlines()]

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, index):
            point_cloud = np.loadtxt(self.file_list[index], dtype=np.float32, delimiter=',')
            if self.cfg.sample_method == 'uniform':
                point_cloud = point_cloud[: self.cfg.input_points_num]
            else:
                raise NotImplementedError

            if not self.cfg.with_normal_channel:
                point_cloud = point_cloud[:, :3]

            return point_cloud

        def collate_fn(self, list_data):
            #
            pass

