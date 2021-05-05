import os
import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation as R
from lib.datasets.modelnet.dataset_config import DatasetConfig


class ModelNetDataset(torch.utils.data.Dataset):
        def __init__(self, cfg: DatasetConfig, is_training):
            super(ModelNetDataset, self).__init__()
            if is_training:
                filelist_path = os.path.join(cfg.root, cfg.train_filelist_path)
            else:
                filelist_path = os.path.join(cfg.root, cfg.test_filelist_path)
            with open(filelist_path) as f:
                self.file_list = [os.path.join(cfg.root, i.strip()) for i in f.readlines()]
            if cfg.with_classes:
                with open(os.path.join(cfg.root, cfg.classes_names)) as f:
                    classes_names = f.readlines()
                self.classes_idx = {l.strip(): cls_idx for cls_idx, l in enumerate(classes_names)}
            self.cfg = cfg

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, index):
            # load
            point_cloud = np.loadtxt(self.file_list[index], dtype=np.float32, delimiter=',')

            # sampling
            if self.cfg.sample_method == 'uniform':
                assert point_cloud.shape[0] >= self.cfg.input_points_num
                point_cloud = point_cloud[: self.cfg.input_points_num]
            else:
                raise NotImplementedError

            # normals
            if not self.cfg.with_normal_channel:
                point_cloud = point_cloud[:, :3]

            # random rotation
            if self.cfg.random_rotation:
                if self.cfg.with_normal_channel: raise NotImplementedError
                point_cloud = R.random().apply(point_cloud).astype(np.float32)

            # classes
            if self.cfg.with_classes:
                cls_idx = self.classes_idx[os.path.split(self.file_list[index])[1].rsplit('_', 1)[0]]
                return point_cloud, cls_idx
            else:
                return point_cloud

        def collate_fn(self, list_data):
            #
            pass


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_classes = True
    config.random_rotation = True
    dataset = ModelNetDataset(config, True)
    dataloader = torch.utils.data.DataLoader(dataset, 16, shuffle=True)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    print('Done')