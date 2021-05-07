import os
import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation as R

# from lib.points_layers import RandLANeighborFea, RotationInvariantDistFea
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

            # if cfg.precompute is None:
            #     self.neighbor_fea_generator = None
            # elif cfg.precompute == 'RotationInvariantDistFea':
            #     self.neighbor_fea_generator = RotationInvariantDistFea(cfg.neighbor_num, cfg.anchors_points)
            # elif cfg.precompute == 'RandLANeighborFea':
            #     self.neighbor_fea_generator = RandLANeighborFea(cfg.anchors_points)
            # else:
            #     raise NotImplementedError

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

        # def collate_fn(self, list_data):
        #     if isinstance(list_data[0], tuple):
        #         list_data = zip(*list_data)
        #         data = [torch.tensor(np.stack(d, axis=0)) for d in list_data]
        #     else:
        #         data = [np.stack(list_data, axis=0)]
        #
        #     if self.cfg.with_normal_channel:
        #         xyz = data[0][:, :, :3]
        #     else:
        #         xyz = data[0]
        #
        #     if self.cfg.precompute is None:
        #         pass
        #
        #     elif self.cfg.precompute == 'RotationInvariantDistFea' and self.cfg.model_sample_method == 'uniform':
        #         points_num = xyz.shape[1]
        #         raw_neighbor_feature_list = []
        #         neighbors_idx_list = []
        #         for rate in self.cfg.model_sample_rates:
        #             sampled_xyz = xyz[:, :int(points_num * rate)]
        #             raw_neighbor_feature, neighbors_idx = self.neighbor_fea_generator(sampled_xyz)
        #             raw_neighbor_feature_list.append(raw_neighbor_feature)
        #             neighbors_idx_list.append(neighbors_idx)
        #         data.append(raw_neighbor_feature_list)
        #         data.append(neighbors_idx_list)
        #
        #     else:
        #         raise NotImplementedError
        #
        #     return data


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_classes = True
    config.random_rotation = True

    dataset = ModelNetDataset(config, True)
    dataloader = torch.utils.data.DataLoader(dataset, 16, shuffle=False)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    print('Done')