import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data._utils.collate import default_collate
from scipy.spatial.transform import Rotation as R
try:
    import MinkowskiEngine as ME
except ImportError: pass

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

            self.cfg = cfg

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, index):
            # load
            point_cloud = np.loadtxt(self.file_list[index], dtype=np.float32, delimiter=',')

            # sampling
            assert point_cloud.shape[0] >= self.cfg.input_points_num
            if point_cloud.shape[0] > self.cfg.input_points_num:
                if self.cfg.sample_method == 'uniform':
                    uniform_choice = np.random.choice(point_cloud.shape[0], self.cfg.input_points_num, replace=False)
                    point_cloud = point_cloud[uniform_choice]
                else:
                    raise NotImplementedError

            # xyz
            xyz = point_cloud[:, :3]

            # normals
            if self.cfg.with_normal_channel:
                normals = point_cloud[:, 3:]

            # random rotation
            if self.cfg.random_rotation:
                if self.cfg.with_normal_channel: raise NotImplementedError
                xyz = R.random().apply(xyz).astype(np.float32)

            # quantize: ndarray -> torch.Tensor
            if self.cfg.resolution != 0:
                assert self.cfg.resolution > 1
                xyz *= self.cfg.resolution
                if self.cfg.with_normal_channel:
                    xyz, normals = ME.utils.sparse_quantize(xyz, normals)
                else:
                    xyz = ME.utils.sparse_quantize(xyz)

            # classes
            if self.cfg.with_classes:
                cls_idx = self.classes_idx[os.path.split(self.file_list[index])[1].rsplit('_', 1)[0]]

            # return
            if self.cfg.with_normal_channel:
                if self.cfg.with_classes:
                    return xyz, normals, cls_idx
                else:
                    return xyz, normals

            elif not self.cfg.with_normal_channel:
                if self.cfg.with_classes:
                    return xyz, cls_idx
                else:
                    return xyz

        def collate_fn(self, batch):
            if self.cfg.resolution == 0:
                return default_collate(batch)

            elif self.cfg.resolution != 0:
                if isinstance(batch[0], tuple):
                    batch = list(zip(*batch))
                else:
                    batch = (batch, )

                if self.cfg.with_classes:
                    batch_cls = torch.tensor(batch[-1])

                if self.cfg.with_normal_channel:
                    batch_coords, batch_feats = ME.utils.sparse_collate(batch[0], batch[1])
                    if self.cfg.with_classes:
                        return batch_coords, batch_feats, batch_cls
                    else:
                        return batch_coords, batch_feats

                elif not self.cfg.with_normal_channel:
                    batch_coords = [torch.cat((torch.tensor([[batch_idx]], dtype=coord_smaple.dtype)
                                               .expand(coord_smaple.shape[0], -1),
                                               coord_smaple), dim=1)
                                    for batch_idx, coord_smaple in enumerate(batch[0])]
                    batch_coords = torch.cat(batch_coords, dim=0)
                    if self.cfg.with_classes:
                        return batch_coords, batch_cls
                    else:
                        return batch_coords


if __name__ == '__main__':
    config = DatasetConfig()
    config.input_points_num = 10000
    config.with_classes = False
    config.with_normal_channel = False
    config.resolution = 128

    dataset = ModelNetDataset(config, True)
    dataloader = torch.utils.data.DataLoader(dataset, 16, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    print('Done')