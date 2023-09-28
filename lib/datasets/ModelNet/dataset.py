import os
import pathlib

import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation as R

from lib.data_utils import PCData, pc_data_collate_fn
from lib.datasets.ModelNet.dataset_config import DatasetConfig
from lib.data_utils import o3d_coords_sampled_from_triangle_mesh, normalize_coords


class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(ModelNetDataset, self).__init__()
        filelist_abs_path = os.path.join(
            cfg.root, cfg.train_filelist_path if is_training else cfg.test_filelist_path)

        # generate files list
        if not os.path.exists(filelist_abs_path):
            logger.info('no filelist is given. Trying to generate...')
            if is_training:
                file_list = pathlib.Path(cfg.root).glob('*/train/*.off')
            else:
                file_list = pathlib.Path(cfg.root).glob('*/test/*.off')
            with open(filelist_abs_path, 'w') as f:
                f.write('\n'.join([str(_.relative_to(cfg.root)) for _ in file_list]))

        # load files list
        logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            self.file_list = [os.path.join(cfg.root, f.readline().strip())]
            self.data_file_format = os.path.splitext(self.file_list[0])[1]
            try:
                assert self.data_file_format in ['.off', '.txt']
                for line in f:
                    line = line.strip()
                    assert os.path.splitext(line)[1] == self.data_file_format
                    self.file_list.append(os.path.join(cfg.root, line))
                if is_training: assert len(self.file_list) == 9843
                else: assert len(self.file_list) == 2468
            except AssertionError as e:
                logger.info('wrong number or format of files.')
                raise e

        # load classes indices
        if cfg.with_class:
            with open(os.path.join(cfg.root, cfg.classes_names)) as f:
                classes_names = f.readlines()
            self.classes_idx = {l.strip(): cls_idx for cls_idx, l in enumerate(classes_names)}

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]

        # for modelnet40_normal_resampled
        if file_path.endswith('.txt'):
            point_cloud = np.loadtxt(file_path, dtype=np.float64, delimiter=',')
            assert point_cloud.shape[0] >= self.cfg.input_points_num
            if point_cloud.shape[0] > self.cfg.input_points_num:
                if self.cfg.sample_method == 'uniform':
                    uniform_choice = np.random.choice(point_cloud.shape[0], self.cfg.input_points_num, replace=False)
                    point_cloud = point_cloud[uniform_choice]
                else:
                    raise NotImplementedError
            xyz = point_cloud[:, :3]
        # for original modelnet dataset
        elif file_path.endswith('.off'):
            xyz = o3d_coords_sampled_from_triangle_mesh(
                file_path,
                self.cfg.input_points_num,
                sample_method=self.cfg.mesh_sample_point_method
            )[0]
        else:
            raise NotImplementedError

        if self.cfg.random_rotation:
            xyz = R.random().apply(xyz)

        normalize_coords(xyz)

        if self.cfg.resolution != 0:
            assert self.cfg.resolution > 1
            xyz *= self.cfg.resolution
            xyz = xyz.astype(np.int32)
            xyz = np.unique(xyz, axis=0)

        # classes
        if self.cfg.with_class:
            cls_idx = self.classes_idx[os.path.split(self.file_list[index])[1].rsplit('_', 1)[0]]
        else:
            cls_idx = None

        return PCData(
            xyz=torch.from_numpy(xyz),
            class_idx=cls_idx,
            file_path=file_path
        )

    def collate_fn(self, batch):
        return pc_data_collate_fn(batch, sparse_collate=self.cfg.resolution != 0)


if __name__ == '__main__':
    config = DatasetConfig()
    config.input_points_num = 200000
    config.with_class = False
    config.resolution = 128
    config.root = 'datasets/modelnet40_manually_aligned'

    from loguru import logger
    dataset = ModelNetDataset(config, True, logger)
    dataloader = torch.utils.data.DataLoader(dataset, 16, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample: PCData = next(dataloader)

    from lib.vis import plt_draw_xyz, plt_batched_sparse_xyz
    batched_xyz = sample.xyz
    if config.resolution == 0:
        plt_draw_xyz(batched_xyz[0])
        plt_draw_xyz(batched_xyz[1])
    else:
        plt_batched_sparse_xyz(batched_xyz, 0, True)
        plt_batched_sparse_xyz(batched_xyz, 1, True)
    print('Done')
