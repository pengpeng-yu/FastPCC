import os.path as osp
import pathlib

import numpy as np
import torch
import torch.utils.data

from lib.data_utils import PCData, pc_data_collate_fn, write_ply_file
from lib.morton_code import morton_encode_magicbits
from lib.datasets.KITTIOdometry.dataset_config import DatasetConfig


class KITTIOdometry(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(KITTIOdometry, self).__init__()
        self.cfg = cfg
        self.is_training = is_training
        self.logger = logger

        if is_training:
            filelist_abs_path = osp.join(cfg.root, cfg.train_filelist_path)
        else:
            filelist_abs_path = osp.join(cfg.root, cfg.test_filelist_path)

        if not osp.exists(filelist_abs_path):
            self.file_list = [osp.join(self.cfg.root, _) for _ in self.gen_filelist(filelist_abs_path)]
        else:
            self.file_list = self.load_filelist(filelist_abs_path)

    def gen_filelist(self, filelist_abs_path):
        self.logger.info('no filelist is given. Trying to generate...')
        file_list = []
        if self.is_training:
            for subset_idx in self.cfg.train_subset_index:
                file_list.extend(self.get_subset(self.cfg.root, subset_idx))
        else:
            for subset_idx in self.cfg.test_subset_index:
                file_list.extend(self.get_subset(self.cfg.root, subset_idx))
        with open(filelist_abs_path, 'w') as f:
            f.writelines((_ + '\n' for _ in file_list))
        return file_list

    def load_filelist(self, filelist_abs_path):
        self.logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            file_list = [osp.join(self.cfg.root, line.strip()) for line in f.readlines()[::self.cfg.list_sampling_interval]]
        return file_list

    @staticmethod
    def get_subset(root, index):
        ls = [str(_.relative_to(root)) for _ in pathlib.Path(root).glob(f'{index:02d}/velodyne/*.bin')]
        ls.sort()
        return ls

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        xyz = np.fromfile(file_path, '<f4').reshape(-1, 4)[:, :3]
        org_points_num = xyz.shape[0]

        # For calculating distortion metrics
        if not self.is_training:
            p, n = osp.split(file_path)
            if not self.cfg.flag_sparsepcgc:
                cache_path = osp.join(p, n.replace('.bin', '_n.ply'))
                if not osp.isfile(cache_path):
                    write_ply_file(xyz, cache_path, estimate_normals=True)
            else:
                cache_path = osp.join(p, n.replace('.bin', '_q1mm_n.ply'))
                org_xyz = np.unique((xyz * 1000).round(), axis=0)
                if not osp.isfile(cache_path):
                    write_ply_file(org_xyz, cache_path, estimate_normals=True)

        org_point = xyz.min(0)
        xyz -= org_point
        scale = 400 / (self.cfg.resolution - 1)
        xyz /= scale
        xyz.round(out=xyz)

        if self.cfg.random_flip:
            if np.random.rand() > 0.5:
                xyz[:, 0] = -xyz[:, 0] + xyz[:, 0].max()
            if np.random.rand() > 0.5:
                xyz[:, 1] = -xyz[:, 1] + xyz[:, 1].max()

        if not self.cfg.morton_sort:
            xyz = np.unique(xyz.astype(np.int32), axis=0)
            xyz = torch.from_numpy(xyz)
        else:
            xyz = xyz.astype(np.int32)
            xyz = torch.from_numpy(xyz)
            xyz = xyz[torch.argsort(morton_encode_magicbits(xyz, inverse=self.cfg.morton_sort_inverse))]
            xyz = torch.unique_consecutive(xyz, dim=0)

        resolution = 59.70 + 1
        inv_trans = torch.from_numpy(np.concatenate((org_point.reshape(-1), (scale,)), 0, dtype=np.float32))
        if self.cfg.flag_sparsepcgc:
            resolution = 30000 + 1
            inv_trans *= 1000
        return PCData(
            xyz=xyz,
            file_path=cache_path if not self.is_training else file_path,
            org_points_num=org_points_num,
            resolution=resolution,  # For the peak value in pc_error
            inv_transform=inv_trans
        )

    def collate_fn(self, batch):
        return pc_data_collate_fn(batch)
