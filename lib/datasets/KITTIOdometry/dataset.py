import os.path as osp
import pathlib

import numpy as np
import open3d as o3d
import torch
import torch.utils.data

from lib.data_utils import PCData, pc_data_collate_fn, write_ply_file, kd_tree_partition_randomly
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
            ply_file_abs_path = osp.join(cfg.ply_file_root, cfg.ply_file_train_filelist_path)
        else:
            filelist_abs_path = osp.join(cfg.root, cfg.test_filelist_path)
            ply_file_abs_path = osp.join(cfg.ply_file_root, cfg.ply_file_test_filelist_path)

        if not osp.exists(filelist_abs_path):
            self.file_list = [osp.join(cfg.root, _) for _ in self.gen_filelist(filelist_abs_path)]
        else:
            self.file_list = self.load_filelist(cfg.root, filelist_abs_path)

        if osp.exists(ply_file_abs_path):
            self.file_list.extend(self.load_filelist(cfg.ply_file_root, ply_file_abs_path))

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

    def load_filelist(self, root, filelist_abs_path):
        self.logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            file_list = [osp.join(root, line.strip()) for line in f.readlines()[::self.cfg.list_sampling_interval]]
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
        flag_kitti_bin_file = file_path.endswith('bin')
        if flag_kitti_bin_file:
            xyz = np.fromfile(file_path, '<f4').reshape(-1, 4)[:, :3]
        else:
            assert file_path.endswith('ply')
            xyz = o3d.t.io.read_point_cloud(file_path).point.positions.numpy().astype('<f4')
        org_points_num = xyz.shape[0]

        # For calculating distortion metrics
        if not self.is_training and flag_kitti_bin_file:
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
        if flag_kitti_bin_file:
            scale = (self.cfg.resolution - 1) / 400
            inv_scale = 400 / (self.cfg.resolution - 1)
        else:
            scale = self.cfg.ply_file_coord_scaler
            inv_scale = 1 / self.cfg.ply_file_coord_scaler
        xyz *= scale
        xyz = xyz.round().astype(np.int32)
        xyz = np.unique(xyz, axis=0)

        if self.is_training:
            par_num = self.cfg.kd_tree_partition_max_points_num
            if par_num != 0 and xyz.shape[0] > par_num:
                xyz = kd_tree_partition_randomly(xyz, par_num)
                tmp_org_point = xyz.min(0)
                xyz -= tmp_org_point
                org_point += tmp_org_point

            if self.cfg.random_flip:
                if np.random.rand() > 0.5:
                    xyz[:, 0] = -xyz[:, 0] + xyz[:, 0].max()
                if np.random.rand() > 0.5:
                    xyz[:, 1] = -xyz[:, 1] + xyz[:, 1].max()

        xyz = torch.from_numpy(xyz)
        if self.cfg.morton_sort:
            xyz = xyz[torch.argsort(morton_encode_magicbits(xyz, inverse=self.cfg.morton_sort_inverse))]

        inv_trans = torch.from_numpy(np.concatenate((org_point.reshape(-1), (inv_scale,)), 0, dtype=np.float32))
        if flag_kitti_bin_file and not self.cfg.flag_sparsepcgc:
            resolution = 59.70 + 1
        elif flag_kitti_bin_file and self.cfg.flag_sparsepcgc:
            resolution = 30000 + 1
            inv_trans *= 1000
        else:
            resolution = self.cfg.ply_file_resolution
        return PCData(
            xyz=xyz,
            file_path=cache_path if (not self.is_training and flag_kitti_bin_file) else file_path,
            org_points_num=org_points_num,
            resolution=resolution,  # For the peak value in pc_error
            inv_transform=inv_trans
        )

    def collate_fn(self, batch):
        return pc_data_collate_fn(batch, self.cfg.kd_tree_partition_max_points_num)
