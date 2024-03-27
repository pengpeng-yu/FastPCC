import os
import os.path as osp
from glob import glob
import hashlib

import numpy as np
import open3d as o3d
import torch
import torch.utils.data

from lib.data_utils import PCData, pc_data_collate_fn, write_ply_file
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
            self.file_list = self.gen_filelist(filelist_abs_path)
        else:
            self.file_list = self.load_filelist(filelist_abs_path)

        self.cache_root = osp.join(
            cfg.root, 'cache',
            hashlib.new(
                'md5',
                f'{filelist_abs_path}'
                f'{cfg.coord_scaler}'.encode('utf-8')
            ).hexdigest()
        )
        self.cached_file_list = [
            _.replace(cfg.root, self.cache_root, 1).replace('.bin', '.ply', 1)
            for _ in self.file_list]

        if osp.isfile(osp.join(
            self.cache_root,
            'train_all_cached' if is_training else 'test_all_cached'
        )):
            logger.info(f'using cache : {self.cache_root}')
            self.file_list = self.cached_file_list
            self.cached_file_list = None
            self.use_cache = True
            self.gen_cache = False
        else:
            os.makedirs(self.cache_root, exist_ok=True)
            with open(osp.join(self.cache_root, 'dataset_config.yaml'), 'w') as f:
                f.write(cfg.to_yaml())
            self.use_cache = False
            self.gen_cache = True

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
            file_list = [line.strip() for line in f]
        return file_list

    @staticmethod
    def get_subset(root, index):
        return glob(osp.join(root, f'{index:02d}/velodyne/*.bin'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        if self.gen_cache:  # TODO: provide original coords in eval
            xyz = np.fromfile(file_path, '<f4').reshape(-1, 4)[:, :3]
            xyz *= self.cfg.coord_scaler
            xyz.round(out=xyz)
            xyz -= xyz.min(0)
            xyz = np.unique(xyz.astype(np.int32), axis=0)
            cache_file_path = self.cached_file_list[index]
            write_ply_file(xyz, cache_file_path, self.cfg.ply_cache_dtype, make_dirs=True)
            return
        else:
            xyz = np.asarray(o3d.io.read_point_cloud(file_path).points)

        return PCData(
            xyz=torch.from_numpy(xyz),
            file_path=file_path
        )

    def collate_fn(self, batch):
        return pc_data_collate_fn(batch, sparse_collate=True)
