import os
import os.path as osp
from glob import glob
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.utils.data

from lib.data_utils import PCData, pc_data_collate_fn, kd_tree_partition_randomly
from lib.data_utils import o3d_coords_sampled_from_triangle_mesh, normalize_coords
from lib.morton_code import morton_encode_magicbits
from lib.datasets.ShapeNetCorev2.dataset_config import DatasetConfig


class ShapeNetCorev2(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(ShapeNetCorev2, self).__init__()
        assert cfg.resolution > 1
        self.is_training = is_training

        # define files list path and cache path
        if is_training:
            filelist_abs_path = osp.join(cfg.root, cfg.train_filelist_path)
            official_divisions = cfg.train_divisions
        else:
            filelist_abs_path = osp.join(cfg.root, cfg.test_filelist_path)
            official_divisions = cfg.test_divisions
        if isinstance(official_divisions, str):
            official_divisions = (official_divisions,)

        # generate files list
        if not osp.exists(filelist_abs_path):
            logger.info('no filelist is given. Trying to generate...')
            if 'all' not in official_divisions:
                file_list = []
                with open(osp.join(cfg.root, cfg.shapenet_all_csv)) as f:
                    f.readline()
                    for line in f:
                        _, synset_id, _, model_id, split = line.strip().split(',')
                        file_path = osp.join(synset_id, model_id, 'models', 'model_normalized.obj')
                        if osp.exists(osp.join(cfg.root, file_path)):
                            if split in official_divisions:
                                file_list.append(file_path)
            else:
                file_list = (_[len(cfg.root)+1:] for _ in glob(f'{cfg.root}/*/*/*/*.obj'))
            with open(filelist_abs_path, 'w') as f:
                for _ in file_list:
                    # 7edb40d76dff7455c2ff7551a4114669 seems to be problematic
                    if osp.split(osp.split(_)[0])[0].endswith('7edb40d76dff7455c2ff7551a4114669'):
                        continue
                    f.write(_)
                    f.write('\n')

        # load files list
        self.file_list = []
        logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            for line in f:
                line = line.strip()
                self.file_list.append(osp.join(cfg.root, line))

        if cfg.generate_cache:
            self.cache_root = osp.join(
                cfg.root, 'cache',
                hashlib.new(
                    'md5',
                    f'{filelist_abs_path} '
                    f'{cfg.mesh_sample_points_num} '
                    f'{cfg.mesh_sample_point_method} '
                    f'{cfg.mesh_sample_point_resolution} '
                    f'{cfg.ply_cache_dtype} '.encode('utf-8')
                ).hexdigest()
            )
            self.cached_file_list = [
                _.replace(cfg.root, self.cache_root, 1).replace('.obj', '.npz', 1)
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
        else:
            self.cached_file_list = None
            self.use_cache = self.gen_cache = False

        logger.info(f'filelist[0]: {self.file_list[0]}')
        logger.info(f'filelist[1]: {self.file_list[1]}')
        logger.info(f'length of filelist: {len(self.file_list)}')
        self.cfg = cfg
        self.logger = logger

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        if self.use_cache:
            xyz = np.load(file_path)['xyz'].astype(np.float64, copy=False)
        else:
            xyz = o3d_coords_sampled_from_triangle_mesh(
                file_path,
                self.cfg.mesh_sample_points_num,
                sample_method=self.cfg.mesh_sample_point_method,
            )
            normalize_coords(xyz)
            xyz *= self.cfg.mesh_sample_point_resolution
            if self.gen_cache:
                xyz = xyz.astype(self.cfg.ply_cache_dtype, copy=False)
                xyz = np.unique(xyz, axis=0)
                cache_file_path = self.cached_file_list[index]
                os.makedirs(osp.dirname(cache_file_path), exist_ok=True)
                np.savez_compressed(cache_file_path, xyz=xyz)
                return
        resolution = self.cfg.mesh_sample_point_resolution

        if self.cfg.random_rotation:
            xyz = R.random().apply(xyz)
            xyz -= xyz.min(0)

        if self.cfg.resolution != resolution:
            xyz *= self.cfg.resolution / resolution
        xyz = np.unique(xyz.astype(np.int32), axis=0)

        if self.is_training:
            par_num = self.cfg.kd_tree_partition_max_points_num
            if par_num != 0 and xyz.shape[0] > par_num:
                par_num = self.cfg.kd_tree_partition_max_points_num
                xyz = kd_tree_partition_randomly(xyz, par_num)
                xyz -= xyz.min(0)

            if self.cfg.random_offset != 0:
                xyz += np.random.randint(0, self.cfg.random_offset, 3, dtype=np.int32)

        xyz = torch.from_numpy(xyz)
        if self.cfg.morton_sort:
            xyz = xyz[torch.argsort(morton_encode_magicbits(xyz, inverse=self.cfg.morton_sort_inverse))]

        return PCData(
            xyz=xyz,
            file_path=file_path
        )

    def collate_fn(self, batch):
        return pc_data_collate_fn(batch)
