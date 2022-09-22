import os
import pathlib

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.utils.data

try:
    import MinkowskiEngine as ME
except ImportError:
    pass

from lib.data_utils import PCData, pc_data_collate_fn, kd_tree_partition_randomly
from lib.datasets.PlyVoxel.Base.dataset_config import DatasetConfig


class PlyVoxel(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training: bool, logger):
        super(PlyVoxel, self).__init__()
        self.is_training = is_training

        def get_collections(x, repeat):
            return x if isinstance(x, tuple) or isinstance(x, list) else (x,) * repeat

        roots = (cfg.root,) if isinstance(cfg.root, str) else cfg.root
        filelist_paths = get_collections(cfg.filelist_path, len(roots))
        file_path_patterns = get_collections(cfg.file_path_pattern, len(roots))
        ori_resolutions = get_collections(cfg.ori_resolution, len(roots))
        resolutions = get_collections(cfg.resolution, len(roots))

        self.voxelized = all([r != 0 for r in resolutions])
        if not self.voxelized:
            assert all([r == 0 for r in resolutions])

        # define files list path
        for root, filelist_path, file_path_pattern in zip(roots, filelist_paths, file_path_patterns):
            filelist_abs_path = os.path.join(root, filelist_path)
            # generate files list
            if not os.path.exists(filelist_abs_path):
                logger.info(f'no filelist of {root} is given. Trying to generate using {file_path_pattern}...')
                file_list = pathlib.Path(root).glob(file_path_pattern)
                with open(filelist_abs_path, 'w') as f:
                    f.write('\n'.join([str(_.relative_to(root)) for _ in file_list]))

        # load files list
        self.file_list = []
        self.file_resolutions = []
        self.file_ori_resolutions = []
        for root, filelist_path, ori_resolution, resolution in zip(roots, filelist_paths, ori_resolutions, resolutions):
            filelist_abs_path = os.path.join(root, filelist_path)
            logger.info(f'using filelist: "{filelist_abs_path}"')
            with open(filelist_abs_path) as f:
                for line in f:
                    line = line.strip()
                    self.file_list.append(os.path.join(root, line))
                    self.file_resolutions.append(resolution)
                    self.file_ori_resolutions.append(ori_resolution)

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load
        file_path = self.file_list[index]
        ori_resolution = self.file_ori_resolutions[index]
        resolution = self.file_resolutions[index]
        try:
            point_cloud = o3d.io.read_point_cloud(file_path)  # colors are normalized by 255!
        except Exception as e:
            print(f'Error when loading {file_path}')
            raise e

        # xyz
        xyz = np.asarray(point_cloud.points)

        if self.cfg.random_rotation:
            xyz = R.random().apply(xyz)
            xyz -= xyz.min(0)

        if not self.voxelized:
            xyz /= (ori_resolution - 1)
        else:
            if resolution != ori_resolution:
                xyz *= ((resolution - 1) / (ori_resolution - 1))
            xyz = np.round(xyz)

        if self.cfg.with_color:
            color = np.asarray(point_cloud.colors).astype(np.float32)
            color *= 255
            assert np.prod(color.shape) != 0
        else:
            color = None

        if self.cfg.with_normal:
            normal = np.asarray(point_cloud.normals).astype(np.float32)
            assert np.prod(normal.shape) != 0
        else:
            normal = None

        if self.is_training and self.cfg.kd_tree_partition_max_points_num != 0:
            xyz, (color, normal) = kd_tree_partition_randomly(
                xyz,
                self.cfg.kd_tree_partition_max_points_num,
                (color, normal)
            )
            resolution = int(np.ceil((xyz.max(0) - xyz.min(0)).max()).item()) + 1

        return PCData(
            xyz=torch.from_numpy(xyz),
            color=torch.from_numpy(color) if color is not None else None,
            normal=torch.from_numpy(normal) if normal is not None else None,
            file_path=file_path,
            ori_resolution=ori_resolution,
            resolution=resolution
        )

    def collate_fn(self, batch):
        return pc_data_collate_fn(
            batch, sparse_collate=self.voxelized,
            kd_tree_partition_max_points_num=self.cfg.kd_tree_partition_max_points_num
            if not self.is_training else 0
        )


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_color = True
    config.with_normal = False

    from loguru import logger
    dataset = PlyVoxel(config, False, logger)
    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample: PCData = next(dataloader)

    from lib.vis import plt_batched_sparse_xyz
    sample_coords = sample.xyz
    plt_batched_sparse_xyz(sample_coords, 0, False)
    plt_batched_sparse_xyz(sample_coords, 1, False)
    print('Done')
