import os.path as osp
import pathlib

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.utils.data

from lib.data_utils import PCData, pc_data_collate_fn, kd_tree_partition_randomly
from lib.morton_code import morton_encode_magicbits
from lib.datasets.PlyVoxel.Base.dataset_config import DatasetConfig


class PlyVoxel(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training: bool, logger):
        super(PlyVoxel, self).__init__()
        self.is_training = is_training

        def get_collections(x, repeat):
            if isinstance(x, tuple) or isinstance(x, list):
                assert len(x) == repeat, \
                    f'Unexpected length ({len(x)}) of a dataset config item,' \
                    f'which is expected to have a length of {repeat}.'
                return x
            else:
                return (x,) * repeat

        roots = (cfg.root,) if isinstance(cfg.root, str) else cfg.root
        filelist_paths = get_collections(cfg.filelist_path, len(roots))
        file_path_patterns = get_collections(cfg.file_path_pattern, len(roots))
        resolutions = get_collections(cfg.resolution, len(roots))
        test_time_coord_scaler = get_collections(cfg.test_time_coord_scaler, len(roots))
        partition_max_points_num = get_collections(cfg.kd_tree_partition_max_points_num, len(roots))

        # define files list path
        for root, filelist_path, file_path_pattern in zip(roots, filelist_paths, file_path_patterns):
            filelist_abs_path = osp.join(root, filelist_path)
            # generate files list
            if not osp.exists(filelist_abs_path):
                logger.warning(f'no filelist of {root} is given. Trying to generate using {file_path_pattern}...')
                file_list = pathlib.Path(root).glob(file_path_pattern)
                with open(filelist_abs_path, 'w') as f:
                    f.write('\n'.join([str(_.relative_to(root)) for _ in file_list]))

        # load files list
        self.file_list = []
        self.file_resolutions = []
        self.file_coord_scaler_list = []
        self.file_partition_max_points_num_list = []
        for root, filelist_path, resolution, scaler, par_num in \
                zip(roots, filelist_paths, resolutions, test_time_coord_scaler, partition_max_points_num):
            filelist_abs_path = osp.join(root, filelist_path)
            logger.info(f'using filelist: "{filelist_abs_path}"')
            with open(filelist_abs_path) as f:
                for line in f:
                    line = line.strip()
                    self.file_list.append(osp.join(root, line))
                    self.file_resolutions.append(resolution)
                    self.file_coord_scaler_list.append(scaler)
                    self.file_partition_max_points_num_list.append(par_num)

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load
        file_path = self.file_list[index]
        try:
            point_cloud = o3d.io.read_point_cloud(file_path)  # colors are normalized by 255!
        except Exception as e:
            print(f'Error when loading {file_path}')
            raise e

        # xyz
        xyz = np.asarray(point_cloud.points)

        if self.is_training and self.cfg.random_rotation:
            xyz = R.random().apply(xyz)
            xyz -= xyz.min(0)

        coord_scaler = self.file_coord_scaler_list[index]
        if not self.is_training and coord_scaler != 1:
            xyz = (xyz * coord_scaler).round()
        xyz = xyz.astype(np.int32)

        if self.cfg.with_color:
            color = np.asarray(point_cloud.colors)
            assert color.size != 0
            color *= 255
            np.round(color, out=color)
            color = color.astype(np.float32)
        else:
            color = None

        par_num = self.file_partition_max_points_num_list[index]
        if self.is_training:
            if par_num != 0 and xyz.shape[0] > par_num:
                xyz, (color,) = kd_tree_partition_randomly(
                    xyz, par_num, (color,)
                )
                xyz -= xyz.min(0)

            if self.cfg.random_flip:
                if np.random.rand() > 0.5:
                    xyz[:, 0] = -xyz[:, 0] + xyz[:, 0].max()
                if np.random.rand() > 0.5:
                    xyz[:, 1] = -xyz[:, 1] + xyz[:, 1].max()
                if np.random.rand() > 0.5:
                    xyz[:, 2] = -xyz[:, 2] + xyz[:, 2].max()

            if self.cfg.random_rgb_offset != 0:
                color += np.random.randint(
                    -self.cfg.random_rgb_offset,
                    self.cfg.random_rgb_offset + 1, (1, 3))
                color = color.clip(0, 255)

            if self.cfg.random_rgb_perm:
                color = np.ascontiguousarray(color[:, torch.randperm(3).tolist()])

        xyz = torch.from_numpy(xyz)
        color = torch.from_numpy(color) if color is not None else None
        if self.cfg.morton_sort:
            order = torch.argsort(morton_encode_magicbits(xyz, inverse=self.cfg.morton_sort_inverse))
            xyz = xyz[order]
            if color is not None:
                color = color[order]

        return PCData(
            xyz=xyz,
            color=color,
            file_path=file_path,
            class_idx=par_num,  # I use class_idx to transmit par_num to collate_fn
            resolution=None if self.is_training else self.file_resolutions[index],
            inv_transform=torch.tensor((0, 0, 0, 1 / coord_scaler), dtype=torch.float32) if coord_scaler != 1 else None
        )

    def collate_fn(self, batch):
        return pc_data_collate_fn(
            batch,
            kd_tree_partition_max_points_num=batch[0].class_idx
            if not self.is_training else 0  # Test-time partitioning
        )


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_color = True

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
