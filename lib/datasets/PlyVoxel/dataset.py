import os
import pathlib

import open3d as o3d
import numpy as np
import torch
import torch.utils.data

try:
    import MinkowskiEngine as ME
except ImportError:
    pass

from lib.data_utils import PCData, pc_data_collate_fn
from lib.datasets.PlyVoxel.dataset_config import DatasetConfig


class PlyVoxel(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training: bool, logger):
        super(PlyVoxel, self).__init__()
        if is_training is True and cfg.kd_tree_partition_max_points_num != 0:
            raise NotImplementedError

        def get_collections(x, repeat):
            return x if isinstance(x, tuple) or isinstance(x, list) else (x,) * repeat

        roots = (cfg.root,) if isinstance(cfg.root, str) else cfg.root
        filelist_paths = get_collections(cfg.filelist_path, len(roots))
        file_path_patterns = get_collections(cfg.file_path_pattern, len(roots))
        ori_resolutions = get_collections(cfg.ori_resolution, len(roots))
        resolutions = get_collections(cfg.resolution, len(roots))

        assert all([ori == tgt or tgt == 0 for ori, tgt in zip(ori_resolutions, resolutions)])

        if sum(resolutions) == 0:
            self.voxelized = False
        else:
            self.voxelized = True

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
        try:
            point_cloud = o3d.io.read_point_cloud(file_path)  # colors are normalized by 255!
        except Exception as e:
            print(f'Error when loading {file_path}')
            raise e

        # xyz
        xyz = np.asarray(point_cloud.points, dtype=np.float32)

        if not self.voxelized:
            xyz = xyz / self.file_ori_resolutions[index]

        if self.cfg.with_color:
            color = torch.from_numpy(np.asarray(point_cloud.colors, dtype=np.float32))
            assert np.prod(color.shape) != 0
        else:
            color = None

        if self.cfg.with_normal:
            normal = torch.from_numpy(np.asarray(point_cloud.normals, dtype=np.float32))
            assert np.prod(normal.shape) != 0
        else:
            normal = None

        return PCData(
            xyz=xyz if isinstance(xyz, torch.Tensor) else torch.from_numpy(xyz),
            colors=color,
            normals=normal,
            file_path=file_path if self.cfg.with_file_path else None,
            ori_resolution=self.file_ori_resolutions[index] if self.cfg.with_ori_resolution else None,
            resolution=self.file_resolutions[index] if self.cfg.with_resolution else None
        )

    def collate_fn(self, batch):
        return pc_data_collate_fn(
            batch, sparse_collate=self.voxelized,
            kd_tree_partition_max_points_num=self.cfg.kd_tree_partition_max_points_num
        )


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_color = True
    config.with_normal = False
    config.with_file_path = True

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
