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

from lib.datasets.PlyVoxel.dataset_config import DatasetConfig


class PlyVoxel(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(PlyVoxel, self).__init__()
        # only for test purpose
        if is_training is True:
            raise NotImplementedError

        def get_collections(x, repeat):
            return x if isinstance(x, tuple) or isinstance(x, list) else (x,) * repeat

        roots = (cfg.root,) if isinstance(cfg.root, str) else cfg.root
        filelist_paths = get_collections(cfg.filelist_path, len(roots))
        file_path_patterns = get_collections(cfg.file_path_pattern, len(roots))
        ori_resolutions = get_collections(cfg.ori_resolution, len(roots))
        resolutions = get_collections(cfg.resolution, len(roots))

        if not all([ori == tgt for ori, tgt in zip(ori_resolutions, resolutions)]):
            raise NotImplementedError

        # define files list path and cache path
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
        for root, filelist_path, resolution in zip(roots, filelist_paths, resolutions):
            filelist_abs_path = os.path.join(root, filelist_path)
            logger.info(f'using filelist: "{filelist_abs_path}"')
            with open(filelist_abs_path) as f:
                for line in f:
                    line = line.strip()
                    self.file_list.append(os.path.join(root, line))
                    self.file_resolutions.append(resolution)

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
        xyz = torch.from_numpy(np.asarray(point_cloud.points, dtype=np.int32))

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

        if self.cfg.with_file_path:
            rel_file_path = file_path
        else:
            rel_file_path = None

        return {'xyz': xyz,
                'color': color,
                'normal': normal,
                'file_path': rel_file_path,
                'resolution': self.file_resolutions[index] if self.cfg.with_resolution else None}

    def collate_fn(self, batch):
        assert isinstance(batch, list)

        has_file_path = self.cfg.with_file_path
        has_normal = self.cfg.with_normal
        has_color = self.cfg.with_color
        has_resolution = self.cfg.with_resolution

        xyz_list = []
        file_path_list = []
        normal_list = []
        color_list = []
        resolution_list = []

        for sample in batch:
            xyz_list.append(sample['xyz'])
            if has_file_path:
                file_path_list.append(sample['file_path'])
            if has_normal:
                normal_list.append(sample['normal'])
            if has_color:
                color_list.append(sample['color'])
            if has_resolution:
                resolution_list.append(sample['resolution'])

        return_obj = []

        batch_xyz = ME.utils.batched_coordinates(xyz_list)
        return_obj.append(batch_xyz)

        if has_normal:
            return_obj.append(torch.cat(normal_list, dim=0))

        if has_color:
            return_obj.append(torch.cat(color_list, dim=0))

        if has_file_path:
            return_obj.append(file_path_list)

        if has_resolution:
            return_obj.append(torch.tensor(resolution_list, dtype=torch.int32))

        if len(return_obj) == 1:
            return_obj = return_obj[0]
        else:
            return_obj = tuple(return_obj)
        return return_obj


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_color = True
    config.with_normal = False
    config.with_file_path = True

    from loguru import logger

    dataset = PlyVoxel(config, False, logger)

    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample = next(dataloader)

    from main_debug import plt_batch_sparse_coord

    if config.with_color or config.with_normal or config.with_file_path:
        sample_coords = sample[0]
    else:
        sample_coords = sample
    plt_batch_sparse_coord(sample_coords, 0, False)
    plt_batch_sparse_coord(sample_coords, 1, False)
    print('Done')
