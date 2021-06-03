import os
import pathlib

import open3d as o3d
import numpy as np
import torch
import torch.utils.data

try:
    import MinkowskiEngine as ME
except ImportError: pass

from lib.datasets.VFB8iv2.dataset_config import DatasetConfig


class VFB8iv2(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(VFB8iv2, self).__init__()
        # only for test purpose
        assert is_training is False

        if not cfg.resolution == 1024:
            raise NotImplementedError

        # define files list path and cache path
        filelist_abs_path = os.path.join(cfg.root, cfg.filelist_path)

        # generate files list
        if not os.path.exists(filelist_abs_path):
            logger.info('no filelist is given. Trying to generate...')
            root_path = pathlib.Path(cfg.root)
            file_list = []
            for class_name in ['longdress', 'loot', 'redandblack', 'soldier']:
                file_list.extend(root_path.glob(class_name + '/Ply/*.ply'))
            with open(filelist_abs_path, 'w') as f:
                f.write('\n'.join([str(_.relative_to(root_path)) for _ in file_list]))

        # load files list
        logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            self.file_list = [os.path.join(cfg.root, _.strip()) for _ in f]

        if len(self.file_list) != 1200:
            logger.warning(f'wrong number of files. 1202 expected, got {len(self.file_list)}')

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load
        file_path = self.file_list[index]
        point_cloud = o3d.io.read_point_cloud(file_path)

        # xyz
        xyz = np.asarray(point_cloud.points, dtype=np.int32)

        if self.cfg.with_color:
            color = np.asarray(point_cloud.colors, dtype=np.int32)
        else:
            color = None

        return {'xyz': xyz,
                'color': color,
                'file_path': os.path.relpath(file_path, self.cfg.root) if self.cfg.with_file_path else None}

    def collate_fn(self, batch):
        assert isinstance(batch, list)

        has_file_path = self.cfg.with_file_path

        xyz_list = []
        file_path_list = [] if has_file_path else None

        for sample in batch:
            if isinstance(sample['xyz'], torch.Tensor):
                xyz_list.append(sample['xyz'])
            else:
                xyz_list.append(torch.from_numpy(sample['xyz']))
            if has_file_path:
                file_path_list.append(sample['file_path'])

        return_obj = []

        batch_xyz = ME.utils.batched_coordinates(xyz_list)
        return_obj.append(batch_xyz)

        if has_file_path:
            return_obj.append(file_path_list)

        if len(return_obj) == 1:
            return_obj = return_obj[0]
        else:
            return_obj = tuple(return_obj)
        return return_obj


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_color = False

    from loguru import logger
    dataset = VFB8iv2(config, False, logger)

    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample = next(dataloader)

    from main_debug import plt_batch_sparse_coord
    if config.with_color or config.with_file_path:
        sample_coords = sample[0]
    else:
        sample_coords = sample
    plt_batch_sparse_coord(sample_coords, 0, False)
    plt_batch_sparse_coord(sample_coords, 1, False)
    print('Done')
