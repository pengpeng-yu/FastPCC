import os
import pathlib

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.utils.data

try:
    import MinkowskiEngine as ME
except ImportError: pass

from lib.data_utils import binvox_rw
from lib.datasets.ShapeNetCorev2.dataset_config import DatasetConfig
from lib.data_utils import o3d_coords_from_triangle_mesh, normalize_coords


class ShapeNetCorev2(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(ShapeNetCorev2, self).__init__()

        if cfg.data_format in ['.solid.binvox', '.surface.binvox'] or \
                cfg.data_format == ['.solid.binvox', '.surface.binvox'] or \
                cfg.data_format == ['.surface.binvox', '.solid.binvox']:
            if cfg.resolution != 128:
                raise NotImplementedError
        elif cfg.data_format != '.obj':
            raise NotImplementedError

        # define files list path and cache path
        if is_training:
            filelist_abs_path = os.path.join(cfg.root, cfg.train_filelist_path)
        else:
            filelist_abs_path = os.path.join(cfg.root, cfg.test_filelist_path)

        # generate files list
        if not os.path.exists(filelist_abs_path):
            logger.info('no filelist is given. Trying to generate...')

            file_list = []
            with open(os.path.join(cfg.root, cfg.shapenet_all_csv)) as f:
                f.readline()
                for line in f:
                    _, synset_id, _, model_id, split = line.strip().split(',')
                    file_paths = [os.path.join(synset_id, model_id, 'models', 'model_normalized' + d_format)
                                  for d_format in ([cfg.data_format]
                                  if isinstance(cfg.data_format, str) else cfg.data_format)
                                  if model_id != '7edb40d76dff7455c2ff7551a4114669']
                    # 7edb40d76dff7455c2ff7551a4114669 seems to be problematic

                    for file_path in file_paths:
                        if os.path.exists(os.path.join(cfg.root, file_path)):
                            if (is_training and split == 'train') or \
                                    not is_training and split == 'test':
                                file_list.append(file_path)

            with open(filelist_abs_path, 'w') as f:
                f.writelines([_ + '\n' for _ in file_list])

        # load files list
        self.file_list = []
        logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            for line in f:
                line = line.strip()
                ext_name = '.' + '.'.join(os.path.split(line)[1].rsplit('.', 2)[1:])
                assert ext_name == cfg.data_format or ext_name in cfg.data_format, \
                    f'"{line}" in "{filelist_abs_path}" is inconsistent with ' \
                    f'data format "{cfg.data_format}" in config'
                self.file_list.append(os.path.join(cfg.root, line))
        logger.info(f'filelist[0]: {self.file_list[0]}')
        logger.info(f'filelist[1]: {self.file_list[1]}')
        logger.info(f'length of filelist: {len(self.file_list)}')

        try:
            if cfg.data_format == '.surface.binvox':
                if is_training:
                    assert len(self.file_list) == 35765 - 80 - 2  # 80 have no binvox files
                else:
                    assert len(self.file_list) == 10266 - 13 - 2  # 13 has no binvox files
            elif cfg.data_format == '.solid.binvox':
                pass
            elif cfg.data_format == '.obj':
                pass
        except AssertionError as e:
            logger.info('wrong number of files.')
            raise e

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load
        file_path = self.file_list[index]

        if self.cfg.data_format != '.obj':
            with open(file_path, 'rb') as f:
                xyz = binvox_rw.read_as_coord_array(f).data.astype(np.int32).T
                xyz = np.ascontiguousarray(xyz)
        else:
            xyz = o3d_coords_from_triangle_mesh(file_path,
                                                self.cfg.points_num,
                                                self.cfg.mesh_sample_point_method)

        if self.cfg.random_rotation:
            xyz = R.random().apply(xyz).astype(np.float32)

        xyz = normalize_coords(xyz)

        if self.cfg.resolution != 0:
            assert self.cfg.resolution > 1
            xyz *= self.cfg.resolution
            xyz = ME.utils.sparse_quantize(xyz)

        return_obj = {'xyz': xyz,
                      'file_path': file_path if self.cfg.with_file_path else None}
        return return_obj

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

        if self.cfg.resolution != 0:
            batch_xyz = ME.utils.batched_coordinates(xyz_list)
        else:
            batch_xyz = torch.stack(xyz_list, dim=0)
        return_obj.append(batch_xyz)

        if has_file_path:
            return_obj.append(file_path_list)

        if self.cfg.with_resolution:
            return_obj.append(self.cfg.resolution)

        if len(return_obj) == 1:
            return_obj = return_obj[0]
        else:
            return_obj = tuple(return_obj)
        return return_obj


if __name__ == '__main__':
    config = DatasetConfig()
    config.data_format = '.obj'
    config.points_num = 5000000

    from loguru import logger
    dataset = ShapeNetCorev2(config, True, logger)

    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample = next(dataloader)

    from main_debug import plt_batch_sparse_coord
    if config.with_file_path or config.with_resolution:
        sample_coords = sample[0]
    else:
        sample_coords = sample
    plt_batch_sparse_coord(sample_coords, 0, False)
    plt_batch_sparse_coord(sample_coords, 1, False)
    print('Done')
