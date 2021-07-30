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

from lib.data_utils import binvox_rw, PCData, pc_data_collate_fn
from lib.datasets.ShapeNetCorev2.dataset_config import DatasetConfig
from lib.data_utils import o3d_coords_from_triangle_mesh, normalize_coords


class ShapeNetCorev2(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(ShapeNetCorev2, self).__init__()

        if cfg.data_format in ['.solid.binvox', '.surface.binvox'] or \
                cfg.data_format == ['.solid.binvox', '.surface.binvox'] or \
                cfg.data_format == ['.surface.binvox', '.solid.binvox']:
            if cfg.resolution != 128 and cfg.resolution != 0:
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
                    assert len(self.file_list) == 35765 - 80 - 1  # 80 have no binvox files
                else:
                    assert len(self.file_list) == 10266 - 13  # 13 has no binvox files
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
        file_path = self.file_list[index]

        if self.cfg.data_format != '.obj':
            with open(file_path, 'rb') as f:
                xyz = binvox_rw.read_as_coord_array(f).data.astype(np.int32).T
                xyz = np.ascontiguousarray(xyz)
        else:
            xyz = o3d_coords_from_triangle_mesh(file_path,
                                                self.cfg.mesh_sample_points_num,
                                                self.cfg.mesh_sample_point_method)

        if self.cfg.random_rotation:
            xyz = R.random().apply(xyz).astype(np.float32)

        xyz = normalize_coords(xyz)

        if self.cfg.resolution != 0:
            assert self.cfg.resolution > 1
            xyz *= self.cfg.resolution
            xyz = ME.utils.sparse_quantize(xyz)

        return PCData(
            xyz=xyz if isinstance(xyz, torch.Tensor) else torch.from_numpy(xyz),
            file_path=file_path if self.cfg.with_file_path else None,
            ori_resolution=None if
            not self.cfg.with_ori_resolution or self.cfg.data_format == '.obj'
            else 128,
            resolution=self.cfg.resolution if self.cfg.with_resolution else None)

    def collate_fn(self, batch):
        return pc_data_collate_fn(batch, sparse_collate=self.cfg.resolution != 0)


if __name__ == '__main__':
    config = DatasetConfig()
    # config.data_format = '.obj'
    config.mesh_sample_points_num = 5000000

    from loguru import logger
    dataset = ShapeNetCorev2(config, True, logger)

    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample: PCData = next(dataloader)

    from lib.vis import plt_draw_xyz, plt_batched_sparse_xyz
    batched_xyz = sample.xyz
    if config.resolution == 0:
        plt_draw_xyz(batched_xyz[0])
        plt_draw_xyz(batched_xyz[1])
    else:
        plt_batched_sparse_xyz(batched_xyz, 0, True)
        plt_batched_sparse_xyz(batched_xyz, 1, True)
    print('Done')
