import os
import pathlib

import numpy as np
import torch
import torch.utils.data

try:
    import MinkowskiEngine as ME
except ImportError: pass

from lib.datasets.MVUB.dataset_config import DatasetConfig


class MVUB(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(MVUB, self).__init__()
        # only for test purpose
        assert is_training is False

        # define files list path and cache path
        filelist_abs_path = os.path.join(cfg.root, cfg.filelist_path)

        # generate files list
        if not os.path.exists(filelist_abs_path):
            logger.info('no filelist is given. Trying to generate...')
            file_list = pathlib.Path(cfg.root).glob('*/ply/*.ply')
            with open(filelist_abs_path, 'w') as f:
                f.write('\n'.join([str(_.relative_to(cfg.root)) for _ in file_list]))

        # load files list
        logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            self.file_list = [os.path.join(cfg.root, _.strip()) for _ in f]
            try:
                assert len(self.file_list) == 1202
            except AssertionError as e:
                logger.info('wrong number of files.')
                raise e

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load
        file_path = self.file_list[index]
        with open(file_path) as f:
            f.readline()
            f.readline()
            vertex_num = int(f.readline().strip().split(' ')[-1])
            point_cloud = np.loadtxt(f, dtype=np.int32, skiprows=7)
            assert point_cloud.shape[0] == vertex_num, 'wrong number of points'

        # xyz
        xyz = point_cloud[:, :3]

        if self.cfg.with_color:
            color = point_cloud[:, 3: 6]

        if self.cfg.with_color:
            return xyz, color
        else:
            return xyz

    def collate_fn(self, batch):
        if isinstance(batch[0], tuple):
            batch = list(zip(*batch))
        else:
            batch = (batch, )

        if self.cfg.with_color:
            batch_coords, batch_feats = ME.utils.sparse_collate(batch[0], batch[1])
            return batch_coords, batch_feats

        elif not self.cfg.with_color:
            batch_coords = ME.utils.batched_coordinates(batch[0])
            return batch_coords


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_color = True

    from loguru import logger
    dataset = MVUB(config, False, logger)

    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample = next(dataloader)

    from main_debug import plt_batch_sparse_coord
    if config.with_color:
        sample_coords = sample[0]
    else:
        sample_coords = sample
    plt_batch_sparse_coord(sample_coords, 0, False)
    plt_batch_sparse_coord(sample_coords, 1, False)
    print('Done')
