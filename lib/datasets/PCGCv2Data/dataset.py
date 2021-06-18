import os
import pathlib

import h5py
import numpy as np
import torch
import torch.utils.data

try:
    import MinkowskiEngine as ME
except ImportError: pass

from lib.data_utils import PCData, pc_data_collate_fn
from lib.datasets.PCGCv2Data.dataset_config import DatasetConfig


class PCGCv2Data(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(PCGCv2Data, self).__init__()

        if cfg.resolution != 128:
            raise NotImplementedError

        # define files list path and cache path
        if is_training:
            filelist_abs_path = os.path.join(cfg.root, cfg.train_filelist_path)
        else:
            filelist_abs_path = os.path.join(cfg.root, cfg.test_filelist_path)

        # generate files list
        if not os.path.exists(filelist_abs_path):
            logger.info('no filelist is given. Trying to generate...')
            file_list = tuple(pathlib.Path(cfg.root).glob('*.h5'))

            assert 0 < cfg.train_split_ratio <= 1.0
            if is_training:
                file_list = file_list[: int(cfg.train_split_ratio * len(file_list))]
            else:
                if cfg.train_split_ratio == 1.0: pass
                else: file_list = file_list[int(cfg.train_split_ratio * len(file_list)):]

            with open(filelist_abs_path, 'w') as f:
                f.writelines([str(_.relative_to(cfg.root)) + '\n' for _ in file_list])

        # load files list
        logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            self.file_list = [os.path.join(cfg.root, _.strip()) for _ in f]

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        xyz = h5py.File(file_path)['data'][:, :3]

        return PCData(xyz=torch.from_numpy(xyz.astype(np.int32)),
                      file_path=file_path if self.cfg.with_file_path else None,
                      ori_resolution=128 if self.cfg.with_ori_resolution else None,
                      resolution=self.cfg.resolution if self.cfg.with_resolution else None)

    def collate_fn(self, batch):
        return pc_data_collate_fn(batch, sparse_collate=True)


if __name__ == '__main__':
    config = DatasetConfig()

    from loguru import logger
    dataset = PCGCv2Data(config, True, logger)

    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample = next(dataloader)

    from main_debug import plt_batch_sparse_coord
    sample_coords = sample.xyz
    plt_batch_sparse_coord(sample_coords, 0, False)
    plt_batch_sparse_coord(sample_coords, 1, False)
    print('Done')
