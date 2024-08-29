import math
import os.path as osp
import pathlib

import numpy as np
import cv2
import torch.utils.data

from lib.data_utils import IMData, im_data_collate_fn
from lib.datasets.ImageFolder.dataset_config import DatasetConfig


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(ImageFolder, self).__init__()

        def get_collections(x, repeat):
            return x if isinstance(x, tuple) or isinstance(x, list) else (x,) * repeat

        roots = (cfg.root,) if isinstance(cfg.root, str) else cfg.root
        filelist_paths = get_collections(cfg.filelist_path, len(roots))
        file_path_patterns = get_collections(cfg.file_path_pattern, len(roots))

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
        for root, filelist_path in zip(roots, filelist_paths):
            filelist_abs_path = osp.join(root, filelist_path)
            logger.info(f'using filelist: "{filelist_abs_path}"')
            with open(filelist_abs_path) as f:
                for line in f:
                    line = line.strip()
                    self.file_list.append(osp.join(root, line))

        self.cfg = cfg
        self.is_training = is_training

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        try:
            im: np.ndarray = cv2.imread(file_path).astype(np.float32)
        except Exception as e:
            print(f'Error when loading {file_path}')
            raise e

        if self.cfg.channels_order == 'RGB':
            im = im[:, :, ::-1]
        elif self.cfg.channels_order != 'BGR':
            raise NotImplementedError

        if self.is_training and self.cfg.random_h_flip and np.random.rand() > 0.5:
            im = im[:, ::-1]

        return IMData(im=im, file_path=file_path)

    def collate_fn(self, batch):
        if self.is_training:
            target_shape = self.cfg.target_shape_for_training
        else:
            target_shape = math.ceil(batch[0].im.shape[0] / self.cfg.stride_for_test) * self.cfg.stride_for_test, \
                math.ceil(batch[0].im.shape[1] / self.cfg.stride_for_test) * self.cfg.stride_for_test
        return im_data_collate_fn(
            batch,
            target_shape=target_shape,
            channel_last_to_channel_first=True
        )


if __name__ == '__main__':
    config = DatasetConfig()

    from loguru import logger
    dataset = ImageFolder(config, False, logger)
    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    print('Done')
