import os
import pathlib

import numpy as np
import cv2
import torch.utils.data

from lib.data_utils import IMData, im_data_collate_fn, \
    im_resize_with_crop, im_resize_with_pad, im_pad
from lib.datasets.ImageFolder.dataset_config import DatasetConfig


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger):
        super(ImageFolder, self).__init__()
        assert len(cfg.target_shapes) % 2 == 0 and \
               all([_ > 0 and isinstance(_, int) for _ in cfg.target_shapes])

        def get_collections(x, repeat):
            return x if isinstance(x, tuple) or isinstance(x, list) else (x,) * repeat

        roots = (cfg.root,) if isinstance(cfg.root, str) else cfg.root
        filelist_paths = get_collections(cfg.filelist_path, len(roots))
        file_path_patterns = get_collections(cfg.file_path_pattern, len(roots))

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
        for root, filelist_path in zip(roots, filelist_paths):
            filelist_abs_path = os.path.join(root, filelist_path)
            logger.info(f'using filelist: "{filelist_abs_path}"')
            with open(filelist_abs_path) as f:
                for line in f:
                    line = line.strip()
                    self.file_list.append(os.path.join(root, line))

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        try:
            im: np.ndarray = cv2.imread(file_path).astype(np.float32)
        except Exception as e:
            print(f'Error when loading {file_path}')
            raise e

        if self.cfg.normalization_scaler > 0:
            im = im / self.cfg.normalization_scaler

        if self.cfg.channels_order == 'RGB':
            im = im[:, :, ::-1]
        elif self.cfg.channels_order != 'BGR':
            raise NotImplementedError

        return IMData(im=im, file_path=file_path)

    def collate_fn(self, batch):
        return im_data_collate_fn(
            batch,
            target_shapes=self.cfg.target_shapes,
            resize_strategy=self.cfg.resize_strategy,
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
