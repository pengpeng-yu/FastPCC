import os
import sys
import importlib

import numpy as np
import torch
import torch.utils.data

from lib import torch_utils
from lib.config import Config
from lib import utils
from lib import dataset as dataset_module

def main():
    # Initialize config
    cfg = Config()
    if len(sys.argv) > 1 and (not '=' in sys.argv[1]) and sys.argv[1].endswith('.yaml'):
        cfg.merge_with_yaml(sys.argv[1])
        cfg.merge_with_dotlist(sys.argv[2:])
    else:
        cfg.merge_with_dotlist(sys.argv[1:])

    os.makedirs('runs', exist_ok=True)
    run_dir = utils.auto_index_dir('runs', cfg.test.rundir_name)
    os.makedirs(run_dir, exist_ok=False)

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.to_yaml())

    test(cfg)


def test(cfg: Config, logger=None, model: torch.nn.Module=None):
    dataset: torch.utils.data.Dataset = getattr(dataset_module, cfg.dataset.class_name)(cfg.dataset, False)
    dataloader = torch.utils.data.DataLoader(dataset, cfg.test.batch_size, shuffle=False,
                                             num_workers=cfg.test.num_workers, drop_last=False)
    if model is not None:
        model.eval()
        current_device = next(model.parameters()).device
        cfg_device = cfg.test.device
        if current_device.type != cfg_device:
            logger.warning(f'you require using {cfg_device} in testing while using {current_device.type} in training, '
                           f'which means validation during training will ignore the setting of test.device')
        device = current_device
    else:
        from loguru import logger
        try:
            PointCompressor = importlib.import_module(cfg.model_path).PointCompressor
        except Exception as e:
            raise ImportError(*e.args)
        model: torch.nn.Module = PointCompressor(cfg.model)
        model.load_state_dict(torch.load(cfg.test.weights_from_ckpt)['state_dict'])
        device = torch.device(cfg.test.device)
        model.to(device)
        model.eval()

    if torch_utils.is_parallel(model):
        model = model.module
    model.entropy_bottleneck.update()

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device, non_blocking=True)
        with torch.no_grad():
            test_res = model(data)
            break  # TODO
            # TODO: compute loss and compression rate

    return {'bpp': 0,
            'point2point_loss': 0,
            'point2plane_loss': 0}


if __name__ == '__main__':
    main()
