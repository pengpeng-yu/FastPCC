import os
import sys
import importlib

import numpy as np
import torch
import torch.utils.data

from lib import torch_utils
from lib.config import Config
from lib import utils

def main():
    # Initialize config
    cfg = Config()
    if len(sys.argv) > 1 and (not '=' in sys.argv[1]) and sys.argv[1].endswith('.yaml'):
        cfg.merge_with_yaml(sys.argv[1])
        cfg.merge_with_dotlist(sys.argv[2:])
    else:
        cfg.merge_with_dotlist(sys.argv[1:])

    os.makedirs('runs', exist_ok=True)
    run_dir = utils.autoindex_obj(os.path.join('runs', cfg.test.rundir_name))
    os.makedirs(run_dir, exist_ok=False)

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.to_yaml())

    test(cfg)


def test(cfg: Config, logger=None, model: torch.nn.Module=None):
    try:
        Dataset = importlib.import_module(cfg.dataset_path).Dataset
    except Exception as e:
        raise ImportError(*e.args)
    dataset: torch.utils.data.Dataset = Dataset(cfg.dataset, True)
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
            Model = importlib.import_module(cfg.model_path).Model
        except Exception as e:
            raise ImportError(*e.args)
        model = Model(cfg.model)
        ckpt_path = utils.autoindex_obj(cfg.test.weights_from_ckpt)
        logger.info(f'loading weights from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
        device = torch.device(cfg.test.device if cfg.test.device == 'cpu' or cfg.test.device.startswith('cuda')
                              else f'cuda:{cfg.test.device}')
        model.to(device)
        model.eval()

    if torch_utils.is_parallel(model):
        model = model.module
    if hasattr(model, 'entropy_bottleneck'):
        model.entropy_bottleneck.update()
    else:
        logger.warning('no entropy_bottleneck was found in model')

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device, non_blocking=True)
        with torch.no_grad():
            model_output = model(data)
            torch.save(data, 'runs/data.pt')
            torch.save(model_output['decoder_output'], 'runs/decoder_output.pt')
            break  # TODO
            # TODO: compute loss and compression rate

    return {'bpp': 0,
            'point2point_loss': 0,
            'point2plane_loss': 0}


if __name__ == '__main__':
    main()
