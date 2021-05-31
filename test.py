import os
import sys
import pathlib
import importlib
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data

from lib import torch_utils
from lib.config import Config
from lib import utils


def main():
    # Initialize config
    cfg = Config()
    if len(sys.argv) > 1 and ('=' not in sys.argv[1]) and sys.argv[1].endswith('.yaml'):
        cfg.merge_with_yaml(sys.argv[1])
        cfg.merge_with_dotlist(sys.argv[2:])
    else:
        cfg.merge_with_dotlist(sys.argv[1:])

    os.makedirs('runs', exist_ok=True)
    run_dir = pathlib.Path(utils.autoindex_obj(os.path.join('runs', cfg.test.rundir_name)))
    os.makedirs(run_dir, exist_ok=False)

    with open(run_dir / 'config.yaml', 'w') as f:
        f.write(cfg.to_yaml())

    from loguru import logger
    logger.remove()
    loguru_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | ' \
                    '<level>{level: <8}</level> | ' \
                    '<cyan>{name}</cyan>:<cyan>{line}</cyan>  ' \
                    '<level>{message}</level>'
    logger.add(sys.stderr, colorize=True, format=loguru_format, level='DEBUG')
    logger.add(run_dir / 'log.txt', format=loguru_format, level=0, mode='w')

    print(test(cfg, logger))


def test(cfg: Config, logger, model: torch.nn.Module = None):
    try:
        Dataset = importlib.import_module(cfg.dataset_path).Dataset
    except Exception as e:
        raise ImportError(*e.args)

    # cache
    dataset: torch.utils.data.Dataset = Dataset(cfg.dataset, False, logger)
    if hasattr(dataset, 'gen_cache') and dataset.gen_cache is True:
        datacache_loader = torch.utils.data.DataLoader(dataset, cfg.test.batch_size,
                                                       num_workers=cfg.test.num_workers * 2, drop_last=False,
                                                       collate_fn=lambda batch: None)
        for _ in tqdm(datacache_loader):
            pass
        with open(os.path.join(dataset.cache_root, 'test_all_cached'), 'w') as f:
            pass
        logger.info('finish caching')
        # rebuild dataset to use cache
        dataset: torch.utils.data.Dataset = Dataset(cfg.dataset, False, logger)

    dataloader = torch.utils.data.DataLoader(dataset, cfg.test.batch_size, shuffle=False,
                                             num_workers=cfg.test.num_workers, drop_last=False, pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    if model is not None:
        model.eval()
        current_device = next(model.parameters()).device
        logger.info(f'start testing using device {current_device}')
        # cfg_device = cfg.test.device
        # if current_device.type != cfg_device:
        #     logger.warning(f'validation during training will ignore the setting of test.device')
        device = current_device
    else:
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
        torch.cuda.set_device(device)
        model.eval()

    if hasattr(model, 'entropy_bottleneck'):
        model.entropy_bottleneck.update()
    elif torch_utils.is_parallel(model) and hasattr(model.module, 'entropy_bottleneck'):
        model.module.entropy_bottleneck.update()
    else:
        logger.warning('no entropy_bottleneck was found in model')

    try:
        if torch_utils.is_parallel(model):
            model.module.log_pred_res('reset')
        else:
            model.log_pred_res('reset')
    except AttributeError: pass

    steps_one_epoch = len(dataloader)
    for step_idx, batch_data in enumerate(dataloader):
        if isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.to(device, non_blocking=True)
        elif isinstance(batch_data, list) or isinstance(batch_data, tuple):
            batch_data = [d.to(device, non_blocking=True) for d in batch_data]
        else:
            raise NotImplementedError

        with torch.no_grad():
            test_output = model(batch_data)
            # torch.save(data, 'runs/data.pt')
            # torch.save(model_output['decoder_output'], 'runs/decoder_output.pt')
            if cfg.test.log_frequency > 0 and (step_idx == 0 or (step_idx + 1) % cfg.test.log_frequency == 0):
                logger.info(f'test step {step_idx}/{steps_one_epoch - 1}')

    try:
        if torch_utils.is_parallel(model):
            test_results = model.module.log_pred_res('show')
        else:
            test_results = model.log_pred_res('show')
    except AttributeError:
        test_results = {}

    logger.info(f'test end')
    return {item_name: item for item_name, item in test_results.items()
            if isinstance(item, int) or isinstance(item, float)}


if __name__ == '__main__':
    main()
