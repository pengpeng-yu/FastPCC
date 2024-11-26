import os
import os.path as osp
import sys
import pathlib
import importlib
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data

from lib.config import Config
from lib.utils import autoindex_obj
from lib.torch_utils import select_device, is_parallel
from lib.data_utils import SampleData


def main():
    # Initialize config
    cfg = Config()
    cfg.merge_with_dotlist(sys.argv[1:])
    cfg.check()

    os.makedirs('runs', exist_ok=True)
    run_dir = pathlib.Path(autoindex_obj(osp.join('runs', cfg.test.rundir_name)))
    os.makedirs(run_dir, exist_ok=False)
    with open(run_dir / 'config.yaml', 'w') as f:
        f.write(cfg.to_yaml())
    from loguru import logger
    logger.remove()
    loguru_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green>' \
                    ' <level>{level: <4}</level>' \
                    ' <cyan>{name}</cyan>:<cyan>{line}</cyan>' \
                    ' <level>{message}</level>'
    logger.add(sys.stderr, colorize=True, format=loguru_format, level='DEBUG')
    logger.add(run_dir / 'log.txt', format=loguru_format, level=0, mode='w')
    print(test(cfg, logger, run_dir))


def test(cfg: Config, logger, run_dir, model: torch.nn.Module = None):
    try:
        Dataset = importlib.import_module(cfg.test.dataset_module_path).Dataset
    except Exception as e:
        raise ImportError(*e.args)

    results_dir = osp.join(run_dir, 'results') if cfg.test.save_results else None
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)

    # cache
    dataset: torch.utils.data.Dataset = Dataset(cfg.test.dataset, False, logger)
    if hasattr(dataset, 'gen_cache') and dataset.gen_cache is True:
        datacache_loader = torch.utils.data.DataLoader(
            dataset, cfg.test.batch_size,
            num_workers=cfg.test.num_workers * 2, drop_last=False,
            collate_fn=lambda batch: None
        )
        for _ in tqdm(datacache_loader):
            pass
        with open(osp.join(dataset.cache_root, 'test_all_cached'), 'w') as f:
            pass
        logger.info('finish caching')
        # rebuild dataset to use cache
        dataset: torch.utils.data.Dataset = Dataset(cfg.test.dataset, False, logger)

    dataloader = torch.utils.data.DataLoader(
        dataset, cfg.test.batch_size, shuffle=False,
        num_workers=cfg.test.num_workers, drop_last=False, pin_memory=cfg.test.pin_memory,
        collate_fn=dataset.collate_fn
    )
    if model is not None:
        if model.training:
            model.eval()
        device = next(model.parameters()).device
    else:
        try:
            Model = importlib.import_module(cfg.model_module_path).Model
        except Exception as e:
            raise ImportError(*e.args)
        model = Model(cfg.model)
        if cfg.test.from_ckpt != '':
            ckpt_path = autoindex_obj(cfg.test.from_ckpt)
            logger.info(f'loading weights from {ckpt_path}')
            try:
                incompatible_keys = model.load_state_dict(
                    torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
            except RuntimeError as e:
                logger.error('error when loading model_state_dict')
                raise e
            logger.info('resumed weights in checkpoint "{}"'.format(ckpt_path))
            if incompatible_keys[0] != [] or incompatible_keys[1] != []:
                logger.warning(incompatible_keys)
        else:
            logger.warning(f'no weight is loaded')
        device, cuda_ids = select_device(
            logger,
            local_rank=-1,
            device=cfg.test.device
        )
        model.to(device)
        model.eval()

    logger.info(f'start testing using device {device}')
    steps_one_epoch = len(dataloader)
    for step_idx, batch_data in enumerate(dataloader):
        if isinstance(batch_data, torch.Tensor):
            batch_data = [batch_data.to(device, non_blocking=True), results_dir]
        elif isinstance(batch_data, list) or isinstance(batch_data, tuple):
            batch_data = [d.to(device, non_blocking=True) if isinstance(d, torch.Tensor) else d for d in batch_data] \
                         + [results_dir]
        elif isinstance(batch_data, dict):
            batch_data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_data.items()}
            batch_data.update({'results_dir': results_dir})
        elif isinstance(batch_data, SampleData):
            batch_data.to(device=device, non_blocking=True)
            batch_data.results_dir = results_dir
        else:
            raise NotImplementedError

        with torch.no_grad():
            batch_out = model(batch_data)

        if cfg.test.log_frequency > 0 and (step_idx == 0 or (step_idx + 1) % cfg.test.log_frequency == 0):
            logger.info(f'test step {step_idx}/{steps_one_epoch - 1}')

    try:
        if is_parallel(model):
            metric_results = model.module.evaluator.show(results_dir)
        else:
            metric_results = model.evaluator.show(results_dir)
    except AttributeError:
        metric_results = {}

    for value in metric_results.values():
        assert isinstance(value, int) or isinstance(value, float)
    logger.info(f'test end')
    return metric_results


if __name__ == '__main__':
    main()
