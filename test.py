import os
import os.path as osp
import sys
import pathlib
import importlib
from tqdm import tqdm

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from lib.config import Config
from lib.utils import autoindex_obj
from lib.torch_utils import select_device, unwrap_ddp, load_loose_state_dict
from lib.data_utils import SampleData


def main():
    cfg = Config()
    cfg.merge_with_dotlist(sys.argv[1:])
    cfg.check()

    os.makedirs('runs', exist_ok=True)
    run_dir = pathlib.Path(autoindex_obj(
        osp.join('runs', cfg.test.rundir_name) if not osp.isabs(cfg.test.rundir_name)
        else cfg.test.rundir_name))
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

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    else:
        logger.warning(f'ignore config "float32_matmul_precision={cfg.float32_matmul_precision}" '
                       f'since torch.set_float32_matmul_precision is not available')
    if not cfg.more_reproducible:
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        cudnn.benchmark, cudnn.deterministic = False, True
    print(test(cfg, logger, run_dir))


def test(cfg: Config, logger, run_dir, model: torch.nn.Module = None):
    try:
        Dataset = importlib.import_module(cfg.test.dataset_module_path).Dataset
    except Exception as e:
        raise ImportError(*e.args)

    results_dir = osp.join(run_dir, 'results')
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
        collate_fn=getattr(dataset, 'collate_fn', None)
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
        device, cuda_ids = select_device(logger, local_rank=-1, device=cfg.test.device)
        model = Model(cfg.model)
        if cfg.test.from_ckpt != '':
            ckpt_path = autoindex_obj(cfg.test.from_ckpt)
            logger.info(f'loading weights from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            sd_key = 'ema_state_dict' if 'ema_state_dict' in ckpt else 'state_dict'
            incompatible_keys, missing_keys, unexpected_keys = load_loose_state_dict(model, ckpt[sd_key])
            del ckpt
            logger.info('resumed model_state_dict from checkpoint "{}"'.format(ckpt_path))
            if len(incompatible_keys) != 0:
                logger.warning(f'incompatible keys:\n{incompatible_keys}')
            if len(missing_keys) != 0:
                logger.warning(f'missing keys:\n{missing_keys}')
            if len(unexpected_keys) != 0:
                logger.warning(f'unexpected keys:\n{unexpected_keys}')
            del incompatible_keys, missing_keys, unexpected_keys
        else:
            logger.warning(f'no weight is loaded')
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
        metric_results = unwrap_ddp(model).evaluator.show(results_dir)
    except AttributeError:
        metric_results = {}

    for value in metric_results.values():
        assert isinstance(value, int) or isinstance(value, float)
    logger.info(f'test end')
    return metric_results


if __name__ == '__main__':
    main()
