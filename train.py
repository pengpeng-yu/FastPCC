import os
import shutil
import sys
import importlib
import time
import pathlib
from tqdm import tqdm
from functools import partial
from typing import Dict, Union, List, Callable

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import tensorboard
from tensorboard import program

from test import test
from lib.config import Config
from lib import utils
from lib import torch_utils
from lib.data_utils import SampleData


def main():
    # DDP arg
    for arg_idx, arg in enumerate(sys.argv):
        if '--local_rank' in arg:
            local_rank = int(arg[len('--local_rank='):])
            sys.argv.pop(arg_idx)
            break
    else:
        local_rank = -1

    # Initialize config, run dir, logger
    cfg = Config()
    cfg.merge_with_dotlist(sys.argv[1:])
    cfg.check()

    from loguru import logger
    logger.remove()

    if local_rank in (-1, 0):
        loguru_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> |' \
                        ' <level>{level: <8}</level> |' \
                        ' <cyan>{name}</cyan>:<cyan>{line}</cyan>  ' \
                        '<level>{message}</level>'
        logger.add(sys.stderr, colorize=True, format=loguru_format, level='DEBUG')

        os.makedirs('runs', exist_ok=True)
        run_dir = pathlib.Path(utils.autoindex_obj(os.path.join('runs', cfg.train.rundir_name)))
        ckpts_dir = run_dir / 'ckpts'
        utils.make_new_dirs(run_dir, logger)
        utils.make_new_dirs(ckpts_dir, logger)

        logger.add(run_dir / 'log.txt', format=loguru_format, level=0, mode='w')
        logger.info('preparing for training...')
        with open(run_dir / 'config.yaml', 'w') as f:
            f.write(cfg.to_yaml())

        # Tensorboard
        tb_logdir = run_dir / 'tb_logdir'
        utils.make_new_dirs(tb_logdir, logger)
        if cfg.train.resume_tensorboard:
            try:
                last_tb_dir = pathlib.Path(cfg.train.resume_from_ckpt).parent.parent / 'tb_logdir'
                for log_file in os.listdir(last_tb_dir):
                    shutil.copy(last_tb_dir / log_file, tb_logdir)
            except Exception as e:
                e.args = (*e.args, 'Error when copying tensorboard log')
                raise e
            else:
                logger.info(f'resumed tensorboard log file(s) in {last_tb_dir}')

        tb_writer = SummaryWriter(str(tb_logdir))
        tb_program = program.TensorBoard()
        try:
            tb_program.configure(argv=[None, '--logdir', str(tb_logdir)])
            tb_url = tb_program.launch()
        except Exception as e:
            logger.error(f'fail to launch Tensorboard')
            raise e
        logger.info(f'TensorBoard {tensorboard.__version__} at {tb_url}')

        train(cfg, local_rank, logger, tb_writer, run_dir, ckpts_dir)
        
    else:
        train(cfg, local_rank, logger)


def train(cfg: Config, local_rank, logger, tb_writer=None, run_dir=None, ckpts_dir=None):
    # Parallel training
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    device = torch_utils.select_device(logger, local_rank, cfg.train.device, cfg.train.batch_size)

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        assert cfg.train.batch_size % world_size == 0, '--batch-size must be multiple of CUDA device count'
        process_batch_size = cfg.train.batch_size // world_size

    else:
        process_batch_size = cfg.train.batch_size
    logger.info(f'world_size: {world_size}, global_rank: {global_rank}, local_rank: {local_rank}')

    # Initialize random number generator (RNG) seeds
    if not cfg.train.more_reproducible:
        np.random.seed(global_rank + 2)
        torch_utils.init_torch_seeds(global_rank + 2)
    else:
        np.random.seed(0)
        torch_utils.init_torch_seeds(0)

    # Initialize model
    try:
        Model = importlib.import_module(cfg.model_path).Model
    except Exception as e:
        raise ImportError(*e.args)

    model = Model(cfg.model)

    if hasattr(model, 'params_divider'):
        params_divider = model.params_divider
    else:
        if len(cfg.train.optimizer) == 2:
            params_divider: Callable[[str], int] = lambda s: 0 if not s.endswith("aux_param") else 1
        else:
            assert len(cfg.train.optimizer) == 1
            params_divider: Callable[[str], int] = lambda s: 0

    if device.type != 'cpu' and global_rank == -1 and torch.cuda.device_count() == 1 and False:  # disabled
        model = model.to(device)
        logger.info('using single GPU')
    elif device.type != 'cpu' and global_rank == -1 and torch.cuda.device_count() >= 1:
        if torch.cuda.device_count() > 1:
            logger.error('These are bugs with DP mode when device_count() > 1 due to the output format of model. '
                         'Please use DDP')
            logger.error('terminated')
            raise NotImplementedError
        model = torch.nn.DataParallel(model.to(device))
        logger.info('using DataParallel')
    elif device.type != 'cpu' and global_rank != -1:
        logger.info('using DistributedDataParallel')
        model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=cfg.train.find_unused_parameters)
        if not cfg.train.shuffle:
            logger.warning('ignore cfg.train.shuffle == False due to DDP mode')
    else: logger.info('using CPU')

    # Initialize dataset
    try:
        Dataset = importlib.import_module(cfg.train.dataset_path).Dataset
    except Exception as e:
        raise ImportError(*e.args)

    # cache
    dataset: torch.utils.data.Dataset = Dataset(cfg.train.dataset, True, logger)
    if hasattr(dataset, 'gen_cache') and dataset.gen_cache is True:
        datacache_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) \
            if global_rank != -1 else None
        datacache_loader = torch.utils.data.DataLoader(dataset, process_batch_size,
                                                       sampler=datacache_sampler,
                                                       num_workers=cfg.train.num_workers * 2, drop_last=False,
                                                       collate_fn=lambda batch: None)
        for _ in tqdm(datacache_loader):
            pass
        with open(os.path.join(dataset.cache_root, 'train_all_cached'), 'w') as f:
            pass
        logger.info('finish caching')
        # rebuild dataset to use cache
        dataset: torch.utils.data.Dataset = Dataset(cfg.train.dataset, True, logger)

    dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=cfg.train.shuffle) \
        if global_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(dataset, process_batch_size,
                                             cfg.train.shuffle if dataset_sampler is None else None,
                                             sampler=dataset_sampler, num_workers=cfg.train.num_workers, drop_last=True,
                                             pin_memory=True, collate_fn=dataset.collate_fn)
    steps_one_epoch = len(dataloader)

    # Initialize optimizers and schedulers
    params_list: List[List[torch.nn.Parameter]] = [[] for _ in range(len(cfg.train.optimizer))]

    for param_name, param in model.named_parameters():
        division_idx = params_divider(param_name)
        assert division_idx >= 0
        params_list[division_idx].append(param)

    optimizer_list = []
    scheduler_list = []

    def get_optimizer_class(name: str, momentum: float):
        if name == 'Adam':
            return partial(torch.optim.Adam, betas=(momentum, 0.999))
        elif name == 'AdamW':
            return partial(torch.optim.AdamW, betas=(momentum, 0.999))
        elif name == 'SGD':
            return partial(torch.optim.SGD, momentum=momentum, nesterov=momentum != 0.0)
        else:
            raise NotImplementedError

    for idx, params in enumerate(params_list):
        if params is None:
            logger.warning(f'The {idx}th division of parameters defined in Model: {Model} '
                           f'is empty. Is this intentional?')
            optimizer_list.append(None)
            scheduler_list.append(None)
            continue

        optimizer = get_optimizer_class(cfg.train.optimizer[idx], cfg.train.momentum[idx])(
                params=params,
                lr=cfg.train.learning_rate[idx],
                weight_decay=cfg.train.weight_decay[idx]
            )

        if cfg.train.scheduler[idx] == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.train.lr_step_size[idx],
                gamma=cfg.train.lr_step_gamma[idx])

        elif cfg.train.scheduler[idx] == 'OneCycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.train.learning_rate[idx],
                total_steps=steps_one_epoch * cfg.train.epochs,
                pct_start=cfg.train.lr_pct_start[idx],
                div_factor=cfg.train.lr_init_div_factor[idx],
                base_momentum=cfg.train.momentum[idx] - 0.05,
                max_momentum=cfg.train.momentum[idx] + 0.05,
                final_div_factor=cfg.train.lr_final_div_factor[idx])

        else: raise NotImplementedError

        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    assert not all([optimizer is None for optimizer in optimizer_list])

    # Resume checkpoint
    start_epoch = 0
    if cfg.train.resume_from_ckpt != '':
        ckpt_path = utils.autoindex_obj(cfg.train.resume_from_ckpt)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

        if 'state_dict' in cfg.train.resume_items:
            try:
                if torch_utils.is_parallel(model):
                    incompatible_keys = model.module.load_state_dict(ckpt['state_dict'], strict=False)
                else: incompatible_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
            except RuntimeError as e:
                logger.error('error when loading model_state_dict')
                raise e
            logger.info('resumed model_state_dict from checkpoint "{}"'.format(ckpt_path))
            if incompatible_keys[0] != [] or incompatible_keys[1] != []:
                logger.warning(incompatible_keys)

        if 'start_epoch' in cfg.train.resume_items:
            for idx, scheduler in enumerate(scheduler_list):
                scheduler.last_epoch = int(
                    ckpt['scheduler_state_dict'][idx]['last_epoch']
                )
            logger.info('start training from epoch {}'.format(start_epoch))

        if 'scheduler_state_dict' in cfg.train.resume_items:
            for idx, scheduler in enumerate(scheduler_list):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'][idx])
            logger.warning('resuming scheduler_state_dict, '
                           'hyperparameters of scheduler defined in yaml file will be overridden')

        if 'optimizer_state_dict' in cfg.train.resume_items:
            for idx, optimizer in enumerate(optimizer_list):
                optimizer.load_state_dict(ckpt['optimizer_state_dict'][idx])
            logger.warning('resuming optimizer_state_dict, '
                           'hyperparameters of optimizer defined in yaml file will be overridden')

        del ckpt

    # Training loop
    logger.info('start training...')
    total_steps = (cfg.train.epochs - start_epoch) * steps_one_epoch
    global_step = steps_one_epoch * start_epoch
    ave_time_onestep = None
    if cfg.train.amp:
        scaler = amp.GradScaler()
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        if global_rank != -1:
            dataloader.sampler.set_epoch(epoch)

        for step_idx, batch_data in enumerate(dataloader):
            start_time = time.time()
            if isinstance(batch_data, torch.Tensor):
                batch_data = batch_data.to(device, non_blocking=True)
            elif isinstance(batch_data, list) or isinstance(batch_data, tuple):
                batch_data = [d.to(device, non_blocking=True) if isinstance(d, torch.Tensor) else d
                              for d in batch_data]
            elif isinstance(batch_data, dict):
                batch_data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_data.items()}
            elif isinstance(batch_data, SampleData):
                batch_data.training_step = global_step
                batch_data.to(device=device, non_blocking=True)
            else: raise NotImplementedError

            if cfg.train.amp:
                with amp.autocast():
                    loss_dict: Dict[str, Union[float, torch.Tensor]] = model(batch_data)

                for idx, optimizer in enumerate(optimizer_list):
                    if optimizer is not None:
                        optimizer.zero_grad()

                scaler.scale(loss_dict['loss']).backward()

                for idx, optimizer in enumerate(optimizer_list):
                    if optimizer is not None:
                        if cfg.train.max_grad_norm[idx] != 0:
                            torch.nn.utils.clip_grad_norm_(
                                optimizer.param_groups[0]['params'],
                                cfg.train.max_grad_norm[idx]
                            )
                        scaler.step(optimizer)

                scaler.update()

            else:
                loss_dict: Dict[str, Union[float, torch.Tensor]] = model(batch_data)

                for idx, optimizer in enumerate(optimizer_list):
                    if optimizer is not None:
                        optimizer.zero_grad()

                loss_dict['loss'].backward()

                for idx, optimizer in enumerate(optimizer_list):
                    if optimizer is not None:
                        if cfg.train.max_grad_norm[idx] != 0:
                            torch.nn.utils.clip_grad_norm_(
                                optimizer.param_groups[0]['params'],
                                cfg.train.max_grad_norm[idx]
                            )
                        optimizer.step()

            # logging
            time_this_step = time.time() - start_time
            ave_time_onestep = time_this_step if ave_time_onestep is None else \
                ave_time_onestep * 0.9 + time_this_step * 0.1

            if cfg.train.log_frequency > 0 and (step_idx == 0 or (step_idx + 1) % cfg.train.log_frequency == 0):
                expected_total_time, eta = utils.eta_by_seconds((total_steps - global_step - 1) * ave_time_onestep)
                logger.info(f'step '
                            f'{step_idx}/{steps_one_epoch - 1} of epoch {epoch}/{cfg.train.epochs - 1}, '
                            f'speed: '
                            f'{utils.totaltime_by_seconds(ave_time_onestep * steps_one_epoch)}/epoch, '
                            f'eta(current): '
                            f'{utils.eta_by_seconds((steps_one_epoch - step_idx - 1) * ave_time_onestep)[1]}, '
                            f'eta(total): '
                            f'{eta} in {expected_total_time}')

                # tensorboard items
                if local_rank in (-1, 0):
                    for idx, optimizer in enumerate(optimizer_list):
                        tb_writer.add_scalar(
                            f'Train/Learning_rate_{idx}',
                            optimizer.param_groups[0]['lr'], global_step
                        )

                    total_wo_aux = 0.0
                    for item_name, item in loss_dict.items():
                        if item_name != 'loss':
                            item_category = item_name.rsplit("_", 1)[-1].capitalize()
                            tb_writer.add_scalar(f'Train/{item_category}/{item_name}', item, global_step)
                            if item_category == 'Loss' and not item_name.endswith('aux_loss'):
                                total_wo_aux += loss_dict[item_name]

                    tb_writer.add_scalar('Train/Loss/total_wo_aux', total_wo_aux, global_step)

            global_step += 1

            for idx, scheduler in enumerate(scheduler_list):
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

        for idx, scheduler in enumerate(scheduler_list):
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()

        if local_rank in (-1, 0):
            tb_writer.add_scalar('Train/Epochs', epoch, global_step - 1)

        # Save checkpoints
        model.eval()  # Set training = False before saving, which is necessary for entropy models.
        if local_rank in (-1, 0) and (epoch + 1) % cfg.train.ckpt_frequency == 0:
            ckpt_name = 'epoch_{}.pt'.format(epoch)
            ckpt = {
                'state_dict':
                    model.module.state_dict() if torch_utils.is_parallel(model) else model.state_dict(),
                'optimizer_state_dict': [
                    optimizer.state_dict() if optimizer is not None else None
                    for optimizer in optimizer_list],
                'scheduler_state_dict': [
                    scheduler.state_dict() if scheduler is not None else None
                    for scheduler in scheduler_list]
            }
            torch.save(ckpt, ckpts_dir / ckpt_name)
            del ckpt

        # Model test
        torch.cuda.empty_cache()
        if global_rank in (-1, 0) and cfg.train.test_frequency > 0 and (epoch + 1) % cfg.train.test_frequency == 0:
            test_items = test(cfg, logger, run_dir, model)
            for item_name, item in test_items.items():
                tb_writer.add_scalar('Test/' + item_name, item, global_step - 1)

        torch.cuda.empty_cache()

    dist.destroy_process_group()
    logger.info('train end')


if __name__ == '__main__':
    main()
