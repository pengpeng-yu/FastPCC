import os
import os.path as osp
import sys
import importlib
import time
import datetime
import pathlib
from tqdm import tqdm
from functools import partial
from typing import Dict, Union, List, Callable
import subprocess
import socket
import traceback
import signal
from contextlib import nullcontext

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX as MODULE_EXTRA_STATE_KEY_SUFFIX
import torch.backends.cudnn as cudnn
from lib.model_ema import ModelEmaV3

from test import test
from lib.config import Config
from lib.utils import autoindex_obj, make_new_dirs, eta_by_seconds, totaltime_by_seconds
from lib.torch_utils import select_device, unwrap_ddp, load_loose_state_dict
from lib.data_utils import SampleData


def main():
    # DDP arg
    for arg_idx, arg in enumerate(sys.argv):
        if '--local_rank' in arg or '--local-rank' in arg:
            local_rank = int(arg[len('--local_rank='):])
            sys.argv.pop(arg_idx)
            break
    else:
        try:
            local_rank = int(os.environ['LOCAL_RANK'])
        except KeyError:
            local_rank = -1

    # Initialize config, run dir, logger
    cfg = Config()
    cfg.merge_with_dotlist(sys.argv[1:])
    cfg.check()
    from loguru import logger
    logger.remove()

    if local_rank in (-1, 0):
        loguru_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green>' \
                        ' <level>{level: <4}</level>' \
                        ' <level>{message}</level>'
        logger.add(sys.stderr, colorize=True, format=loguru_format, level='DEBUG')
        os.makedirs('runs', exist_ok=True)
        run_dir = pathlib.Path(autoindex_obj(
            osp.join('runs', cfg.train.rundir_name) if not osp.isabs(cfg.train.rundir_name)
            else cfg.train.rundir_name))
        ckpts_dir = run_dir / 'ckpts'
        make_new_dirs(run_dir, logger)
        make_new_dirs(ckpts_dir, logger)
        logger.add(run_dir / 'log.txt', format=loguru_format, level=0, mode='w')
        logger.info('preparing for training...')
        with open(run_dir / 'config.yaml', 'w') as f:
            f.write(cfg.to_yaml())

        # Tensorboard
        tb_logdir = run_dir / 'tb_logdir'
        make_new_dirs(tb_logdir, logger)
        tb_writer = SummaryWriter(str(tb_logdir))
        tb_port = cfg.train.tensorboard_port
        if tb_port == -1:
            logger.warning(f'disable launching Tensorboard due to train.tensorboard_port=-1.\n'
                           f'Set a valid value, e.g., train.tensorboard_port=6006, to enable it.')
            tb_proc = None
        else:
            try:
                while True:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        port_available = s.connect_ex(('localhost', tb_port)) != 0
                    if port_available: break
                    tb_port += 1
                tb_proc = subprocess.Popen(
                    [osp.join(osp.split(sys.executable)[0], 'tensorboard'),
                     f'--port={tb_port}', '--logdir', str(tb_logdir)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8'
                )
                for _ in range(50):
                    stdout_line = tb_proc.stdout.readline()
                    logger.info(stdout_line)
                    if stdout_line.startswith('TensorBoard '):
                        logger.info(stdout_line.rsplit('(', 1)[0])
                        break
                else:
                    raise Exception
                tb_proc.stdout.close()

                def handle_sigterm(signal, frame):
                    tb_proc.terminate()
                    sys.exit(0)
                signal.signal(signal.SIGTERM, handle_sigterm)
            except Exception as e:
                logger.warning(f'failed to launch Tensorboard')
                tb_proc = None

        try:
            train(cfg, local_rank, logger, tb_writer, run_dir, ckpts_dir)
        except (KeyboardInterrupt, Exception):
            e_type, e_value, e_traceback = sys.exc_info()
            logger.error(e_type)
            logger.error(e_value)
            logger.error('\n' + ''.join(traceback.format_tb(e_traceback)))
            if tb_proc is not None:
                tb_proc.kill()
        else:
            if tb_proc is not None:
                tb_proc.kill()
                logger.info(f'run "tensorboard --logdir {tb_logdir} --port={tb_port}" to reopen tensorboard.')

    else:
        train(cfg, local_rank, logger)


def train(cfg: Config, local_rank, logger, tb_writer=None, run_dir=None, ckpts_dir=None):
    # Parallel training
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE']) if 'LOCAL_WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    device, cuda_ids = select_device(logger, local_rank, cfg.train.device, cfg.train.batch_size)

    if local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(hours=2))
        assert cfg.train.batch_size % world_size == 0, '--batch-size must be multiple of CUDA device count'
        process_batch_size = cfg.train.batch_size // world_size

    else:
        process_batch_size = cfg.train.batch_size
    logger.info(f'world_size: {world_size}, global_rank: {global_rank}, '
                f'local_world_size: {local_world_size}, local_rank: {local_rank}')

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    else:
        logger.warning(f'ignore config "float32_matmul_precision={cfg.float32_matmul_precision}" '
                       f'since torch.set_float32_matmul_precision is not available')

    if not cfg.more_reproducible:
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        cudnn.benchmark, cudnn.deterministic = False, True

    # Initialize model
    try:
        Model = importlib.import_module(cfg.model_module_path).Model
    except Exception as e:
        raise ImportError(*e.args)
    model = Model(cfg.model).to(device)
    if cfg.train.ema is True:
        ema = ModelEmaV3(
            model, decay=cfg.train.ema_decay, use_warmup=cfg.train.ema_warmup, foreach=cfg.train.ema_foreach,
            warmup_gamma=cfg.train.ema_warmup_gamma, warmup_power=cfg.train.ema_warmup_power)
    logger.info(f'repr(model): \n{repr(model)}')
    total_params = 0
    logger.info("Submodules and their parameter counts:")
    for name, submodule in model.named_children():
        num_params = sum(p.numel() for p in submodule.parameters())
        logger.info(f"  {name}: {num_params / 1e6:.4f}M")
        total_params += num_params
    logger.info(f"Total parameters: {total_params / 1e6:.4f}M")

    if hasattr(model, 'params_divider'):
        params_divider = model.params_divider
    else:
        assert len(cfg.train.optimizer) == 1
        params_divider: Callable[[str], int] = lambda s: 0

    if cuda_ids[0] != -1 and global_rank == -1 and len(cuda_ids) == 1:
        logger.info('using a single GPU')
    elif cuda_ids[0] != -1 and global_rank == -1 and len(cuda_ids) >= 1:
        if len(cuda_ids) > 1:
            logger.error('These are designs incompatible with DP mode. Please use DDP.')
            logger.error('terminated')
            raise NotImplementedError
    elif cuda_ids[0] != -1 and global_rank != -1:
        logger.info('using DistributedDataParallel')
        # Old versions of pytorch infer params and buffers from state_dict in DDP module,
        # which may cause errors of broadcasting if a model has non-tensor states.
        # (lib.entropy_models.continuous_base.DistributionQuantizedCDFTable._extra_state, in my case.)
        # Explicit named_params and named_buffers are used since this pull,
        # https://github.com/pytorch/pytorch/pull/65181.
        # For backward compatibility, I ignore all "_extra_state", as a simple workaround.
        dpp_params_and_buffers_to_ignore = []
        for state_name in model.state_dict():
            if state_name.endswith(MODULE_EXTRA_STATE_KEY_SUFFIX):
                dpp_params_and_buffers_to_ignore.append(state_name)
        DDP._set_params_and_buffers_to_ignore_for_model(model, dpp_params_and_buffers_to_ignore)
        model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=cfg.train.find_unused_parameters,
                    bucket_cap_mb=cfg.train.bucket_cap_mb or None)
        if not cfg.train.shuffle:
            logger.warning('ignore cfg.train.shuffle == False due to DDP mode')
    else: logger.info('using CPU')

    # Initialize dataset
    try:
        Dataset = importlib.import_module(cfg.train.dataset_module_path).Dataset
    except Exception as e:
        raise ImportError(*e.args)

    # cache
    dataset: torch.utils.data.Dataset = Dataset(cfg.train.dataset, True, logger)
    if hasattr(dataset, 'gen_cache') and dataset.gen_cache is True:
        datacache_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=local_world_size, rank=local_rank, shuffle=False, drop_last=False
        ) if global_rank != -1 else None
        datacache_loader = torch.utils.data.DataLoader(
            dataset, process_batch_size,
            sampler=datacache_sampler,
            num_workers=cfg.train.num_workers * 2, drop_last=False,
            collate_fn=lambda batch: None
        )
        for _ in tqdm(datacache_loader):
            pass
        if isinstance(model, DDP):
            dist.barrier()
        with open(osp.join(dataset.cache_root, 'train_all_cached'), 'w') as f:
            pass
        logger.info('finish caching')
        # rebuild dataset to use cache
        dataset: torch.utils.data.Dataset = Dataset(cfg.train.dataset, True, logger)

    dataset_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=cfg.train.shuffle, drop_last=True
    ) if global_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(
        dataset, process_batch_size,
        cfg.train.shuffle if dataset_sampler is None else None,
        sampler=dataset_sampler, num_workers=cfg.train.num_workers, drop_last=True,
        pin_memory=cfg.train.pin_memory, persistent_workers=cfg.train.num_workers != 0,
        prefetch_factor=cfg.train.prefetch_factor if cfg.train.num_workers != 0 else None,
        collate_fn=getattr(dataset, 'collate_fn', None)
    )
    steps_one_epoch = len(dataloader)
    total_steps = cfg.train.epochs * steps_one_epoch

    # Initialize optimizers and schedulers
    params_list: List[List[torch.nn.Parameter]] = [[] for _ in range(len(cfg.train.optimizer))]
    param_names_list = [[] for _ in range(len(cfg.train.optimizer))]
    for param_name, param in model.named_parameters():
        division_idx = params_divider(param_name)
        if division_idx >= 0:
            params_list[division_idx].append(param)
            param_names_list[division_idx].append(param_name)
    optimizer_list = []
    scheduler_list = []
    if len(params_list) > 1:
        for idx, param_names in enumerate(param_names_list):
            logger.info(f'Param Group {idx}:\n' + '\n'.join(param_names))

    def get_optimizer_class(name: str, momentum: float, beta2: float = 0.999):
        if name == 'Adam':
            return partial(torch.optim.Adam, betas=(momentum, beta2), amsgrad=False)
        if name == 'AdamAMS':
            return partial(torch.optim.Adam, betas=(momentum, beta2), amsgrad=True)
        elif name == 'AdamW':
            return partial(torch.optim.AdamW, betas=(momentum, beta2), amsgrad=False)
        elif name == 'AdamWAMS':
            return partial(torch.optim.AdamW, betas=(momentum, beta2), amsgrad=True)
        elif name == 'RAdam':
            return partial(torch.optim.RAdam, betas=(momentum, beta2))
        elif name == 'Adamax':
            return partial(torch.optim.Adamax, betas=(momentum, beta2))
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
                gamma=cfg.train.lr_step_gamma[idx]
            )
        else: raise NotImplementedError
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)
    assert not all([optimizer is None for optimizer in optimizer_list])

    # Resume checkpoint
    start_epoch = 0
    if cfg.train.from_ckpt != '':
        ckpt_path = autoindex_obj(cfg.train.from_ckpt)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if 'state_dict' in cfg.train.resume_items:
            incompatible_keys, missing_keys, unexpected_keys = load_loose_state_dict(
                model, ckpt['state_dict'] if 'state_dict' in ckpt else ckpt['ema_state_dict'])
            logger.info('resumed model_state_dict from checkpoint "{}"'.format(ckpt_path))
            if len(incompatible_keys) != 0:
                logger.warning(f'incompatible keys:\n{incompatible_keys}')
            if len(missing_keys) != 0:
                logger.warning(f'missing keys:\n{missing_keys}')
            if len(unexpected_keys) != 0:
                logger.warning(f'unexpected keys:\n{unexpected_keys}')
            if cfg.train.ema:
                load_loose_state_dict(
                    ema.module, ckpt['ema_state_dict'] if 'ema_state_dict' in ckpt else ckpt['state_dict'])
                logger.info('resumed ema_state_dict from checkpoint "{}"'.format(ckpt_path))
        if 'scheduler_state_dict' in cfg.train.resume_items:
            for idx, scheduler in enumerate(scheduler_list):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'][idx])
            logger.warning('resumed scheduler_state_dict, '
                           'hyperparameters of scheduler defined in yaml file will be overridden')
            start_epoch = ckpt['last_epoch'] + 1
            logger.info('start training from epoch {}'.format(start_epoch))
        if 'optimizer_state_dict' in cfg.train.resume_items:
            for idx, optimizer in enumerate(optimizer_list):
                optimizer.load_state_dict(ckpt['optimizer_state_dict'][idx])
            logger.warning('resumed optimizer_state_dict, '
                           'hyperparameters of optimizer defined in yaml file will be overridden')
        del ckpt_path, incompatible_keys, missing_keys, unexpected_keys, ckpt

    # Training loop
    logger.info('start training...')
    global_step = steps_one_epoch * start_epoch
    ave_time_onestep = None
    scaler = amp.GradScaler(enabled=cfg.train.amp)
    for epoch in range(start_epoch, cfg.train.epochs):
        if not model.training:
            model.train()
        if global_rank != -1:
            dataloader.sampler.set_epoch(epoch)

        dataloader_iter = iter(dataloader)
        for step_idx in range(len(dataloader)):
            start_time = time.time()
            batch_data = next(dataloader_iter)
            if isinstance(batch_data, torch.Tensor):
                batch_data = batch_data.to(device, non_blocking=True)
            elif isinstance(batch_data, list) or isinstance(batch_data, tuple):
                batch_data = [d.to(device, non_blocking=True) if isinstance(d, torch.Tensor) else d
                              for d in batch_data]
            elif isinstance(batch_data, dict):
                batch_data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_data.items()}
            elif isinstance(batch_data, SampleData):
                batch_data.training_step = global_step // cfg.train.grad_acc_steps  # Optimization steps
                batch_data.to(device=device, non_blocking=True)
            else: raise NotImplementedError

            no_sync = isinstance(model, DDP) and (global_step + 1) % cfg.train.grad_acc_steps != 0
            with model.no_sync() if no_sync else nullcontext():
                with amp.autocast(enabled=cfg.train.amp):
                    loss_dict: Dict[str, Union[float, torch.Tensor]] = model(batch_data)
                scaler.scale(loss_dict['loss'] / cfg.train.grad_acc_steps).backward()

            if (global_step + 1) % cfg.train.grad_acc_steps == 0:
                for idx, optimizer in enumerate(optimizer_list):
                    if optimizer is not None:
                        scaler.unscale_(optimizer)
                        if cfg.train.max_grad_norm[idx] != 0:
                            torch.nn.utils.clip_grad_norm_(
                                optimizer.param_groups[0]['params'],
                                cfg.train.max_grad_norm[idx],
                                error_if_nonfinite=True
                            )
                        scaler.step(optimizer)
                scaler.update()
                for idx, optimizer in enumerate(optimizer_list):
                    if optimizer is not None:
                        optimizer.zero_grad()
                if cfg.train.ema is True:
                    ema.update(unwrap_ddp(model), step=(global_step // cfg.train.grad_acc_steps))

            # logging
            time_this_step = time.time() - start_time
            ave_time_onestep = time_this_step if ave_time_onestep is None else \
                ave_time_onestep * 0.9 + time_this_step * 0.1
            if cfg.train.log_frequency > 0 and (global_step == 0 or (global_step + 1) % cfg.train.log_frequency == 0):
                expected_total_time, eta = eta_by_seconds((total_steps - global_step - 1) * ave_time_onestep)
                logger.info(
                    f'step '
                    f'{step_idx}/{steps_one_epoch - 1} of epoch {epoch}/{cfg.train.epochs - 1}, '
                    f'speed: '
                    f'{totaltime_by_seconds(ave_time_onestep * steps_one_epoch)}/e, '
                    f'eta(cur): '
                    f'{eta_by_seconds((steps_one_epoch - step_idx - 1) * ave_time_onestep)[1]}, '
                    f'eta(total): '
                    f'{eta} in {expected_total_time}'
                )
                # tensorboard items
                if local_rank in (-1, 0):
                    for idx, optimizer in enumerate(optimizer_list):
                        tb_writer.add_scalar(
                            f'Train/Learning_rate_{idx}',
                            optimizer.param_groups[0]['lr'], global_step
                        )
                    for item_name, item in loss_dict.items():
                        if isinstance(item, (torch.Tensor, np.ndarray)):
                            item = item.item()
                        if item_name != 'loss':
                            item_category = item_name.rsplit("_", 1)[-1].capitalize()
                            tb_writer.add_scalar(f'Train/{item_category}/{item_name}', item, global_step)
                    tb_writer.add_scalar('Train/Loss/total', loss_dict['loss'].item(), global_step)

            global_step += 1

            if cfg.train.cuda_empty_cache_frequency != 0 and \
                    global_step % cfg.train.cuda_empty_cache_frequency == 0:
                logger.info(f'torch.cuda.max_memory_reserved(): {torch.cuda.max_memory_reserved()}. '
                            f'Call torch.cuda.empty_cache() now')
                torch.cuda.empty_cache()

        for idx, scheduler in enumerate(scheduler_list):
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()

        if local_rank in (-1, 0):
            tb_writer.add_scalar('Train/Epochs', epoch, global_step - 1)

        # Save checkpoints
        if local_rank in (-1, 0) and (epoch + 1) % cfg.train.ckpt_frequency == 0:
            model.eval()  # Set training = False before saving, which is necessary for entropy models.
            ckpt_name = 'epoch_{}.pt'.format(epoch)
            ckpt = {
                'state_dict': unwrap_ddp(model).state_dict(),
                'optimizer_state_dict': [
                    optimizer.state_dict() if optimizer is not None else None
                    for optimizer in optimizer_list],
                'scheduler_state_dict': [
                    scheduler.state_dict() if scheduler is not None else None
                    for scheduler in scheduler_list],
                'last_epoch': epoch
            }
            if cfg.train.ema:
                ckpt['ema_state_dict'] = ema.module.state_dict()
            torch.save(ckpt, ckpts_dir / ckpt_name)
            del ckpt

        # Model test
        if global_rank in (-1, 0) and cfg.train.test_frequency > 0 and (epoch + 1) % cfg.train.test_frequency == 0 and \
                (not getattr(unwrap_ddp(model), 'warmup_forward', False)):
            test_items = test(cfg, logger, run_dir, model)
            for item_name, item in test_items.items():
                tb_writer.add_scalar('Test/' + item_name, item, global_step - 1)
            if cfg.train.ema:
                test_items = test(cfg, logger, run_dir, ema.module)
                for item_name, item in test_items.items():
                    tb_writer.add_scalar('TestEma/' + item_name, item, global_step - 1)
            torch.cuda.empty_cache()

    if tb_writer is not None: tb_writer.close()
    logger.info('train end')


if __name__ == '__main__':
    main()
