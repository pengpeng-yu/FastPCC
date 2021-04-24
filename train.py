import os
import shutil
import sys
import importlib
import time
import datetime

from loguru import logger
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
    if len(sys.argv) > 1 and (not '=' in sys.argv[1]) and sys.argv[1].endswith('.yaml'):
        cfg.merge_with_yaml(sys.argv[1])
        cfg.merge_with_dotlist(sys.argv[2:])
    else:
        cfg.merge_with_dotlist(sys.argv[1:])

    logger.remove()
    loguru_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan>  <level>{message}</level>'

    if local_rank in (-1, 0):
        logger.add(sys.stderr, colorize=True, format=loguru_format, level='DEBUG')
        os.makedirs('runs', exist_ok=True)
        run_dir = utils.autoindex_obj(os.path.join('runs', cfg.train.rundir_name))
        ckpts_dir = os.path.join(run_dir, 'ckpts')
        utils.make_new_dirs(run_dir, logger)
        utils.make_new_dirs(ckpts_dir, logger)

        logger.add(os.path.join(run_dir, 'log.txt'), format=loguru_format, level=0, mode='w')
        logger.info('preparing for training...')
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            f.write(cfg.to_yaml())

        # Tensorboard
        tb_logdir = os.path.join(run_dir, 'tb_logdir')
        utils.make_new_dirs(tb_logdir, logger)
        if cfg.train.resume_tensorboard:
            try:
                last_tb_dir = os.path.join(os.path.dirname(os.path.dirname(cfg.train.resume_from_ckpt)), 'tb_logdir')
                for log_file in os.listdir(last_tb_dir):
                    shutil.copy(os.path.join(last_tb_dir, log_file), tb_logdir)
            except Exception as e:
                e.args = (*e.args, 'Error when copying tensorboard log')
                raise e
            else:
                logger.info(f'resumed tensorboard log file(s) in {last_tb_dir}')

        tb_writer = SummaryWriter(tb_logdir)
        tb_program = program.TensorBoard()
        tb_ports = ['6006', '6007', '6008']
        for tb_port in tb_ports:
            try:
                tb_program.configure(argv=[None, '--logdir', tb_logdir, '--port', tb_port])
                tb_url = tb_program.launch()
            except Exception as e:
                logger.warning(f'fail to launch Tensorboard at http://localhost:{int(tb_port)}')
            else: break
        else:
            logger.error(f'fail to launch Tensorboard at port{tb_ports}')
            return
        logger.info(f'TensorBoard {tensorboard.__version__} at {tb_url}')

        train(cfg, local_rank, logger, tb_writer, ckpts_dir)
        
    else:
        train(cfg, local_rank, logger)


def train(cfg: Config, local_rank, logger, tb_writer=None, ckpts_dir=None):
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
    if device.type != 'cpu' and global_rank == -1 and torch.cuda.device_count() == 1 and False:  # disabled
        model = model.to(device)
        logger.info('using single GPU')
    elif device.type != 'cpu' and global_rank == -1 and torch.cuda.device_count() >= 1:
        if torch.cuda.device_count() > 1:
            logger.error('These are bugs with DP mode when device_count() > 1 due to the output format of model. '
                         'Please use DDP')
            logger.error('terminated')
            return
        model = torch.nn.DataParallel(model.to(device))
        logger.info('using DataParallel')
    elif device.type != 'cpu' and global_rank != -1:
        logger.info('using DistributedDataParallel')
        model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)
        if not cfg.train.shuffle:
            logger.warning('ignore cfg.train.shuffle == False due to DDP mode')
    else: logger.info('using CPU')

    # Initialize dataset
    try:
        Dataset = importlib.import_module(cfg.dataset_path).Dataset
    except Exception as e:
        raise ImportError(*e.args)
    dataset: torch.utils.data.Dataset = Dataset(cfg.dataset, True)
    dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if global_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(dataset, process_batch_size,
                                             cfg.train.shuffle if dataset_sampler is None else None,
                                             sampler=dataset_sampler, num_workers=cfg.train.num_workers, drop_last=True)

    # Initialize optimizer and scheduler
    if cfg.train.optimizer == 'adam': Optimizer = torch.optim.Adam
    elif cfg.train.optimizer == 'sgd': Optimizer =  torch.optim.SGD
    else: raise NotImplementedError
    parameters = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_parameters = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = Optimizer([{'params': parameters, 'lr': cfg.train.learning_rate, 'weight_decay': cfg.train.weight_decay},
                           {'params': aux_parameters, 'lr': cfg.train.learning_rate, 'weight_decay': cfg.train.aux_weight_decay}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.train.lr_step_size, cfg.train.lr_step_gamma)

    # Resume checkpoint
    start_epoch = 0
    if cfg.train.resume_from_ckpt != '':
        ckpt = torch.load(utils.autoindex_obj(cfg.train.resume_from_ckpt), map_location=torch.device('cpu'))
        if 'state_dict' in cfg.train.resume_items:
            try:
                if torch_utils.is_parallel(model):
                    model.module.load_state_dict(ckpt['state_dict'])
                else: model.load_state_dict(ckpt['state_dict'])
            except Exception as e:
                logger.error('error when loading model_state_dict')
                raise e
            logger.info('resuming model_state_dict from checkpoint "{}"'.format(cfg.train.resume_from_ckpt) )

        if 'start_epoch' in cfg.train.resume_items:
            start_epoch = ckpt['scheduler_state_dict']['last_epoch']
            scheduler.last_epoch = int(start_epoch)
            logger.info('start training from epoch {}'.format(start_epoch))

        if 'scheduler_state_dict' in cfg.train.resume_items:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            logger.warning('resuming scheduler_state_dict, '
                           'hyperparameters of scheduler defined in yaml file will be overrided')
        if 'optimizer_state_dict' in cfg.train.resume_items:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            logger.warning('resuming optimizer_state_dict, '
                           'hyperparameters of optimizer defined in yaml file will be overrided')

        del ckpt

    # Training loop
    logger.info('start training...')
    steps_one_epoch = len(dataloader)
    total_steps = (cfg.train.epochs - start_epoch) * steps_one_epoch
    global_step = steps_one_epoch * start_epoch
    ave_time_onestep = None
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        if global_rank != -1:
            dataloader.sampler.set_epoch(epoch)

        for step_idx, data in enumerate(dataloader):
            start_time = time.time()
            data = data.to(device)

            loss_dict = model(data)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            time_this_step = time.time() - start_time
            ave_time_onestep = time_this_step if ave_time_onestep is None \
                               else ave_time_onestep * 0.9 + time_this_step * 0.1

            if cfg.train.log_frequency > 0 and step_idx % cfg.train.log_frequency == 0:
                expected_total_time, eta = utils.eta_by_seconds((total_steps - global_step - 1) * ave_time_onestep)
                logger.info(f'step {step_idx + 1}/{steps_one_epoch} of epoch {epoch + 1}/{cfg.train.epochs}, '
                            f'speed: {utils.totaltime_by_seconds(ave_time_onestep * steps_one_epoch)}/epoch, '
                            f'eta(current): {utils.eta_by_seconds((steps_one_epoch - step_idx - 1) * ave_time_onestep)[1]}, '
                            f'eta(total): {eta} in {expected_total_time}')

                # tensorboard items
                if local_rank in (-1, 0):
                    tb_writer.add_scalar('Train/Learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    total_wo_aux = 0.0
                    for item_name, item in loss_dict.items():
                        if item_name == 'loss':
                            pass
                        else:
                            item_categroy = item_name.rsplit("_", 1)[-1].capitalize()
                            tb_writer.add_scalar(f'Train/{item_categroy}/{item_name}', item, global_step)
                            if item_categroy == 'Loss' and item_name != 'aux_loss':
                                total_wo_aux += loss_dict[item_name]
                    tb_writer.add_scalar('Train/Loss/total_wo_aux', total_wo_aux, global_step)

            global_step += 1
            # break  # TODO
        scheduler.step()

        # Model test
        if global_rank in (-1, 0) and cfg.train.test_frequency > 0 and epoch % cfg.train.test_frequency == 0:
            test_items = test(cfg, logger, model)
            for item_name, item in test_items.items():
                tb_writer.add_scalar('Test/' + item_name, item, len(dataloader) * epoch)

        # Save checkpoints
        if local_rank in (-1, 0) and epoch % cfg.train.ckpt_frequency == 0:
            ckpt_name = 'epoch_{}.pt'.format(epoch)
            ckpt = {
                'state_dict': model.module.state_dict() if torch_utils.is_parallel(model) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(ckpt, os.path.join(ckpts_dir, ckpt_name))
            del ckpt

    if torch_utils.is_parallel(model):
        model.module.entropy_bottleneck.update()
    else:
        model.entropy_bottleneck.update()

    dist.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()