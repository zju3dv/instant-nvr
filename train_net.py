import torch
import datetime
import time
import torch.nn as nn
import torch.multiprocessing
import torch.distributed as dist

import os
import random
import numpy as np
import os.path as osp

from termcolor import colored, cprint

from lib.config import cfg, args
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.evaluators import make_evaluator
from lib.utils.net_utils import load_model, save_model, load_network
from lib.utils.base_utils import bcolors, dump_cfg, get_time, git_committed, git_hash

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

if cfg.profiling:
    from torch.profiler import profile, record_function, ProfilerActivity, schedule
    print(colored(f"profiling results will be saved to: {cfg.profiling_dir}", 'yellow'))
    if cfg.clear_previous_profiling:
        print(colored(f'removing profiling result in: {cfg.profiling_dir}', 'red'))
        os.system(f'rm -rf {cfg.profiling_dir}')
    prof = profile(schedule=schedule(
        skip_first=10,
        wait=5,
        warmup=5,
        active=10,
        repeat=5
    ),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profiling_dir, use_gzip=True),
        record_shapes=True,
        #    profile_memory=True,
        with_stack=True,  # FIXME: sometimes with_stack causes segmentation fault
        with_flops=True,
        with_modules=True
    )


def fix_random(fix):
    if fix:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def print_training_stages(stage_info, stage_idx):
    return


def change_training_stages(epoch):
    if not hasattr(cfg, 'training_stages'):
        return
    stages = cfg.training_stages
    for id, stage in enumerate(stages[::-1]):
        start = stage['_start'] if not cfg.record_demo else stage["_start"] * (500 / cfg.ep_iter)
        if epoch >= start:
            print_training_stages(stage, len(stages) - id)
            for key in stage:
                if key != "_start":
                    setattr(cfg, key, stage[key])
            break


def train(cfg, network):
    fix_random(cfg.fix_random)
    if not cfg.debug:
        dump_cfg(cfg, osp.join(cfg.result_dir, "config.yaml"))
        dump_cfg(cfg, osp.join(cfg.result_dir, "{}.yaml".format(get_time())))

    trainer = make_trainer(cfg, network)
    if not cfg.silent:print("Finish initialize trainer...")
    optimizer = make_optimizer(cfg, network)
    if not cfg.silent:print("Finish initialize optimizer...")
    scheduler = make_lr_scheduler(cfg, optimizer)
    if not cfg.silent:print("Finish initialize lr scheduler...")
    recorder = make_recorder(cfg)
    if not cfg.silent:print("Finish initialize recorder...")
    evaluator = make_evaluator(cfg)
    if not cfg.silent:print("Finish initialize evaluator...")

    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)

    breakpoint()
    if cfg.pretrained_model != "none" and begin_epoch == 0:
        load_network(network, cfg.pretrained_model)
        nn.init.kaiming_normal_(network.tpose_human.embedder.data)
        nn.init.kaiming_normal_(network.tpose_human.color_network.residual.embedder.data)
        nn.init.kaiming_normal_(network.tpose_deformer.embedder.data)

    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg,
                                    split='train',
                                    is_distributed=cfg.distributed,
                                    # max_iter=cfg.ep_iter * (cfg.train.epoch - begin_epoch)
                                    max_iter=cfg.ep_iter
                                    )
    test_loader = make_data_loader(cfg, split='test')
    val_loader = make_data_loader(cfg, split='val')
    if cfg.prune_using_geo:
        tmesh_loader = make_data_loader(cfg, split='tmesh')

    # if begin_epoch == 0:
    #     trainer.val(-1, val_loader, evaluator, recorder)
    fix_random(cfg.fix_random)
    if cfg.profiling and cfg.profiler == 'torch':
        # print(f'profiler_id: {id(prof)}')
        prof.start()

    # try:
    print(colored(f"[*] Training experiment {cfg.exp_name} started, log_interval: {cfg.log_interval}", 'green'))
    for epoch in range(begin_epoch, cfg.train.epoch):
        change_training_stages(epoch)
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        trainer.train(begin_epoch, train_loader, optimizer, recorder)  # might exists a trainer change
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                        cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                        optimizer,
                        scheduler,
                        recorder,
                        cfg.trained_model_dir,
                        epoch,
                        last=True)
            train_loader.dataset.save_global()

        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

        if (epoch + 1) % cfg.vis_ep == 0:
            trainer.vis(epoch, test_loader)

        if cfg.prune_using_geo:
            trainer.tmesh(epoch, tmesh_loader)
    trainer.timer['end'] = time.time()
    trainer.print_time_elapsed()
    # except Exception as e:
    #     if isinstance(e, KeyboardInterrupt):
    #         print(bcolors.WARNING + "Interrupted by user" + bcolors.ENDC)
    #         print(bcolors.WARNING + "Saving Checkpoint" + bcolors.ENDC)
    #         if not cfg.no_save and not cfg.profiling:
    #             save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, cfg.train.epoch, last=True)
    #     else:
    #         raise e

    if cfg.profiling:
        # print(f'profiler_id: {id(prof)}')
        prof.stop()

    if not cfg.no_save and not cfg.profiling:
        save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, cfg.train.epoch, last=True)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, split='test')
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    fix_random(cfg.fix_random)  # different number of data might use the seed
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()

    network = make_network(cfg)

    if cfg.dry_run:
        print(network)
        for name, p in network.named_parameters():
            if p.requires_grad:
                print(name, p.numel())
        print(sum(p.numel() for p in network.parameters() if p.requires_grad))
        return

    if not cfg.silent:print("Finish initialize network...")
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    if cfg.detect_anomaly:
        with torch.autograd.detect_anomaly():
            main()
    else:
        main()
