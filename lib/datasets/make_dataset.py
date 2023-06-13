import cv2
from .transforms import make_transforms
from . import samplers
import torch
import torch.utils.data
import importlib
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config.config import cfg
from termcolor import colored

torch.multiprocessing.set_sharing_strategy('file_system')


def _dataset_factory(split):
    splitcfg = getattr(cfg, split)
    if hasattr(splitcfg, "dataset_module") and hasattr(splitcfg, "dataset_kwargs"):
        module = splitcfg.dataset_module
        args = splitcfg.dataset_kwargs
    else:
        module = getattr(cfg, split + "_dataset_module")
        args = getattr(cfg, split + "_dataset")

    dataset = importlib.import_module(module).Dataset(**args)
    return dataset


def make_dataset(cfg, dataset_name, transforms, split='train'):
    dataset = _dataset_factory(split)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed, split):
    if split == 'train':
        if is_distributed:
            return samplers.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    elif split == 'test':
        if cfg.test.sampler == 'FrameSampler':
            sampler = samplers.FrameSampler(dataset, cfg.test.frame_sampler_interval)
            return sampler
        if is_distributed:
            return samplers.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    elif split == 'prune':
        sampler = samplers.FrameSampler(dataset, cfg.prune.frame_sampler_interval)
    elif split == 'val' and not cfg.record_demo:
        sampler = samplers.FrameSampler(dataset, cfg.val.frame_sampler_interval)
    elif split == 'val' and cfg.record_demo:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    elif split == 'bullet':
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    elif split == 'tmesh':
        sampler = samplers.FrameSampler(dataset, cfg.tmesh.frame_sampler_interval)
    elif split == 'tdmesh':
        sampler = samplers.FrameSampler(dataset, cfg.tdmesh.frame_sampler_interval)
    else:
        raise NotImplementedError

    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, split):
    batch_sampler = getattr(cfg, split).batch_sampler
    sampler_meta = getattr(cfg, split).sampler_meta

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)

    return batch_sampler


cv2.setNumThreads(1)  # MARK: OpenCV undistort is why all cores are taken


def worker_init_fn(worker_id):
    cv2.setNumThreads(1)  # MARK: OpenCV undistort is why all cores are taken
    # previous randomness issue might just come from here
    if cfg.fix_random:
        np.random.seed(worker_id)
    else:
        np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, split='train', is_distributed=False, max_iter=-1):
    batch_size = getattr(cfg, split).batch_size
    dataset_name = getattr(cfg, split).dataset
    if split == 'train':
        # shuffle = True
        shuffle = cfg.train.shuffle
        drop_last = False
    elif split == 'test' or split == 'prune' or split == 'val' or split == 'tmesh' or split == 'tdmesh' or split == 'bullet':
        shuffle = True if is_distributed else False
        drop_last = False
    else:
        raise NotImplementedError

    transforms = make_transforms(cfg, split)
    dataset = make_dataset(cfg, dataset_name, transforms, split)
    sampler = make_data_sampler(dataset, shuffle, is_distributed, split)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, split)
    num_workers = cfg.train.num_workers
    if cfg.record_demo and split == 'val':
        num_workers = 0
    collator = make_collator(cfg, split)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=num_workers,
                                              collate_fn=collator,
                                              worker_init_fn=worker_init_fn,
                                              pin_memory=True,
                                              prefetch_factor=2)

    return data_loader
