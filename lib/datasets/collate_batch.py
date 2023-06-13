from torch.utils.data.dataloader import default_collate
import torch
import numpy as np


def meta_anisdf_collator(batch):
    batch = [[default_collate([b]) for b in batch_] for batch_ in batch]
    return batch


_collators = {'meta_anisdf': meta_anisdf_collator}


def make_collator(cfg, split):
    collator = getattr(cfg, split).collator

    if collator in _collators:
        return _collators[collator]
    else:
        return default_collate
