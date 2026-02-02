"""
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

CustomDataset class made for logical splitting and data loading.
"""

import torch
from torch.utils.data import Dataset, Sampler


class _FullSliceSampler(Sampler):
    def __iter__(self):
        yield slice(None)

    def __len__(self):
        return 1


class _SubsetSampler(Sampler):
    def __init__(self, total_samples, num_samples, generator):
        self.total_samples = total_samples
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self):
        indices = torch.randperm(self.total_samples, generator=self.generator)[
            : self.num_samples
        ].tolist()
        yield indices

    def __len__(self):
        return 1


class CustomDataset(Dataset):
    def __init__(self, *tensors):
        if len(tensors) == 0:
            raise ValueError("CustomDataset requires at least one tensor")
        length = tensors[0].shape[0]
        if any(t.shape[0] != length for t in tensors):
            raise ValueError("All tensors must have the same first dimension")
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.tensors
        if isinstance(index, (list, tuple)):
            return tuple(t[index] for t in self.tensors)
        try:
            import torch

            if isinstance(index, torch.Tensor):
                return tuple(t[index] for t in self.tensors)
        except Exception:
            pass
        try:
            import numpy as np

            if isinstance(index, np.ndarray):
                return tuple(t[index] for t in self.tensors)
        except Exception:
            pass
        return tuple(t[index] for t in self.tensors)
