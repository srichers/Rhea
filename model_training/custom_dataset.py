'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This class is a lightweight replacement for TensorDataset.
'''

from torch.utils.data import Dataset

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
        return tuple(t[index] for t in self.tensors)
