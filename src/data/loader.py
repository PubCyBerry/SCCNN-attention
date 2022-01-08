import os
import numpy as np
from glob import glob
import pickle

import torch
from torch.utils.data import Dataset


class ROIDataset(Dataset):
    def __init__(self, filename: str) -> None:
        with open(filename, "rb") as f:
            self.data = pickle.load(f)
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        data shape: (116, time series)
        label: not one hot. 0 or 1.
        """
        data = self.data[index]
        label = self.labels[index]
        return data, label


def pad_tensor(vec: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    vec = torch.Tensor(vec)
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


def collate_fn(batch):
    dim = 1
    xs, ys = list(zip(*batch))
    # find longest sequence
    max_len = max(map(lambda x: x.shape[dim], xs))
    xs = torch.stack(
        list(map(lambda x: pad_tensor(x, pad=max_len, dim=dim), xs)), dim=0
    )
    ys = torch.LongTensor(ys)
    return (xs, ys)


if __name__ == "__main__":
    from time import sleep

    from torch.utils.data import ConcatDataset, DataLoader

    from src.data import BalancedSampler

    path = os.path.join("Data", "preprocessed", "all")
    filenames = glob(os.path.join(path, "*.pickle"))

    datasets = list()
    total_length = 0
    for filename in filenames:
        print("filename:", filename)
        dataset = ROIDataset(filename)
        total_length += len(dataset)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)

    for mode in ["under", None, "over"]:
        total_concat_length = 0
        batch_size = 32
        for x, y in DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=8,
            sampler=BalancedSampler(
                dataset, batch_size, shuffle=True, replacement=mode
            ),
        ):
            print(
                "{}, {}, {}, {}".format(
                    x.size(), x.is_pinned(), y.size(), y.is_pinned()
                )
            )
            for i, bins in enumerate(np.bincount(y)):
                print(f"{i}: {bins:2d} ", end="")
            print()
            total_concat_length += len(y)

        print("{}: {} -> {}".format(mode, total_length, total_concat_length))
