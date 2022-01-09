from typing import Optional, List

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class BalancedSampler(Sampler):
    """
    # TODO: add weighted fixed sampler  
    # TODO: add weighted random sampler
    # TODO: add option under/none/oversampling to above two
    Version 1
    define order: how to get batch.
    choose 2 options in batch distribution:
    1. Weighted Batch
    2. Normal Batch (default, None)

    Replacement: 'over', 'under', False
    """

    def __init__(
        self,
        data: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        weights: Optional[List] = None,
        replacement: Optional[str] = False,
    ):
        """
        sparse label available only for now.
        """

        self.data =data 
        self.batch_size = batch_size

        labels = torch.Tensor((list(map(lambda x: x[1], data))))
        self.batch_sequence = self.get_sequence(labels, shuffle, weights, replacement)

    def __iter__(self):
        return iter(self.batch_sequence)

    def __len__(self):
        return len(self.batch_sequence)

    def get_num_classes(self, labels):
        """
        check label is one-hot or sparse.
        then get and return num_classes.
        """
        if len(labels.size()) > 1:
            # print("one hot label")
            return int(labels.size(1))
        else:
            # print("sparse label")
            return int(max(labels)) + 1

    def get_sequence(self, labels, shuffle, weights, replacement):
        """
        It ensures all batches have same class distribution.
        Undersample: drop all residue.
        Oversample : fill other classes by maximum length.
        None       : simply concatenate residue
        total dataset length will be different from your strategy.

        ex)
        label 0 = [3, 1, 2, 5]
        label 1 = [4, 6, 9, 13, 15]
        -> [3, 4, 1, 6, 2, 9, 5, 13, 15]
        """
        num_classes = self.get_num_classes(labels)

        sequence = None
        if weights is None:
            # get label indices, not value.
            clabels = [torch.where(labels == i)[0] for i in range(num_classes)]
            if shuffle:
                for c in clabels:
                    c = c[torch.randperm(len(c))]

            # order by length, ascending
            clabels = sorted(clabels, key=len)
            # 0th element of clabels has minimum length
            min_length = len(clabels[0])
            # last element of clabels has maximum length
            max_length = len(clabels[-1])
            # merge clabels until minimum length

            if replacement is "over":
                length_to_fill = list(map(lambda x: max_length - len(x), clabels[:-1]))
                oversampled = [
                    c[torch.randperm(c.size(0))][:length]
                    for c, length in zip(clabels[:-1], length_to_fill)
                ]
                clabels[:-1] = [
                    torch.concat([c, over])
                    for c, over in zip(clabels[:-1], oversampled)
                ]
                sequence = torch.stack(clabels, dim=1).flatten()

            else:
                sequence = torch.stack(
                    [c[:min_length] for c in clabels], dim=1
                ).flatten()
                if replacement is "under":
                    pass
                elif replacement is None:
                    sequence = torch.concat(
                        [sequence, *[c[min_length:] for c in clabels[1:]]]
                    )

        return sequence