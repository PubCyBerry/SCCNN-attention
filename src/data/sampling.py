from typing import Optional, List
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class SamplerManager:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        weights: List = [0.5, 0.5],
        method: str = "oversampling",
        fix_number: bool = True,
        sweep: bool = False,
    ):
        labels = torch.Tensor(list(map(lambda x: x[1], dataset)))
        weights = torch.Tensor(weights)

        assert sum(weights) == 1, "sum of weights should be 1"
        # if fix_number, all batch proportions are same. Otherwise, randomly sampled.
        if fix_number:
            self.sequence = Weighted_Random_Fixed_Sequencer(
                labels, batch_size, shuffle, weights, method, sweep
            )
        else:
            self.sequence = Weighted_Random_Sampled_Sequencer(
                labels, batch_size, shuffle, weights, method, sweep
            )

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)


def get_class_labels(labels, shuffle=True):
    num_classes = int(max(labels)) + 1
    class_labels = list()
    for i in range(num_classes):
        label = torch.where(labels == i)[0]
        if shuffle:
            label = label[torch.randperm(label.size(0))]
        class_labels.append(label)
    return class_labels

def get_batch_portion(batch_size, weights):
    portion = torch.round(weights * batch_size)
    remainder = batch_size - portion.sum()
    portion[portion.argmin(axis=0)] += remainder
    assert (
        batch_size == portion.sum()
    ), "not enough batch size. batch size: %d portion: %d" % (batch_size, portion.sum())
    return portion

def interpolate(weights, n_div=25):
    left, right = weights, 1 - weights
    result = list()
    for w in np.linspace(0, 1, n_div):
        result.extend(left * (1 - w) + right * (w))
    return torch.Tensor(result).view(-1,2)

def stack_batch(class_labels, batch_size, weights, method, sweep):
    sequence = list()
    if sweep:
        weights_list = interpolate(weights)
        i = 0
        for weights in weights_list:
            portion = get_batch_portion(batch_size, weights)
            for j, (c, p) in enumerate(zip(class_labels, portion)):
                if (i * p) > len(c):
                    class_labels[j] = torch.concat([c, c[torch.randperm(len(c))]])
                p = int(p)
                sequence.extend(c[i * p : (i + 1) * p])
            i += 1

    else:
        portion = get_batch_portion(batch_size, weights)
        class_len = torch.Tensor(list(map(len, class_labels)))
        min_batch = (class_len / portion).floor()
        remainder = (class_len % portion)
        remainders = list()

        if method == 'undersampling':
            m = int(min(min_batch))
            for i, (c, r, p) in enumerate(zip(class_labels, remainder, portion)):
                r, p = int(r), int(p)
                # drop remainder
                if r > 0:
                    c = c[:-r]
                    remainders.append(c[-r:])
                # column becomes batch
                c = c.view(p, -1)
                # drop exceeding
                c = c[:, :m]
                sequence.append(c)

        elif method == 'oversampling':
            lack_batch = (max(min_batch) - min_batch) / min_batch
            M = int(max(min_batch))
            for i, (c, r, p, l) in enumerate(zip(class_labels, remainder, portion, lack_batch)):
                r, p = int(r), int(p)
                if l > 0:
                    quo = int(l)
                    rem = int(((l - quo)* p * min_batch[i]).round())
                    len_c = len(c)
                    for j in range(quo):
                        if r > 0:
                            c = torch.concat((c, c[torch.randperm(len_c)][:-r]))
                            remainders.append(c[-r:])
                        else:
                            c = torch.concat((c, c[torch.randperm(len_c)]))
                    c = torch.concat((c, c[torch.randperm(len_c)][:rem]))

                if r > 0:
                    c = c[:-r]
                    remainders.append(c[-r:])
                c = c.view(p, -1)
                c = c[:, :M]
                sequence.append(c)
    
        else:
            m = int(min(min_batch))
            for i, (c, r, p) in enumerate(zip(class_labels, remainder, portion)):
                r, p = int(r), int(p)
                # drop remainder
                if r > 0:
                    c = c[:-r]
                    remainders.append(c[-r:])
                # column becomes batch
                c = c.view(p, -1)
                # drop exceeding
                c = c[:, :m]
                sequence.append(c)
                remainders.append(c[:,m:].flatten())

        r = torch.concat(remainders)
        r = r[torch.randperm(len(r))]
        sequence = torch.concat(sequence, dim=0).T.flatten()
        sequence = torch.concat([sequence, r], dim=0)

    return torch.LongTensor(sequence)


class Weighted_Random_Fixed_Sequencer:
    def __init__(self, labels, batch_size, shuffle, weights, method, sweep):
        class_labels = get_class_labels(labels, shuffle)
        self.sequence = stack_batch(class_labels, batch_size, weights, method, sweep)

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)


class Weighted_Random_Sampled_Sequencer:
    def __init__(self, labels, batch_size, shuffle, weights, method, sweep):
        class_labels = get_class_labels(labels, shuffle)
        pass
