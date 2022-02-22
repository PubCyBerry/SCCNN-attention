from copy import deepcopy
import numpy as np
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from src.data import collate_fn, SamplerFactory


@dataclass
class LOSODataModule(LightningDataModule):
    data: Dict
    loader: Dict
    dataset: Dict

    def setup(self, stage: Optional[str] = None):

            
        if stage in ("fit", None):
            self.train_dataset = self.dataset(self.data.train_site, self.data.roi)
            self.val_dataset = self.dataset(self.data.test_site, self.data.roi)

        if stage in ("test", None):
            self.test_dataset = self.dataset(self.data.test_site, self.data.roi)

    def train_dataloader(self):
        conf = deepcopy(self.loader.train)
        batch_size = conf.pop("batch_size")
        conf.shuffle = False
        # return DataLoader(self.train_dataset, **conf, collate_fn=collate_fn)

        return DataLoader(
            self.train_dataset,
            **conf,
            collate_fn=collate_fn,
            batch_sampler=SamplerFactory().get(
                class_idxs=[
                    np.where(self.train_dataset.labels == i)[0].tolist()
                    for i in range(2)
                ],
                batch_size=batch_size,
                n_batches=30,
                alpha=1.0,
                kind="fixed",
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)


@dataclass
class SSDataModule(LightningDataModule):
    data: Dict
    loader: Dict
    dataset: Dict

    def setup(self, stage: Optional[str] = None):
        total_dataset = self.dataset(self.data.train_site)
        data_indices = list(range(len(total_dataset)))
        self.train_index, self.test_index = train_test_split(
            data_indices, test_size=0.2
        )
        if stage in ("fit", None):
            self.train_dataset = Subset(total_dataset, self.train_index)
            self.val_dataset = Subset(total_dataset, self.test_index)

        if stage in ("test", None):
            self.test_dataset = Subset(total_dataset, self.test_index)

    def train_dataloader(self):
        conf = deepcopy(self.loader.train)
        conf.shuffle = False
        batch_size = conf.pop("batch_size")
        data_labels = np.array([b[-1] for b in self.train_dataset])
        class_index = [np.where(data_labels == i)[0].tolist() for i in range(2)]
        n_batches = len(data_labels) // batch_size + 5

        return DataLoader(
            self.train_dataset,
            **conf,
            collate_fn=collate_fn,
            batch_sampler=SamplerFactory().get(
                class_idxs=class_index,
                batch_size=batch_size,
                n_batches=n_batches,
                alpha=0.1,
                kind="fixed",
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)


@dataclass
class MNISTDatamodule(LightningDataModule):
    data: Dict
    loader: Dict
    dataset: Dict

    def setup(self, stage: Optional[str] = None):
        from torchvision import datasets, transforms
        from sklearn.model_selection import KFold

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        if stage in ("fit", None):
            train = datasets.MNIST(
                "/workspace/Data/mnist", train=True, download=True, transform=transform
            )
            folds = [split for split in KFold(5).split(range(len(train)))]
            self.train_dataset = Subset(train, folds[self.fold][0])
            self.val_dataset = Subset(train, folds[self.fold][1])

        if stage in ("test", None):
            self.test_dataset = datasets.MNIST(
                "/workspace/Data/mnist", train=False, transform=transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader.train)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from src.data import ROIDataset, SITES_DICT

    conf = OmegaConf.load("/workspace/Configs/config.yaml")
    conf.merge_with_cli()
    print(conf.loader.train.batch_size, conf.loader.eval.batch_size)
    for i, (key, value) in enumerate(SITES_DICT.items()):
        train_site = deepcopy(list(SITES_DICT.keys()))
        test_site = train_site.pop(i)
        conf.data.train_site = train_site
        conf.data.test_site = test_site
        # conf.data.train_site = test_site

        dm = LOSODataModule(conf.data, conf.loader, ROIDataset)
        # dm = SSDataModule(conf.data, conf.loader, ROIDataset)
        dm.setup()
        counter = 0
        for x, y in dm.train_dataloader():
            # print(
            #     "{}, {}, {}, {}".format(
            #         x.size(), x.is_pinned(), y.size(), y.is_pinned()
            #     )
            # )
            # for i, bins in enumerate(np.bincount(y)):
            #     print(f"{i}: {bins:2d} ", end="")
            # print()
            counter += len(x)
        print(train_site, test_site, counter, len(dm.train_dataset))

    import torch

    def get_weights(dataset):
        class_label, class_counts = torch.unique(
            torch.Tensor(dataset.labels), sorted=True, return_counts=True
        )
        num_data = len(dataset)
        weights = 1.0 / (class_counts.type(torch.float) / num_data * 100)
        print(class_label, class_counts)
        return weights

    weights = get_weights(dm.train_dataset)
    print(weights)
    print(dm.train_dataloader().dataset)
