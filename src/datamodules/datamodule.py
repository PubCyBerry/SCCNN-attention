from copy import deepcopy
import numpy as np
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
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
            self.train_dataset = self.dataset(self.data.train_site)
            self.val_dataset = self.dataset(self.data.test_site)

        if stage in ("test", None):
            self.test_dataset = self.dataset(self.data.test_site)

    def train_dataloader(self):
        conf = deepcopy(self.loader.train)
        batch_size = conf.pop("batch_size")

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
class MNISTDatamodule(LightningDataModule):
    data: Dict
    loader: Dict
    dataset: Dict

    def setup(self, stage: Optional[str] = None):
        from torchvision import datasets, transforms
        from sklearn.model_selection import KFold
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if stage in ("fit", None):
            train = datasets.MNIST('/workspace/Data/mnist', train=True, download=True,
                            transform=transform)
            folds = [split for split in KFold(5).split(range(len(train)))]
            self.train_dataset = Subset(train, folds[self.fold][0])
            self.val_dataset = Subset(train, folds[self.fold][1])

        if stage in ("test", None):
            self.test_dataset = datasets.MNIST('/workspace/Data/mnist', train=False,
                            transform=transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader.train)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval)
    


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from src.data import ROIDataset, SITES_DICT

    conf = OmegaConf.load("/workspace/configs/config.yaml")
    for i, (key, value) in enumerate(SITES_DICT.items()):
        train_site = deepcopy(list(SITES_DICT.keys()))
        test_site = train_site.pop(i)
        conf.data.train_site = train_site
        conf.data.test_site = test_site

        dm = LOSODataModule(conf.data, conf.loader, ROIDataset)
        dm.setup()
        for x, y in dm.train_dataloader():
            print(
                "{}, {}, {}, {}".format(
                    x.size(), x.is_pinned(), y.size(), y.is_pinned()
                )
            )
            for i, bins in enumerate(np.bincount(y)):
                print(f"{i}: {bins:2d} ", end="")
            print()
