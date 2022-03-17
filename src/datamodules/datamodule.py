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
            self.train_dataset = self.dataset(self.data.train_site)
            self.val_dataset = self.dataset(self.data.test_site)

        if stage in ("test", None):
            self.test_dataset = self.dataset(self.data.test_site)

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
                n_batches=len(self.train_dataset)//batch_size + 5,
                alpha=1.0,
                kind="fixed",
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)


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
