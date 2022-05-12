from copy import deepcopy
import numpy as np
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split
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
                n_batches=len(self.train_dataset) // batch_size + 5,
                alpha=1.0,
                kind="fixed",
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)


@dataclass
class OneSiteHoldoutDataModule(LightningDataModule):
    data: Dict
    loader: Dict
    dataset: Dict

    def setup(self, stage: Optional[str] = None):
        """
        load class는 site(str)을 인자로 받아 해당 사이트의 데이터를 가져옴
        LOSODataModule은 훈련 데이터와 평가 데이터가 중복되지 않으므로 이것으로 충분함
        OneSiteHoldOut의 경우 훈련 데이터와 평가 데이터가 중복되므로 같은 방식으로 구현할 경우 데이터가 중복됨
        따라서 Setup()에서 전체를 부른 다음 나누는 방식으로 구해야 할 것임
        또한, OneSiteHoldOut을 총 5회 수행해야 하므로 바꿔가면서 5번 돌려야 함
        datamodule, runner가 둘 다 필요한 상황
        하지만 Holdout을 OneSite가 아닌 전체 사이트에 대해서도 할 수 있기 때문에 이 부분은 수정이 필요함
        """
        if stage in ('fit', None):
            split_rate = 0.8
            dataset = self.dataset(self.data.test_site)
                
            n_train = int(len(dataset) * split_rate)
            self.train_dataset, self.val_dataset = random_split(
                dataset, [n_train, len(dataset) - n_train]
            )
        if stage in ('test', None):
            self.test_dataset = self.val_dataset

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
                # class_idxs=[
                #     np.where(self.train_dataset.labels == i)[0].tolist()
                #     for i in range(2)
                # ],
                class_idxs=[
                    np.where(np.array([x[1] for x in self.train_dataset]) == i)[0].tolist()
                    for i in range(2)
                ],
                batch_size=batch_size,
                n_batches=len(self.train_dataset) // batch_size + 5,
                alpha=1.0,
                kind="fixed",
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)

    def __post_init__(cls):
        super().__init__()


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
