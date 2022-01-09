from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from torch.utils.data import ConcatDataset, DataLoader
from pytorch_lightning import LightningDataModule
from src.data import collate_fn, BalancedSampler


@dataclass
class LOSODataModule(LightningDataModule):
    data: Dict
    loader: Dict
    dataset: Dict

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = ConcatDataset([self.dataset(site) for site in self.data.train_site])
            self.val_dataset = self.dataset(self.data.test_site)

        if stage in ("test", None):
            self.test_dataset = self.dataset(self.data.test_site)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **self.loader.train,
            collate_fn=collate_fn,
            sampler=BalancedSampler(
                self.train_dataset,
                self.loader.train.batch_size,
                shuffle=True,
                replacement="over",
            )
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)