import os
import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
import pandas as pd
from sklearn.model_selection import KFold

import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn

from torchmetrics import Accuracy

from src.utils import get_input_info, record_kfold


#############################################################################################
#                           Step 1 / 5: Define KFold DataModule API                         #
# Our KFold DataModule requires to implement the `setup_folds` and `setup_fold_index`       #
# methods.                                                                                  #
#############################################################################################


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be splitted accordingly to        #
# the current fold split.                                                                   #
#############################################################################################


@dataclass
class KFoldDataModule(BaseKFoldDataModule):
    args: Optional[Dict] = None
    loader_config: Optional[Dict] = None
    usr_dataset: Optional[Dataset] = None

    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.table = pd.read_csv(self.args.table_path)
        self.input_info = get_input_info(self.inputs)

        self.train_dataset = self.usr_dataset(
            self.table[self.table["task"] == "train"], self.input_info
        )
        self.test_dataset = self.usr_dataset(
            self.table[self.table["task"] == "test"], self.input_info
        )

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.table = record_kfold(self.table, num_folds)

    def setup_fold_index(self, fold_index: int) -> None:
        self.train_fold = self.usr_dataset(
            self.table[self.table[f"fold_{fold_index+1}"] == "train"], self.input_info
        )
        print("TRAIN FOLD", fold_index + 1, len(self.train_fold))
        self.val_fold = self.usr_dataset(
            self.table[self.table[f"fold_{fold_index+1}"] == "valid"], self.input_info
        )
        print("VALID FOLD", fold_index + 1, len(self.val_fold))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, **self.loader_config.train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, **self.loader_config.eval)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.loader_config.eval)

    def __post_init__(cls):
        super().__init__()


#############################################################################################
#                           Step 3 / 5: Implement the EnsembleVotingModel module            #
# The `EnsembleVotingModel` will take our custom LightningModule and                        #
# several checkpoint_paths.                                                                 #
#                                                                                           #
#############################################################################################


class KFoldTestModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]):
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths]
        )
        self.test_acc = Accuracy()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        # !
        # accuracy with average softmax of folds
        # vs
        # accuracy with average accuracy of folds
        logits = torch.stack([m(batch[0]) for m in self.models])
        acc = torch.stack([self.test_acc(logit, batch[1]) for logit in logits])
        # for i in range(len(acc)):
        #     print(f"fold_{i+1}: {acc[i].item()}")

        self.log_dict({f"final_result": acc.mean()}, prog_bar=True)


#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several steps    #
# to do so which are described in detail within the documentation                           #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html.                 #
# Here, we will implement an outer fit_loop. It means we will implement subclass the        #
# base Loop and wrap the current trainer `fit_loop`.                                        #
#############################################################################################


#############################################################################################
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################


class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str, save_last: bool = False):
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path
        self.save_last = save_last

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(
            self.trainer.lightning_module.state_dict()
        )

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold+1}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        # ! keep eyes on it
        self.trainer.datamodule.setup_fold_index(self.current_fold)
        self.trainer.lightning_module.current_fold = self.current_fold + 1
        self.trainer.current_fold = self.current_fold + 1

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        if self.save_last:
            self.trainer.save_checkpoint(
                osp.join(
                    self.export_path,
                    f"model.{self.current_fold}_last.pt",
                )
            )

        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [
            osp.join(
                self.export_path,
                f"model.{f_idx + 1}.pt",
            )
            for f_idx in range(self.num_folds)
        ]
        voting_model = KFoldTestModel(
            type(self.trainer.lightning_module), checkpoint_paths
        )
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]


#############################################################################################
#                           Step 5 / 5: Connect the KFoldLoop to the Trainer                #
# After creating the `KFoldDataModule` and our model, the `KFoldLoop` is being connected to #
# the Trainer.                                                                              #
# Finally, use `trainer.fit` to start the cross validation training.                        #
#############################################################################################

if __name__ == "__main__":
    from pytorch_lightning import Trainer

    # model = ImageClassifier()
    datamodule = KFoldDataModule()
    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        num_sanity_val_steps=0,
        devices=2,
        accelerator="auto",
        strategy="ddp",
    )
    trainer.fit_loop = KFoldLoop(5, trainer.fit_loop, export_path="/app/Logs")
    # trainer.fit(model, datamodule)
