import os
from dataclasses import dataclass
from typing import Dict, Optional, Any

from src.runners import Base_Runner
from src.data import ROIDataset
from src.tasks import ClassificationTask
from src.strategies import LOSODataModule, LOSOLoop, KFoldDataModule, KFoldLoop
from src.utils import KFoldCheckpoint

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# from pytorch_lightning import seed_everything
# seed_everything(41)


@dataclass
class KFold_Runner(Base_Runner):
    def get_callbacks(self):
        KFoldCheckpoint_callback = KFoldCheckpoint(
            dirpath=os.path.join(self.log.checkpoint_path),
            filename=os.path.join("model.{fold}.pt"),
            monitor="val/accuracy",
        )

        callbacks = dict(
            filter(lambda item: item[0].endswith("callback"), vars().items())
        ).values()
        callbacks = list(callbacks)
        return callbacks if len(callbacks) > 0 else None

    def run(self, profiler: Optional[str] = None):
        dm = self.get_datamodule(dataset=ROIDataset, datamodule=LOSODataModule)
        model = self.get_network(Task=ClassificationTask)

        trainer = Trainer(
            logger=TensorBoardLogger(
                save_dir=self.log.log_path,
                name=self.log.project_name,
                default_hp_metric=True,
                # log_graph=True, # inavailable due to bug
            ),
            # ! use all gpu
            # gpus=-1,
            # auto_select_gpus=True,
            # ! use 2 gpu
            # devices=2,
            # accelerator="auto",
            # strategy="ddp",
            # ! use gpu 0
            # devices=[0],
            # accelerator="gpu",
            devices=self.log.device.gpu,
            accelerator="gpu",
            check_val_every_n_epoch=self.log.val_log_freq_epoch,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_epochs=self.log.epoch,
            profiler=profiler,
            fast_dev_run=self.log.dry_run,
            callbacks=self.get_callbacks(),
            precision=self.log.precision,
        )

        internal_fit_loop = trainer.fit_loop
        trainer.fit_loop = KFoldLoop(
            self.log.n_fold,
            export_path=os.path.join(self.log.checkpoint_path),
            save_last=False,
        )
        trainer.fit_loop.connect(internal_fit_loop)
        trainer.fit(model, datamodule=dm)
        return trainer.callback_metrics

        
@dataclass
class LOSO_Runner(Base_Runner):
    def get_callbacks(self):
        KFoldCheckpoint_callback = KFoldCheckpoint(
            dirpath=os.path.join(self.log.checkpoint_path),
            filename=os.path.join("model.{fold}.pt"),
            monitor="val/accuracy",
        )

        callbacks = dict(
            filter(lambda item: item[0].endswith("callback"), vars().items())
        ).values()
        callbacks = list(callbacks)
        return callbacks if len(callbacks) > 0 else None

    def run(self, profiler: Optional[str] = None):
        dm = self.get_datamodule(dataset=ROIDataset, datamodule=LOSODataModule)
        model = self.get_network(Task=ClassificationTask)

        trainer = Trainer(
            logger=TensorBoardLogger(
                save_dir=self.log.log_path,
                name=self.log.project_name,
                default_hp_metric=True,
                # log_graph=True, # inavailable due to bug
            ),
            # ! use all gpu
            # gpus=-1,
            # auto_select_gpus=True,
            # ! use 2 gpu
            # devices=2,
            # accelerator="auto",
            # strategy="ddp",
            # ! use gpu 0
            # devices=[0],
            # accelerator="gpu",
            devices=self.log.device.gpu,
            accelerator="gpu",
            check_val_every_n_epoch=self.log.val_log_freq_epoch,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_epochs=self.log.epoch,
            profiler=profiler,
            fast_dev_run=self.log.dry_run,
            callbacks=self.get_callbacks(),
            precision=self.log.precision,
        )

        internal_fit_loop = trainer.fit_loop
        trainer.fit_loop = LOSOLoop(
            self.log.n_fold,
            export_path=os.path.join(self.log.checkpoint_path),
            save_last=False,
        )
        trainer.fit_loop.connect(internal_fit_loop)
        trainer.fit(model, datamodule=dm)
        return trainer.callback_metrics