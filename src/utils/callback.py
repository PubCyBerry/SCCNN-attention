import os
from dataclasses import dataclass

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback


class KFoldCheckpoint(Callback):
    def __init__(self, dirpath: str, filename: str, monitor="val/accuracy"):
        super().__init__()
        self.monitor = monitor
        self.best = 0
        self.dirpath = dirpath
        self.filename = filename
        self.current_fold = 0

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        trainer_fold = trainer.current_fold
        if self.current_fold != trainer_fold:
            self.current_fold = trainer_fold
            self.best = 0

        metrics = trainer.callback_metrics
        monitor = f"fold_{self.current_fold}/{self.monitor}"
        current = metrics[monitor]

        if current > self.best:
            self.best = current
            filepath = os.path.join(os.getcwd(), self.dirpath, self.filename)
            filepath = filepath.format(**dict(fold=self.current_fold))
            trainer.save_checkpoint(filepath, weights_only=False)