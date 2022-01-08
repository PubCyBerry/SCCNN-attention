from inspect import getargspec
import seaborn as sns

import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    AUROC,
    ConfusionMatrix,
)

from src import models
from src.utils import plot_confusion_matrix_from_data

class ClassificationTask(LightningModule):
    def __init__(self, opt_args=None, net_args=None, inputs=None):
        super().__init__()
        self.save_hyperparameters()
        self.opt_args = opt_args
        self.model = getattr(models, net_args.model)(net_args)
        self.get_metrics = MetricCollection(
            [
                Accuracy(),
                # Precision(average="macro", num_classes=net_args.num_classes),
                # Recall(average="macro", num_classes=net_args.num_classes),
                # AUROC(num_classes=2),
                # ConfusionMatrix(num_classes=2),
            ]
        )
        # for graph, but not working due to bug
        # self.example_input_array = torch.randn(
        #     size=(20, 116, 112), dtype=torch.float32
        # )


    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.nll_loss(y_hat, y)
        return loss, y_hat, y

    def _shared_epoch_end(self, outputs, loss_name, task=None):
        all_y_hat = torch.cat([x["y_hat"] for x in outputs])
        all_y = torch.cat([x["y"] for x in outputs])
        avg_loss = torch.stack([x[loss_name] for x in outputs]).mean().cpu().detach()
        metrics = self.get_metrics(all_y_hat, all_y)
        logger_log = {
            f"fold_{self.current_fold}/{task}/loss": avg_loss.item(),
            **{
                f"fold_{self.current_fold}/{task}/{key.lower()}": value.cpu().detach()
                for key, value in metrics.items()
            },
            'step': torch.tensor(self.current_epoch, dtype=torch.float32)
        }
        return avg_loss, logger_log

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def training_epoch_end(self, outputs):
        avg_loss, logger_log = self._shared_epoch_end(outputs, "loss", "train")
        self.log_dict(logger_log)
        # self.logger.agg_and_log_metrics(logger_log, step=self.current_epoch)
        # return {"loss": avg_loss, "log": logger_log}

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        return {"val_loss": loss, "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs):
        avg_loss, logger_log = self._shared_epoch_end(outputs, "val_loss", "val")
        self.log_dict(logger_log)
        # self.logger.agg_and_log_metrics(logger_log, step=self.current_epoch)
        self.log_hist()
        self.log_pr(outputs)
        self.log_cm(outputs)
        return {"log": logger_log}

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        return {"test_loss": loss, "y_hat": y_hat, "y": y}

    def test_epoch_end(self, outputs):
        avg_loss, logger_log = self._shared_epoch_end(outputs, "test_loss", "test")
        self.log_dict(logger_log)
        # self.logger.agg_and_log_metrics(logger_log, step=self.current_epoch)
        return {"log": logger_log}

    def configure_optimizers(self):
        opt = getattr(optim, self.opt_args.optimizer)(
            self.model.parameters(), lr=self.opt_args.lr
        )
        return opt
        
        # scheduler = lr_scheduler.StepLR(
        #     opt, step_size=self.opt_args.step_size, gamma=self.opt_args.gamma
        # )
        # opt_sch = {
        #     "scheduler": scheduler,
        #     # "interval": "step",  # called after each training step
        # }
        # return [opt], [opt_sch]

    def log_hist(self):
        for name, param in self.model.named_parameters():
            self.logger.experiment.add_histogram(
                f"fold_{self.current_fold}/{name}",
                param.clone().cpu().detach().numpy(),
                self.current_epoch,
            )

    def log_pr(self, outputs):
        all_y_hat = torch.cat([x["y_hat"] for x in outputs]).argmax(1)
        all_y = torch.cat([x["y"] for x in outputs])
        self.logger.experiment.add_pr_curve(
            f"fold_{self.current_fold}", all_y, all_y_hat, self.current_epoch
        )

    def log_cm(self, outputs):
        """
        https://seaborn.pydata.org/tutorial/color_palettes.html
        Be serious about colormap
        """

        all_y_hat = (
            torch.cat([x["y_hat"] for x in outputs])
            .argmax(1)
            .clone()
            .cpu()
            .detach()
            .numpy()
        )

        all_y = torch.cat([x["y"] for x in outputs]).clone().cpu().detach().numpy()
        columns = ["TDC", "ADHD"]

        self.logger.experiment.add_figure(
            f"fold_{self.current_fold}/ConfusionMatrix_{self.current_epoch}",
            plot_confusion_matrix_from_data(
                all_y,
                all_y_hat,
                columns,
                sns.cubehelix_palette(as_cmap=True),
                fz=14,
            ),
            self.current_epoch,
        )
