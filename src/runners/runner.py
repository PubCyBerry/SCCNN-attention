import os
import pickle
import pandas as pd
from glob import glob
from typing import Optional
from copy import deepcopy

from src.runners import Base_Runner
from src.data import ROIDataset, SITES_DICT
from src.datamodules import LOSODataModule, OneSiteHoldoutDataModule
from src.tasks import ClassificationTask
from src.utils import plot_paper, record_train_test
from src.callbacks import wandb_callback as wbc
from src.callbacks import tensorboard_callback as tbc

import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pytorch_lightning import seed_everything


seed_everything(41)


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            mul = param.shape[0] // 4
            for idx in range(4):
                if "bias" in name:
                    nn.init.constant_(param, 0.00)
                elif "weight_ih" in name:
                    nn.init.xavier_normal_(param.data[idx * mul : (idx + 1) * mul])
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data[idx * mul : (idx + 1) * mul])


class OneSiteHoldout_Runner(Base_Runner):
    def get_callbacks(self, site: str):
        """
        Write only callbacks that logger is not necessary
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                self.log.checkpoint_path,
                self.log.project_name,
                f"version{self.version:03d}",
                site,
            ),
            filename=os.path.join(f"model"),
            monitor=f"{site.upper()}/Accuracy/val",
            mode="max",
            verbose=False,
            save_top_k=1,
        )

        callbacks = dict(
            filter(lambda item: item[0].endswith("callback"), vars().items())
        ).values()
        callbacks = list(callbacks)
        return callbacks if len(callbacks) > 0 else None

    def run(self, profiler: Optional[str] = None):
        os.makedirs(
            os.path.join(self.log.checkpoint_path, self.log.project_name), exist_ok=True
        )
        self.version = len(
            os.listdir(os.path.join(self.log.checkpoint_path, self.log.project_name))
        )

        # TODO: extract to function
        if self.data.get("roi", None) is None:
            self.data.roi = list(range(116))
        else:
            with open('roi_rank.pkl', 'rb') as f:
                self.data.roi = pickle.load(f)[:int(self.data.roi)+1]

        self.network.roi_rank = self.data.roi
        print("ROI = {}".format(self.data.roi))

        # # nyu, kki, peking, ohsu, ni
        # path = glob(os.path.join(self.data.path, "*.pickle"))
        # # nyu, peking, ohsu, kki, ni
        # path[1:4] = path[2], path[3], path[1]
        # SITES = ["Peking", "KKI", "NI", "NYU", "OHSU"]

        
        # generate train_test column
        record_train_test(df_path='Data/nitrc_niak/master_df.csv')

        final_results = list()
        for i in [5, 1, 6, 3, 4]:
            train_site = deepcopy(list(SITES_DICT.keys()))
            test_site = train_site.pop(train_site.index(i))
            self.data.train_site = train_site
            self.data.test_site = test_site
            site_str = SITES_DICT[i]

            dm = self.get_datamodule(
                dataset=ROIDataset, datamodule=OneSiteHoldoutDataModule
            )
            model = self.get_network(Task=ClassificationTask)
            model.apply(initialize_weights)
            model.prefix = site_str.upper() + "/"

            trainer = Trainer(
                logger=[
                    TensorBoardLogger(
                        save_dir=self.log.log_path,
                        name=os.path.join(
                            self.log.project_name,
                            f"version{self.version:03d}",
                            site_str,
                        ),
                        default_hp_metric=False,
                        version=None,
                        # log_graph=True, # inavailable due to bug
                    ),
                    WandbLogger(
                        project=self.log.project_name,
                        save_dir=self.log.log_path,
                    ),
                ],
                # ! use all gpu
                # gpus=-1,
                # auto_select_gpus=True,
                # ! use 2 gpu
                # devices=2,
                # accelerator="auto",
                # strategy="ddp",
                # ! use gpu 0
                # devices=[1],
                # accelerator="gpu",
                devices=[self.log.device.gpu],
                accelerator="gpu",
                check_val_every_n_epoch=self.log.val_log_freq_epoch,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                max_epochs=self.log.epoch,
                profiler=profiler,
                fast_dev_run=self.log.dry_run,
                callbacks=[
                    *self.get_callbacks(site=site_str),
                    wbc.WatchModel(),
                    wbc.LogConfusionMatrix(),
                    wbc.LogF1PrecRecHeatmap(),
                    # tbc.WatchModel(),
                    # tbc.LogConfusionMatrix(),
                    # tbc.LogF1PrecRecHeatmap(),
                ],
                precision=self.log.precision,
                # gradient_clip_val=0.5,
            )
            trainer.test_site_prefix = model.prefix

            trainer.fit(model, datamodule=dm)
            trainer.test(model, datamodule=dm, ckpt_path="best")
            final_results.append(
                trainer.callback_metrics[f"{model.prefix}Accuracy/test"]
            )

        return


class LOSO_Runner(Base_Runner):
    def get_callbacks(self, site: str):
        """
        Write only callbacks that logger is not necessary
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                self.log.checkpoint_path,
                self.log.project_name,
                f"version{self.version:03d}",
                site,
            ),
            filename=os.path.join(f"model"),
            monitor=f"{site.upper()}/Accuracy/val",
            mode="max",
            verbose=False,
            save_top_k=1,
        )

        callbacks = dict(
            filter(lambda item: item[0].endswith("callback"), vars().items())
        ).values()
        callbacks = list(callbacks)
        return callbacks if len(callbacks) > 0 else None

    def run(self, profiler: Optional[str] = None):
        os.makedirs(
            os.path.join(self.log.checkpoint_path, self.log.project_name), exist_ok=True
        )
        self.version = len(
            os.listdir(os.path.join(self.log.checkpoint_path, self.log.project_name))
        )

        # TODO: extract to function
        if self.data.get("roi", None) is None:
            self.data.roi = list(range(116))
        else:
            if "roi_rank" in self.log.project_name:
                with open("Data/nitrc_niak/roi_rank.pkl", "rb") as f:
                    self.data.roi = pickle.load(f)[: int(self.data.roi)]

            else:
                self.data.roi = [int(self.data.roi)]
        self.network.roi_rank = self.data.roi
        print("ROI = {}".format(self.data.roi))

        # # nyu, kki, peking, ohsu, ni
        # path = glob(os.path.join(self.data.path, "*.pickle"))
        # # nyu, peking, ohsu, kki, ni
        # path[1:4] = path[2], path[3], path[1]
        # SITES = ["Peking", "KKI", "NI", "NYU", "OHSU"]

        final_results = list()
        for i in [5, 1, 6, 3, 4]:
            train_site = deepcopy(list(SITES_DICT.keys()))
            test_site = train_site.pop(train_site.index(i))
            self.data.train_site = train_site
            self.data.test_site = test_site
            site_str = SITES_DICT[i]

            dm = self.get_datamodule(dataset=ROIDataset, datamodule=LOSODataModule)
            model = self.get_network(Task=ClassificationTask)
            model.apply(initialize_weights)
            model.prefix = site_str.upper() + "/"

            trainer = Trainer(
                logger=[
                    TensorBoardLogger(
                        save_dir=self.log.log_path,
                        name=os.path.join(
                            self.log.project_name,
                            f"version{self.version:03d}",
                            site_str,
                        ),
                        default_hp_metric=False,
                        version=None,
                        # log_graph=True, # inavailable due to bug
                    ),
                    WandbLogger(
                        project=self.log.project_name,
                        save_dir=self.log.log_path,
                    ),
                ],
                # ! use all gpu
                # gpus=-1,
                # auto_select_gpus=True,
                # ! use 2 gpu
                # devices=2,
                # accelerator="auto",
                # strategy="ddp",
                # ! use gpu 0
                # devices=[1],
                # accelerator="gpu",
                devices=[self.log.device.gpu],
                accelerator="gpu",
                check_val_every_n_epoch=self.log.val_log_freq_epoch,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                max_epochs=self.log.epoch,
                profiler=profiler,
                fast_dev_run=self.log.dry_run,
                callbacks=[
                    *self.get_callbacks(site=site_str),
                    wbc.WatchModel(),
                    wbc.LogConfusionMatrix(),
                    wbc.LogF1PrecRecHeatmap(),
                    # tbc.WatchModel(),
                    # tbc.LogConfusionMatrix(),
                    # tbc.LogF1PrecRecHeatmap(),
                ],
                precision=self.log.precision,
                # gradient_clip_val=0.5,
            )
            trainer.test_site_prefix = model.prefix

            trainer.fit(model, datamodule=dm)
            trainer.test(model, datamodule=dm, ckpt_path="best")
            final_results.append(
                trainer.callback_metrics[f"{model.prefix}Accuracy/test"]
            )

        try:
            import wandb

            wb_logger = wbc.get_wandb_logger(trainer)
            wb_logger.experiment.log(
                {"overall_accuracy": torch.Tensor(final_results).mean().item()}
            )
            wb_logger.experiment.log(
                {
                    "overall_accuracy_image": wandb.Image(
                        plot_paper(
                            results=final_results,
                            path=os.path.join(
                                self.log.log_path,
                                self.log.project_name,
                                f"version{self.version:03d}",
                                "Accuracy.png",
                            ),
                        )
                    )
                }
            )
        except Exception as e:
            print(e)

        return
