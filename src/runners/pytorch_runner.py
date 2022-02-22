import os
from glob import glob
from typing import Optional
from copy import deepcopy

from src import models
from src.data import ROIDataset, SITES_DICT
from src.datamodules import LOSODataModule
from src.tasks import ClassificationTask
from src.utils import plot_paper
from src.callbacks import wandb_callback as wbc
from src.callbacks import tensorboard_callback as tbc

import torch
from torch import nn, optim
import torch.nn.functional as F
from pytorch_lightning import seed_everything

seed_everything(41)


def get_weights(dataset):
    class_label, class_counts = torch.unique(
        torch.Tensor(dataset.labels), sorted=True, return_counts=True
    )
    num_data = len(dataset)
    weights = 1.0 / (class_counts.type(torch.float) / num_data * 100)
    return weights


def train(model, loader, optimizer, criterion, device, epoch, log_interval=10):
    train_loss = 0
    correct = 0
    count = 0

    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        count += len(target)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Acc: {:.2f}%".format(
                    epoch,
                    count,
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.item(),
                    100.0 * correct / count,
                )
            )

    train_loss /= len(loader.dataset)
    train_acc = 100.0 * correct / len(loader.dataset)
    return train_loss, train_acc


def evaluate(model, loader, criterion, device):
    eval_loss = 0
    correct = 0
    count = 0

    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        eval_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        count += len(target)

    eval_loss /= len(loader.dataset)
    eval_acc = 100.0 * correct / len(loader.dataset)
    return eval_loss, eval_acc


def main_pytorch(log, optimizer, loader, network, data):
    device = torch.device(
        f"cuda:{log.device.gpu[0]}" if torch.cuda.is_available() else "cpu"
    )

    final_results = list()
    for i in [5, 1, 6, 3, 4]:
        train_site = deepcopy(list(SITES_DICT.keys()))
        test_site = train_site.pop(train_site.index(i))
        data.train_site = train_site
        data.test_site = test_site
        site_str = SITES_DICT[i]

        dm = LOSODataModule(data, loader, ROIDataset)
        dm.setup()
        model = getattr(models, network.model)(network).to(device)

        criterion = nn.CrossEntropyLoss(get_weights(dm.train_dataset)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=optimizer.lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        for epoch in range(log.epoch):
            train_loss, train_acc = train(model, dm.train_dataloader(), optimizer, criterion, device, epoch)
            val_loss, val_acc = evaluate(model, dm.val_dataloader(), criterion, device)
            print(f'VAL loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
            scheduler.step()

        torch.cuda.empty_cache()
        test_loss, test_acc = evaluate(model, dm.test_dataloader(), criterion, device)
        print(test_loss, test_acc)
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":

    main_pytorch()
