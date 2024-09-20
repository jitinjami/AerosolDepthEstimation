import argparse
import warnings

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from aode.dataset import SolaTestDataset, SolaTrainDataset
from aode.models import BaseModel
from aode.models.swin import SwinTransformerV2
from aode.utils import seed_everything, seed_worker

warnings.filterwarnings('ignore', category=UserWarning)


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.SequentialLR,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader of training set.
        model (torch.nn.Module): Model.
        optimizer (torch.optim.Optimizer): Optimizer.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler.
        lr_scheduler (torch.optim.lr_scheduler.SequentialLR): Learning rate scheduler.
        epoch (int): Current epoch number.
    '''
    model.train()


def main(cfgs: argparse.Namespace) -> None:
    '''Main function for training and evaluation.

    Args:
        cfgs (argparse.Namespace): Configurations.
    '''
    seed_everything(cfgs.seed)
    

    train_dataset = SolaTrainDataset(
        train_csv, train_img_dir, cache=True, transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(train_dataset.means)


if __name__ == '__main__':
    path_config = 'configs/train.yaml'

    with open(path_config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
