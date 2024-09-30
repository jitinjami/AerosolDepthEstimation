import argparse
import warnings

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from aode.dataset import SolaDataset
from aode.evaluate import evaluate
from aode.manager import RunManager
from aode.models import BaseModel
from aode.utils import seed_everything, seed_worker

warnings.filterwarnings('ignore', category=UserWarning)


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    fn_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.SequentialLR,
    man: RunManager,
    device: str,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader of training set.
        model (torch.nn.Module): Model.
        fn_loss (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.SequentialLR): Learning rate
        scheduler.
        man (hwr.manager.RunManager): Running manager.
        device (str): Device to use.
        epoch (int): Current epoch number.
    '''
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    for idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = fn_loss(out, y)
        loss.backward()
        optimizer.update()
        lr_scheduler.step()
        man.update_iteration(idx, loss.item(), lr_scheduler.get_last_lr()[0])

    man.summarize_epoch()

    # save checkpoints every freq_save epoch
    if man.check_step(epoch + 1, man.freq_save, man.epoch_max):
        man.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            lr_scheduler.state_dict(),
        )


def test(
    dataloader: DataLoader,
    model: nn.Module,
    fn_loss: nn.Module,
    man: RunManager,
    device: str,
    epoch: int = None,
) -> None:
    '''Test the model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader of testing or
        validation set.
        model (torch.nn.Module): Model.
        fn_loss (torch.nn.Module): Loss function.
        man (hwr.manager.RunManager): Running manager.
        categories (list[str]): Category infomation for evaluation.
        device (str): Device to use.
        epoch (int, optional): Epoch number. Defaults to None.
    '''
    preds = []  # predictions for evaluation
    labels = []  # labels for evaluation
    man.initialize_epoch(epoch, len(dataloader), True)
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = fn_loss(out, y)
            man.update_iteration(idx, loss.item())

            # decode and cache results every freq_eval epoch
            if man.check_step(epoch + 1, man.freq_eval, man.epoch_max):
                for pred, label in zip(out.cpu(), y.cpu()):
                    preds.append(pred)
                    labels.append(label)

    man.summarize_epoch()

    # evaluate every freq_eval epoch
    if man.check_step(epoch + 1, man.freq_eval, man.epoch_max):
        # TODO: Get evaluate function ready.
        pass
        # results_eval = evaluate(preds=preds, labels=labels)
        # man.update_evaluation(results_eval, preds[:20], labels[:20])


def main(cfgs: argparse.Namespace) -> None:
    '''Main function for training and evaluation.

    Args:
        cfgs (argparse.Namespace): Configurations.
    '''
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)

    dataset_test = SolaDataset(
        cfgs.path_csv_test,
        cfgs.dir_test,
        True,
        cfgs.mean,
        cfgs.std,
        cfgs.cache,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=cfgs.size_batch,
        num_workers=cfgs.num_worker,
    )
    fn_loss = nn.MSELoss()
    model = BaseModel(cfgs.backbone).to(cfgs.device)
    epoch_start = 0

    if not cfgs.test:
        dataset_train = SolaDataset(
            cfgs.path_csv_train,
            cfgs.dir_train,
            True,
            cfgs.mean,
            cfgs.std,
            cfgs.cache,
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=cfgs.size_batch,
            shuffle=True,
            num_workers=cfgs.num_worker,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfgs.seed),
        )
        optimizer = torch.optim.Adam(model.parameters(), cfgs.lr)
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(
                    optimizer,
                    start_factor=0.01,
                    total_iters=len(dataloader_train) * cfgs.epoch_warmup,
                ),
                CosineAnnealingLR(
                    optimizer,
                    len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup),
                ),
            ],
            [len(dataloader_train) * cfgs.epoch_warmup],
        )

    # load checkpoint if given
    if cfgs.checkpoint:
        ckp = torch.load(cfgs.checkpoint)
        model.load_state_dict(ckp['model'])

        if not cfgs.test:
            epoch_start = ckp['epoch'] + 1
            optimizer.load_state_dict(ckp['optimizer'])
            lr_scheduler.load_state_dict(ckp['lr_scheduler'])

        manager.log(f'Load checkpoint from {cfgs.checkpoint}')

    # start running
    for e in range(epoch_start, cfgs.epoch):
        if cfgs.test:
            test(
                dataloader=dataloader_test,
                model=model,
                fn_loss=fn_loss,
                man=manager,
                device=cfgs.device,
                epoch=-1,
            )
            break
        else:
            train_one_epoch(
                dataloader=dataloader_train,
                model=model,
                fn_loss=fn_loss,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                man=manager,
                device=cfgs.device,
                epoch=e,
            )
            test(
                dataloader=dataloader_test,
                model=model,
                fn_loss=fn_loss,
                man=manager,
                device=cfgs.device,
                epoch=e,
            )

    if not cfgs.test:
        manager.summarize_evaluation()


if __name__ == '__main__':
    path_config = 'configs/train.yaml'

    with open(path_config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
