from typing import List, Dict, Any
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from wilds.common.metrics.all_metrics import Accuracy
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
import torch.nn.functional as F
import gc
from torch import Tensor

from biomass.utils import compute_grad_norm
from train_erm import compute_accuracies
from torch.nn.utils import clip_grad_norm_


@hydra.main(
    version_base=None, config_path="configs", config_name="train_supervised_cell"
)
def train_supervised_cell(config: DictConfig) -> None:

    writer = SummaryWriter()

    dataloaders = instantiate(config.dataloaders)
    train_loader = dataloaders.get_train_loader(config.train_batch_size)
    eval_loaders = dataloaders.get_eval_loaders(config.eval_batch_size)

    model = instantiate(config.model)
    model = model.to(config.device)
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=len(train_loader) * config.num_epochs,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
    )

    celoss = CrossEntropyLoss()

    print("Completed Setup")

    # Log evaluation accuracies
    if not config.no_eval:
        model.eval()
    losses, accs = eval_model(model, eval_loaders, celoss, config)
    for key, val in losses.items():
        writer.add_scalar(f"Validation/loss_{key}", val, 0)
    for key, val in accs.items():
        writer.add_scalar(f"Validation/acc_{key}", val, 0)

    for epoch in range(config.num_epochs):

        # Training
        t = time.time()
        model.train()
        for batch_id, (images, labels, lens, plates) in enumerate(train_loader):

            optimizer.zero_grad()

            images = images.to(config.device)
            labels = labels.to(config.device)
            lens = lens.to(config.device)

            preds = model(images)
            labels = torch.repeat_interleave(labels, lens)

            loss = celoss(preds, labels)

            loss.backward()

            acc = torch.sum(preds.argmax(dim=1) == labels) / len(labels)

            if config.clip_grad_norm is not None:
                clip_grad_norm_(model.parameters(), config.clip_grad_norm)

            optimizer.step()

            global_step = epoch * len(train_loader) + batch_id
            writer.add_scalar("Training/loss", loss.cpu(), global_step)
            writer.add_scalar("Training/acc", acc.cpu(), global_step)

            grad_norm = compute_grad_norm(model)
            writer.add_scalar("Training/grad-norm", grad_norm, global_step)

            lr = scheduler.get_last_lr()[0]
            writer.add_scalar("Training/lr", lr, global_step)
            scheduler.step()

        # Evaluation
        if not config.no_eval:
            model.eval()
        losses, accs = eval_model(model, eval_loaders, celoss, config)
        for key, val in losses.items():
            writer.add_scalar(f"Validation/loss_{key}", val, epoch + 1)
        for key, val in accs.items():
            writer.add_scalar(f"Validation/acc_{key}", val, epoch + 1)
        print(accs)

        if config.save_model and epoch % 10 == 9:
            dt = HydraConfig.get().run.dir[8:].replace("-", "_").replace("/", "_")
            torch.save(model.state_dict(), f"checkpoints/supervised_cell_{dt}.pt")

        print("TIME", time.time() - t)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    eval_loaders: Dict[str, DataLoader],
    loss_func: nn.Module,
    config: Dict[str, Any],
):
    global_losses = {}
    global_accs = {}
    for key, dataloader in eval_loaders.items():
        losses = []
        accs = []
        for batch_id, (images, labels, lens, plates) in enumerate(dataloader):

            images = images.to(config.device)
            labels = labels.to(config.device)
            lens = lens.to(config.device)

            preds = model(images)
            labels = torch.repeat_interleave(labels, lens)

            loss = loss_func(preds, labels)
            acc = torch.sum(preds.argmax(dim=1) == labels) / len(labels)

            losses.append(loss.cpu())
            accs.append(acc.cpu())

        global_losses[key] = torch.mean(torch.stack(losses))
        global_accs[key] = torch.mean(torch.stack(accs))
    return global_losses, global_accs


if __name__ == "__main__":
    train_supervised_cell()
