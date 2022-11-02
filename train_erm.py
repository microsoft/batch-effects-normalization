from typing import Dict
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from wilds.common.metrics.all_metrics import Accuracy
from torch.utils.data import DataLoader
import time

from biomass.utils import compute_grad_norm


@hydra.main(version_base=None, config_path="configs", config_name="train_erm")
def train_erm(config: DictConfig) -> None:
    "Training function for empirical risk minimization on a supervised classification dataset."

    writer = SummaryWriter()

    dataloaders = instantiate(config.dataloaders)
    train_loader = dataloaders.get_train_loader(config.train_batch_size)
    eval_loaders = dataloaders.get_eval_loaders(config.eval_batch_size)

    model = instantiate(config.model).to(config.device)
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=len(train_loader) * config.num_epochs,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
    )

    erm = CrossEntropyLoss()

    print("Completed Setup")

    if config.use_train_at_eval:
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = False
    else:
        model.eval()
    # Log evaluation accuracies
    train_results = compute_accuracies(model, {"train": train_loader}, config.device)
    for metric, val in train_results.items():
        writer.add_scalar(f"Evaluation/{metric}", val, 0)

    eval_results = compute_accuracies(model, eval_loaders, config.device)
    for metric, val in eval_results.items():
        writer.add_scalar(f"Evaluation/{metric}", val, 0)
    print(eval_results)

    for epoch in range(config.num_epochs):

        # Training
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = True
        for batch_id, (images, labels, metadata) in enumerate(train_loader):

            optimizer.zero_grad()

            images = images.to(config.device)
            labels = labels.to(config.device)

            preds = model(images)

            loss = erm(preds, labels)

            loss.backward()

            optimizer.step()

            global_step = epoch * len(train_loader) + batch_id
            writer.add_scalar("Training/loss", loss, global_step)

            grad_norm = compute_grad_norm(model)
            writer.add_scalar("Training/grad-norm", grad_norm, global_step)

            lr = scheduler.get_last_lr()[0]
            writer.add_scalar("Training/lr", lr, global_step)
            scheduler.step()

        # Evaluation
        if config.use_train_at_eval:
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.track_running_stats = False
        else:
            model.eval()
        if config.compute_train_acc:
            train_results = compute_accuracies(
                model, {"train": train_loader}, config.device
            )
            for metric, val in train_results.items():
                writer.add_scalar(f"Evaluation/{metric}", val, epoch + 1)
            print(train_results)

        eval_results = compute_accuracies(model, eval_loaders, config.device)
        for metric, val in eval_results.items():
            writer.add_scalar(f"Evaluation/{metric}", val, epoch + 1)
        print(eval_results)

        if config.save_model and epoch % 10 == 9:
            dt = HydraConfig.get().run.dir[8:].replace("-", "_").replace("/", "_")
            torch.save(model.state_dict(), f"checkpoints/erm_{dt}.pt")


def compute_accuracies(
    model: Module, eval_loaders: Dict[str, DataLoader], device: str
) -> Dict[str, float]:
    with torch.no_grad():
        results = {}
        for split, loader in eval_loaders.items():
            all_preds = []
            all_labels = []
            for images, labels, metadata in loader:
                preds = model(images.to(device)).to("cpu")
                all_preds.append(preds)
                all_labels.append(labels)
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            assert len(all_preds) == len(all_labels)

            acc = Accuracy()
            results[f"{split}_acc"] = acc.compute(all_preds.argmax(dim=-1), all_labels)[
                "acc_avg"
            ]

            erm = CrossEntropyLoss()
            results[f"{split}_loss"] = erm(all_preds, all_labels).numpy().item()

    return results


if __name__ == "__main__":
    train_erm()
