from typing import List, Dict
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, SGD
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
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_


from biomass.utils import compute_grad_norm
from train_erm import compute_accuracies


@hydra.main(
    version_base=None, config_path="configs", config_name="train_classifier_cell"
)
def train_classifier_cell(config: DictConfig) -> None:

    writer = SummaryWriter()

    dataloaders = instantiate(config.dataloaders)
    train_loader = dataloaders.get_train_loader(config.train_batch_size)
    eval_loaders = dataloaders.get_eval_loaders(config.eval_batch_size)

    model = instantiate(config.model)
    model.load_state_dict(torch.load(config.model_path))
    model = model.to(config.device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    cross_entropy = CrossEntropyLoss()
    classifier = torch.nn.Linear(config.in_features, config.out_features).to(
        config.device
    )
    optimizer = Adam(
        list(classifier.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    print("Completed Setup")

    # Log evaluation accuracies
    model.eval()
    classifier.eval()
    losses, cell_accs, img_accs = eval_model(
        model,
        classifier,
        eval_loaders,
        cross_entropy,
        config.device,
        config.use_projs,
        config.normalize,
    )
    for key, val in losses.items():
        writer.add_scalar(f"Validation/cell_loss_{key}", val, 0)
    for key, val in cell_accs.items():
        writer.add_scalar(f"Validation/cell_acc_{key}", val, 0)
    for key, val in img_accs.items():
        writer.add_scalar(f"Validation/img_acc_{key}", val, 0)
    print(cell_accs, img_accs)

    for epoch in range(config.num_epochs):

        # Training
        classifier.train()
        for batch_id, (images, labels, lens) in enumerate(train_loader):

            optimizer.zero_grad()

            with torch.no_grad():
                images = images.to(config.device)
                labels = labels.to(config.device)
                lens = lens.to(config.device)

                if config.use_projs:
                    feats = model.project(images)
                else:
                    feats = model.encode(images)

                if config.normalize:
                    feats = F.normalize(feats, dim=1)

                cell_labels = torch.repeat_interleave(labels, lens)
                assert len(feats) == len(cell_labels)

            preds = classifier(feats)
            loss = cross_entropy(preds, cell_labels)

            loss.backward()

            grad_norm = compute_grad_norm(classifier)

            if config.clip_grad_norm is not None:
                clip_grad_norm_(classifier.parameters(), config.clip_grad_norm)

            optimizer.step()

            with torch.no_grad():
                cell_preds = torch.argmax(preds, dim=1)
                acc = torch.sum(cell_preds == cell_labels) / len(cell_labels)

                # Aggregate cell predictions by majority vote
                img_preds_sets = torch.split(cell_preds, lens.cpu().numpy().tolist())
                img_preds = torch.stack(
                    [torch.mode(ips).values for ips in img_preds_sets], dim=0
                )
                assert len(img_preds) == len(labels)
                img_acc = torch.sum(img_preds == labels) / len(labels)

            global_step = epoch * len(train_loader) + batch_id
            writer.add_scalar("Training/cell_loss", loss, global_step)
            writer.add_scalar("Training/cell_acc", acc, global_step)
            writer.add_scalar("Training/img_acc", img_acc, global_step)

            writer.add_scalar("Training/grad-norm", grad_norm, global_step)

            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Training/lr", lr, global_step)

        # Evaluation
        classifier.eval()
        losses, cell_accs, img_accs = eval_model(
            model,
            classifier,
            eval_loaders,
            cross_entropy,
            config.device,
            config.use_projs,
            config.normalize,
        )
        for key, val in losses.items():
            writer.add_scalar(f"Validation/cell_loss_{key}", val, epoch + 1)
        for key, val in cell_accs.items():
            writer.add_scalar(f"Validation/cell_acc_{key}", val, epoch + 1)
        for key, val in img_accs.items():
            writer.add_scalar(f"Validation/img_acc_{key}", val, epoch + 1)

        print(cell_accs, img_accs)

        if config.save_model and epoch % 10 == 9:
            dt = HydraConfig.get().run.dir[8:].replace("-", "_").replace("/", "_")
            torch.save(classifier.state_dict(), f"checkpoints/cell_classifier_{dt}.pt")


@torch.no_grad()
def eval_model(
    model: nn.Module,
    classifier: nn.Module,
    eval_loaders: Dict[str, DataLoader],
    loss_func: nn.Module,
    device: str,
    use_projs: bool,
    normalize: bool,
):
    global_losses = {}
    global_cell_accs = {}
    global_img_accs = {}
    for key, dataloader in eval_loaders.items():
        all_losses = []
        all_cell_accs = []
        all_img_accs = []
        all_cell_sizes = []
        all_img_sizes = []

        for batch_id, (images, labels, lens) in enumerate(dataloader):

            images = images.to(device)
            labels = labels.to(device)
            lens = lens.to(device)

            if use_projs:
                feats = model.project(images)
            else:
                feats = model.encode(images)

            if normalize:
                feats = F.normalize(feats, dim=1)

            # Max pooling
            cell_labels = torch.repeat_interleave(labels, lens)
            assert len(feats) == len(cell_labels)

            preds = classifier(feats)
            loss = loss_func(preds, cell_labels)

            cell_preds = torch.argmax(preds, dim=1)
            cell_acc = torch.sum(cell_preds == cell_labels) / len(cell_labels)

            # Aggregate cell predictions by majority vote
            img_preds_sets = torch.split(cell_preds, lens.cpu().numpy().tolist())
            img_preds = torch.stack(
                [torch.mode(ips).values for ips in img_preds_sets], dim=0
            )
            assert len(img_preds) == len(labels)
            img_acc = torch.sum(img_preds == labels) / len(labels)

            all_losses.append(loss)
            all_cell_accs.append(cell_acc)
            all_img_accs.append(img_acc)
            all_cell_sizes.append(torch.tensor(len(cell_preds), device=device))
            all_img_sizes.append(torch.tensor(len(img_preds), device=device))

        losses = torch.stack(all_losses)
        cell_accs = torch.stack(all_cell_accs)
        img_accs = torch.stack(all_img_accs)
        cell_sizes = torch.stack(all_cell_sizes)
        img_sizes = torch.stack(all_img_sizes)

        global_losses[key] = torch.sum(losses * cell_sizes) / torch.sum(cell_sizes)
        global_cell_accs[key] = torch.sum(cell_accs * cell_sizes) / torch.sum(
            cell_sizes
        )
        global_img_accs[key] = torch.sum(img_accs * img_sizes) / torch.sum(img_sizes)
    return global_losses, global_cell_accs, global_img_accs


if __name__ == "__main__":
    train_classifier_cell()
