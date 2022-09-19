from typing import List, Dict, Any
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
from mahotas.features import haralick


from biomass.utils import compute_grad_norm
from train_erm import compute_accuracies


@hydra.main(
    version_base=None, config_path="configs", config_name="train_classifier_texture"
)
def train_classifier_texture(config: DictConfig) -> None:

    writer = SummaryWriter()

    dataloaders = instantiate(config.dataloaders)
    train_loader = dataloaders.get_train_loader(config.train_batch_size)
    eval_loaders = dataloaders.get_eval_loaders(config.eval_batch_size)

    cross_entropy = CrossEntropyLoss()
    pooler = instantiate(config.pooler).to(config.device)
    classifier = torch.nn.Linear(config.in_features, config.out_features).to(
        config.device
    )
    if config.optimizer == "adam":
        optimizer = Adam(
            list(pooler.parameters()) + list(classifier.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = SGD(
            list(pooler.parameters()) + list(classifier.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    else:
        raise ValueError()
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    print("Completed Setup")

    # Log evaluation accuracies
    pooler.eval()
    classifier.eval()
    losses, accs, top5_accs = eval_model(
        pooler,
        classifier,
        eval_loaders,
        cross_entropy,
        config.device,
        config.normalize,
        config,
    )
    for key, val in losses.items():
        writer.add_scalar(f"Validation/loss_{key}", val, 0)
    for key, val in accs.items():
        writer.add_scalar(f"Validation/acc_{key}", val, 0)
    for key, val in top5_accs.items():
        writer.add_scalar(f"Validation/top5_acc_{key}", val, 0)

    best_acc = accs[config.save_key]

    for epoch in range(config.num_epochs):

        train_losses = []
        train_accs = []

        # Training
        classifier.train()
        pooler.train()
        for batch_id, (images, labels, lens) in enumerate(train_loader):

            optimizer.zero_grad()

            with torch.no_grad():
                labels = labels.to(config.device)

                feats = torch.stack(
                    [
                        torch.cat(
                            [
                                torch.tensor(
                                    haralick(img_ch.numpy()).mean(axis=0),
                                    device=config.device,
                                )
                                for img_ch in img
                            ]
                        )
                        for img in images
                    ]
                )

                if config.normalize:
                    feats = F.normalize(feats, dim=1)

                feats_sets = torch.split(feats, lens.numpy().tolist())

                # Pooling
                feats = torch.stack(
                    [pooler(feats_set) for feats_set in feats_sets],
                    dim=0,
                )

                if config.normalize_img_feats:
                    mu = feats.mean(dim=0, keepdim=True)
                    sigma = feats.std(dim=0, keepdim=True)
                    sigma[sigma == 0.0] = 1.0
                    feats = (feats - mu) / sigma

                assert len(feats) == len(labels)

            preds = classifier(feats)
            loss = cross_entropy(preds, labels)
            top1_preds = torch.argmax(preds, dim=1)
            top5_preds = torch.topk(preds, k=5, dim=1).indices
            acc = torch.sum(top1_preds == labels) / len(labels)
            top5_acc = torch.sum(top5_preds == labels.view(-1, 1)) / len(labels)

            loss.backward()

            optimizer.step()

            train_losses.append(loss.detach().cpu().numpy())
            train_accs.append(acc.cpu().numpy())

            grad_norm = compute_grad_norm(classifier)

            if config.clip_grad_norm is not None:
                clip_grad_norm_(classifier.parameters(), config.clip_grad_norm)

            if batch_id % 1 == 0:
                global_step = epoch * len(train_loader) + batch_id
                writer.add_scalar("Training/loss", np.mean(train_losses), global_step)
                writer.add_scalar("Training/acc", np.mean(train_accs), global_step)
                writer.add_scalar("Training/top5_acc", top5_acc, global_step)
                train_losses = []
                train_accs = []

                writer.add_scalar("Training/grad-norm", grad_norm, global_step)

                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Training/lr", lr, global_step)

        # Evaluation
        classifier.eval()
        pooler.eval()
        losses, accs, top5_accs = eval_model(
            pooler,
            classifier,
            eval_loaders,
            cross_entropy,
            config.device,
            config.normalize,
            config,
        )
        for key, val in losses.items():
            writer.add_scalar(f"Validation/loss_{key}", val, epoch + 1)
        for key, val in accs.items():
            writer.add_scalar(f"Validation/acc_{key}", val, epoch + 1)
        for key, val in top5_accs.items():
            writer.add_scalar(f"Validation/top5_acc_{key}", val, epoch + 1)

        print(accs)

        if config.save_model and accs[config.save_key] > best_acc:
            dt = HydraConfig.get().run.dir[8:].replace("-", "_").replace("/", "_")
            torch.save(
                classifier.state_dict(), f"checkpoints/classifier_texture_{dt}.pt"
            )
            best_acc = accs[config.save_key]


def imgs_to_feats(imgs):
    all_feats = []
    for img in imgs:
        feats = []
        for img_ch in img:
            feats.append(torch.tensor(haralick(img_ch.numpy()).mean(axis=0)))
        all_feats.append(torch.cat(feats))
    return torch.stack(all_feats)


@torch.no_grad()
def eval_model(
    pooler: nn.Module,
    classifier: nn.Module,
    eval_loaders: Dict[str, DataLoader],
    loss_func: nn.Module,
    device: str,
    normalize: bool,
    config: Dict[str, Any],
):
    global_losses = {}
    global_accs = {}
    global_top5_accs = {}
    for key, dataloader in eval_loaders.items():
        all_preds = []
        all_labels = []
        for batch_id, (images, labels, lens) in enumerate(dataloader):

            t = time.time()
            feats = imgs_to_feats(images)
            print("Feats time", time.time() - t)

            if normalize:
                feats = F.normalize(feats, dim=1)

            feats_sets = torch.split(feats, lens.numpy().tolist())

            # Max pooling
            feats = torch.stack([pooler(feats_set) for feats_set in feats_sets], dim=0)
            assert len(feats) == len(labels)

            if config.normalize_img_feats:
                mu = feats.mean(dim=0, keepdim=True)
                sigma = feats.std(dim=0, keepdim=True)
                sigma[sigma == 0.0] = 1.0
                feats = (feats - mu) / sigma

            preds = classifier(feats).to("cpu")

            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        global_losses[key] = loss_func(all_preds, all_labels)
        global_accs[key] = torch.sum(all_preds.argmax(dim=1) == all_labels) / len(
            all_labels
        )
        global_top5_accs[key] = torch.sum(
            torch.topk(all_preds, k=5, dim=1).indices == all_labels.view(-1, 1)
        ) / len(all_labels)
    return global_losses, global_accs, global_top5_accs


if __name__ == "__main__":
    train_classifier_texture()
