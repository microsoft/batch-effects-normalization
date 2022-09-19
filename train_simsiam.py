from typing import List, Dict
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from wilds.common.metrics.all_metrics import Accuracy
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time

from biomass.utils import compute_grad_norm
from train_erm import compute_accuracies

import math


@hydra.main(version_base=None, config_path="configs", config_name="train_simsiam")
def train_simsiam(config: DictConfig) -> None:
    "Training function for empirical risk minimization on a supervised classification dataset."

    writer = SummaryWriter()

    dataloaders = instantiate(config.dataloaders)
    train_loader_cont = dataloaders.get_train_loader(config.train_batch_size)
    train_loader_eval = dataloaders.get_train_loader(
        config.eval_batch_size, use_eval_transform=True
    )
    eval_loaders = dataloaders.get_eval_loaders(config.eval_batch_size)

    model = instantiate(config.model).to(config.device)

    base_lr = config.learning_rate * config.train_batch_size / 256

    optim_params = [
        {"params": model.encoder.parameters(), "fix_lr": False},
        {"params": model.projector.parameters(), "fix_lr": False},
        {"params": model.predictor.parameters(), "fix_lr": True},
    ]
    optimizer = torch.optim.SGD(
        optim_params,
        base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda()

    print("Completed Setup")

    # Log evaluation accuracies
    model.eval()
    eval_knn(
        model,
        train_loader_eval,
        eval_loaders,
        config.knn_neighbors,
        config.device,
        0,
        writer,
    )

    ### TESTING
    # iterator = iter(train_loader_cont)
    # for _ in range(100):
    #     t = time.time()
    #     batch = next(iterator)
    #     images, labels, metadata = batch
    #     images1 = images[0].to(config.device)
    #     images2 = images[1].to(config.device)
    #     labels = labels.to(config.device)
    #     print(time.time() - t)

    for epoch in range(config.num_epochs):

        # Training
        model.train()
        adjust_learning_rate(optimizer, base_lr, epoch, config.num_epochs)
        for batch_id, (images, labels, metadata) in enumerate(train_loader_cont):

            optimizer.zero_grad()

            images1 = images[0].to(config.device)
            images2 = images[1].to(config.device)

            p, z = model(torch.cat([images1, images2], dim=0))
            p1, p2 = torch.chunk(p, 2)
            z1, z2 = torch.chunk(z, 2)

            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            loss.backward()

            optimizer.step()

            global_step = epoch * len(train_loader_cont) + batch_id
            writer.add_scalar("Training/loss", loss, global_step)

            grad_norm = compute_grad_norm(model)
            writer.add_scalar("Training/grad-norm", grad_norm, global_step)

            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Training/lr", lr, global_step)

        # Evaluation
        model.eval()
        eval_knn(
            model,
            train_loader_eval,
            eval_loaders,
            config.knn_neighbors,
            config.device,
            epoch + 1,
            writer,
        )

        if config.save_model and epoch % 10 == 9:
            dt = HydraConfig.get().run.dir[8:].replace("-", "_").replace("/", "_")
            torch.save(model.state_dict(), f"checkpoints/simsiam_{dt}.pt")


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        if "fix_lr" in param_group and param_group["fix_lr"]:
            param_group["lr"] = init_lr
        else:
            param_group["lr"] = cur_lr


def eval_knn(
    model: Module,
    train_loader_eval: DataLoader,
    eval_loaders: Dict[str, DataLoader],
    knn_neighbors: List[int],
    device: str,
    epoch: int,
    writer: SummaryWriter,
):
    # Collect all features in training set
    with torch.no_grad():
        t = time.time()
        all_feats_train = []
        all_labels_train = []
        for images, labels, metadata in train_loader_eval:
            images = images.to(device)
            feats = model.encode(images)
            all_feats_train.append(feats.to("cpu"))
            all_labels_train.append(labels.to("cpu"))
        feats_train = torch.cat(all_feats_train).numpy()
        labels_train = torch.cat(all_labels_train).numpy()

        # Standardize features
        feats_means = 0  # feats_train.mean(axis=0)
        feats_stds = 1  # feats_train.std(axis=0)
        feats_train = (feats_train - feats_means) / feats_stds

        print("Train feature time", time.time() - t)

        writer.add_images("Training/images", images.to("cpu"), epoch)

        t = time.time()
        eval_sets = {}
        for key, eval_loader in eval_loaders.items():
            all_feats_eval = []
            all_labels_eval = []
            for images, labels, metadata in eval_loader:
                images = images.to(device)
                feats = model.encode(images)
                all_feats_eval.append(feats.to("cpu"))
                all_labels_eval.append(labels.to("cpu"))
            eval_sets[key] = (
                (torch.cat(all_feats_eval).numpy() - feats_means) / feats_stds,
                torch.cat(all_labels_eval).numpy(),
            )
        print("Test feature time", time.time() - t)

        writer.add_images("Evaluation/images", images.to("cpu"), epoch)

    t = time.time()
    # Run KNN on latent space
    for neighbors in knn_neighbors:
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(feats_train, labels_train)

        preds_train = knn.predict(feats_train)
        acc = np.mean(labels_train == preds_train)
        writer.add_scalar(f"Evaluation/train_neighbors={neighbors}", acc, epoch)

        for key, (feats_eval, labels_eval) in eval_sets.items():
            preds_eval = knn.predict(feats_eval)
            acc = np.mean(labels_eval == preds_eval)
            writer.add_scalar(f"Evaluation/{key}_neighbors={neighbors}", acc, epoch)
    print("KNN time", time.time() - t)


if __name__ == "__main__":
    train_simsiam()
