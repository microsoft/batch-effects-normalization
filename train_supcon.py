from typing import List, Dict
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

from biomass.utils import compute_grad_norm
from train_erm import compute_accuracies


@hydra.main(version_base=None, config_path="configs", config_name="train_supcon")
def train_erm(config: DictConfig) -> None:
    "Training function for empirical risk minimization on a supervised classification dataset."

    writer = SummaryWriter()

    dataloaders = instantiate(config.dataloaders)
    train_loader_cont = dataloaders.get_train_loader(config.train_batch_size)
    train_loader_eval = dataloaders.get_train_loader(
        config.eval_batch_size, use_eval_transform=True
    )
    eval_loaders = dataloaders.get_eval_loaders(config.eval_batch_size)

    model = instantiate(config.model).to(config.device)
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=len(train_loader_cont) * config.num_epochs,
        num_warmup_steps=len(train_loader_cont) * config.num_warmup_epochs,
    )

    supcon = SupConLoss(
        temperature=config.temperature,
        base_temperature=config.base_temperature,
        contrast_mode="all",
    )

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

    gc.collect()
    torch.cuda.empty_cache()

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
        for batch_id, (images, labels, metadata) in enumerate(train_loader_cont):

            optimizer.zero_grad()

            images1 = images[0].to(config.device)
            images2 = images[1].to(config.device)
            labels = labels.to(config.device)

            projs = model.project(
                torch.stack([images1, images2], dim=1).flatten(start_dim=0, end_dim=1)
            )
            projs = F.normalize(projs, dim=1)

            projs = projs.unflatten(dim=0, sizes=(len(labels), 2))

            loss, duplicate_labels = supcon(projs, labels)

            loss.backward()

            optimizer.step()

            global_step = epoch * len(train_loader_cont) + batch_id
            writer.add_scalar("Training/loss", loss, global_step)
            writer.add_scalar(
                "Training/duplicate_labels", duplicate_labels, global_step
            )

            grad_norm = compute_grad_norm(model)
            writer.add_scalar("Training/grad-norm", grad_norm, global_step)

            lr = scheduler.get_last_lr()[0]
            writer.add_scalar("Training/lr", lr, global_step)
            scheduler.step()

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

        gc.collect()
        torch.cuda.empty_cache()

        if config.save_model and epoch % 10 == 9:
            dt = HydraConfig.get().run.dir[8:].replace("-", "_").replace("/", "_")
            torch.save(model.state_dict(), f"checkpoints/erm_{dt}.pt")


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
        all_metadata_train = []
        for images, labels, metadata in train_loader_eval:
            images = images.to(device)
            feats = model.encode(images)
            feats = F.normalize(feats, dim=1)
            all_feats_train.append(feats.to("cpu"))
            all_labels_train.append(labels)
            all_metadata_train.append(metadata)
        feats_train = torch.cat(all_feats_train)
        labels_train = torch.cat(all_labels_train)
        metadata_train = torch.cat(all_metadata_train)

        print("Train feature time", time.time() - t)

        # writer.add_images("Training/images", images.to("cpu"), epoch)

        t = time.time()
        eval_sets = {}
        for key, eval_loader in eval_loaders.items():
            all_feats_eval = []
            all_labels_eval = []
            all_metadata_eval = []
            for images, labels, metadata in eval_loader:
                images = images.to(device)
                feats = model.encode(images)
                feats = F.normalize(feats, dim=1)
                all_feats_eval.append(feats.to("cpu"))
                all_labels_eval.append(labels)
                all_metadata_eval.append(metadata)
            eval_sets[key] = (
                torch.cat(all_feats_eval),
                torch.cat(all_labels_eval),
                torch.cat(all_metadata_eval),
            )
        print("Test feature time", time.time() - t)

        # writer.add_images("Evaluation/images", images.to("cpu"), epoch)

        t = time.time()
        # Run KNN on latent space
        for neighbors in knn_neighbors:
            # knn = KNeighborsClassifier(n_neighbors=neighbors, metric="cosine")
            # knn.fit(feats_train, labels_train)

            # preds_train = knn.predict(feats_train)
            num_classes = len(torch.unique(labels_train))
            pred_labels = knn_predict(
                feats_train,
                feats_train,
                labels_train,
                num_classes,
                neighbors,
                ignore_first=True,
            )
            acc = np.mean(labels_train.cpu().numpy() == pred_labels[:, 0].cpu().numpy())
            writer.add_scalar(f"Evaluation/train_neighbors={neighbors}", acc, epoch)

            for key, (feats_eval, labels_eval, metadata_eval) in eval_sets.items():
                # preds_eval = knn.predict(feats_eval)
                pred_labels = knn_predict(
                    feats_eval,
                    feats_train,
                    labels_train,
                    num_classes,
                    neighbors,
                    ignore_first=False,
                )
                acc = np.mean(
                    labels_eval.cpu().numpy() == pred_labels[:, 0].cpu().numpy()
                )
                writer.add_scalar(f"Evaluation/{key}_neighbors={neighbors}", acc, epoch)
        print("KNN time", time.time() - t)

        del (
            all_feats_train,
            all_labels_train,
            all_feats_eval,
            all_labels_eval,
            feats_eval,
            labels_eval,
            pred_labels,
            feats_train,
            labels_train,
        )


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(
    feature, feature_bank, feature_labels, classes, knn_k, knn_t=1, ignore_first=False
):
    if len(feature) > 3000:
        feature_lst = torch.split(feature, 3000)
    else:
        feature_lst = [feature]

    pred_lst = []

    feature_bank = feature_bank.to("cuda")
    feature_labels = feature_labels.to("cuda")

    for feature in feature_lst:

        feature = feature.to("cuda")
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank.T)
        # [B, K]
        if ignore_first:
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k + 1, dim=-1)
            sim_weight = sim_weight[:, 1:]
            sim_indices = sim_indices[:, 1:]
        else:
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(
            feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
        )
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(
            feature.size(0) * knn_k, classes, device=sim_labels.device
        )
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0
        )
        # weighted score ---> [B, C]
        pred_scores = torch.sum(
            one_hot_label.view(feature.size(0), -1, classes)
            * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )

        pred_labels = pred_scores.argsort(dim=-1, descending=True)

        pred_labels = pred_labels.cpu()

        pred_lst.append(pred_labels)

    pred_labels = torch.cat(pred_lst)
    return pred_labels


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Taken from https://github.com/HobbitLong/SupContrast/blob/master/losses.py"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0,
        # )
        logits_mask = 1 - torch.eye(batch_size * anchor_count, device=device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss, mask.sum() // 2


if __name__ == "__main__":
    train_erm()
