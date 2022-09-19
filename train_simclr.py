from typing import List, Dict, Any, Optional
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
from biomass.norms import DomainBatchNorm2d


@hydra.main(version_base=None, config_path="configs", config_name="train_simclr")
def train_simclr(config: DictConfig) -> None:

    writer = SummaryWriter()

    dataloaders = instantiate(config.dataloaders)
    train_loader = dataloaders.get_train_loader(config.train_batch_size)
    eval_loaders = dataloaders.get_eval_loaders(config.eval_batch_size)

    model = instantiate(config.model)
    if config.model_path is not None:
        model.load_state_dict(torch.load(config.model_path))

    if config.use_domain_batch_norm:
        for m in model.modules():
            for n, c in m.named_children():
                if isinstance(c, nn.BatchNorm2d):
                    new_c = DomainBatchNorm2d.from_bn2d(c)
                    setattr(m, n, new_c)

    model = model.to(config.device)
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=len(train_loader) * config.num_epochs,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
    )

    simclr = MultiViewSimCLRLoss(temp=config.temperature)

    print("Completed Setup")

    # Log evaluation accuracies
    if not config.no_eval:
        model.eval()
    losses, accs = eval_model(
        model, eval_loaders, simclr, config.num_views, config.device, config
    )
    for key, val in losses.items():
        writer.add_scalar(f"Validation/loss_{key}", val, 0)
    for key, val in accs.items():
        writer.add_scalar(f"Validation/acc_{key}", val, 0)

    prev_projs = None

    for epoch in range(config.num_epochs):

        # Training
        t = time.time()
        model.train()
        for batch_id, (images, labels, lens, plates) in enumerate(train_loader):

            optimizer.zero_grad()

            images = images.to(config.device)

            if config.use_domain_batch_norm:
                # lens = lens.to(config.device)
                # plates = plates.to(config.device)
                # plates = torch.repeat_interleave(plates, lens)
                # for m in model.modules():
                #     if isinstance(m, DomainBatchNorm2d):
                #         m.set_domains(None)

                labels = labels.to(config.device)
                lens = lens.to(config.device)
                labels = torch.repeat_interleave(labels, lens)
                for m in model.modules():
                    if isinstance(m, DomainBatchNorm2d):
                        m.set_domains(labels)

            if config.bn_by_plate:
                unique_plates = torch.unique(plates)
                plates_by_img = torch.repeat_interleave(plates, lens)

                assert len(images) == len(plates_by_img)
                assert len(unique_plates) == 2

                projs = []
                for i, p in enumerate(unique_plates):
                    plate_projs = model.project(images[plates_by_img == p])
                    if i == 1:
                        plate_projs = plate_projs.detach()
                    projs.append(plate_projs)

                projs = torch.cat(projs, dim=0)
            else:
                projs = model.project(images)
                assert torch.all(lens == config.num_views)

            projs = F.normalize(projs, dim=1)

            projs = projs.unflatten(
                dim=0, sizes=(len(projs) // config.num_views, config.num_views)
            )

            loss, acc = simclr(projs, prev_projs)

            loss.backward()

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

            if config.store_prev_batch:
                prev_projs = model.project(images)
                prev_projs = F.normalize(prev_projs, dim=1).detach()

            # if config.use_domain_batch_norm:
            #     all_mean_diffs = []
            #     all_var_diffs = []
            #     for m in model.modules():
            #         if isinstance(m, DomainBatchNorm2d):
            #             all_mean_diffs.append(m.mean_diff)
            #             all_var_diffs.append(m.var_diff)
            #     writer.add_scalar(
            #         "Training/diff-mean",
            #         torch.mean(torch.stack(all_mean_diffs)).cpu(),
            #         global_step,
            #     )
            #     writer.add_scalar(
            #         "Training/diff-var",
            #         torch.mean(torch.stack(all_var_diffs)).cpu(),
            #         global_step,
            #     )

        # Evaluation
        if not config.no_eval:
            model.eval()
        losses, accs = eval_model(
            model, eval_loaders, simclr, config.num_views, config.device, config
        )
        for key, val in losses.items():
            writer.add_scalar(f"Validation/loss_{key}", val, epoch + 1)
        for key, val in accs.items():
            writer.add_scalar(f"Validation/acc_{key}", val, epoch + 1)
        print(accs)

        if config.save_model and epoch % 10 == 9:
            dt = HydraConfig.get().run.dir[8:].replace("-", "_").replace("/", "_")
            torch.save(model.state_dict(), f"checkpoints/simclr_{dt}.pt")

        print("TIME", time.time() - t)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    eval_loaders: Dict[str, DataLoader],
    loss_func: nn.Module,
    num_views: int,
    device: str,
    config: Dict[str, Any],
):
    # if config.use_domain_batch_norm:
    #     for m in model.modules():
    #         if isinstance(m, DomainBatchNorm2d):
    #             m.set_domains(None)

    global_losses = {}
    global_accs = {}
    for key, dataloader in eval_loaders.items():
        losses = []
        accs = []
        for batch_id, (images, labels, lens, plates) in enumerate(dataloader):
            images = images.to(device)

            # if config.use_domain_batch_norm:
            #     for m in model.modules():
            #         if isinstance(m, DomainBatchNorm2d):
            #             m.set_domains(None)
            if config.use_domain_batch_norm:
                labels = labels.to(config.device)
                lens = lens.to(config.device)
                labels = torch.repeat_interleave(labels, lens)
                for m in model.modules():
                    if isinstance(m, DomainBatchNorm2d):
                        m.set_domains(labels)

            if config.bn_by_plate:
                unique_plates = torch.unique(plates)
                plates_by_img = torch.repeat_interleave(plates, lens)

                assert len(images) == len(plates_by_img)
                assert len(unique_plates) == 1

                projs = []
                for p in unique_plates:
                    projs.append(model.project(images[plates_by_img == p]))
                projs = torch.cat(projs, dim=0)
            else:
                projs = model.project(images)

            projs = F.normalize(projs, dim=1)
            projs = projs.unflatten(dim=0, sizes=(len(images) // num_views, num_views))

            loss, acc = loss_func(projs)

            losses.append(loss.cpu())
            accs.append(acc.cpu())

        global_losses[key] = torch.mean(torch.stack(losses))
        global_accs[key] = torch.mean(torch.stack(accs))
    return global_losses, global_accs


class MultiViewSimCLRLoss(nn.Module):
    def __init__(self, temp=0.2):

        super().__init__()
        self.temp = temp

    def forward(self, projs: Tensor, extra_projs: Optional[Tensor] = None) -> Tensor:
        # projs has dimensions (batch x views x features)
        # reduces to SimCLR loss if views = 2
        # extra_projs has dimensions (num x features)

        batch_size, num_views, _ = projs.shape
        projs = projs.transpose(dim0=1, dim1=0).flatten(start_dim=0, end_dim=1)
        sims = projs @ projs.T / self.temp
        arange = torch.arange(num_views * batch_size, device=sims.device)
        sims[arange, arange] = -float("inf")

        pos_sims = []
        for x in range(1, num_views):
            arange_roll = torch.roll(arange, shifts=x * batch_size)
            pos_sims.append(sims[arange, arange_roll])
        sum_pos_sims = torch.stack(pos_sims, dim=1).sum(dim=1)

        if extra_projs is not None:
            extra_sims = projs @ extra_projs.T / self.temp
            sims = torch.cat([sims, extra_sims], dim=1)

        max_sims = torch.max(sims, dim=1).values

        sum_neg_sims = max_sims + torch.log(
            torch.sum(torch.exp(sims - max_sims.view(-1, 1)), dim=1)
        )

        loss = -torch.mean(sum_pos_sims / (num_views - 1) - sum_neg_sims, dim=0)

        with torch.no_grad():
            topk = torch.topk(sims, k=num_views - 1, dim=-1).indices
            correct_counts = torch.sum(
                topk % batch_size == arange.view(-1, 1) % batch_size, dim=1
            )
            acc = torch.mean(correct_counts / (num_views - 1), dim=0)

        return loss, acc


class SimCLRLoss(nn.Module):
    def __init__(self, temp=0.2):
        super().__init__()
        self.temp = temp
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, projs: Tensor) -> Tensor:
        batch_size = len(projs)
        projs = projs.flatten(start_dim=0, end_dim=1)
        sims = projs @ projs.T
        diag = torch.arange(2 * batch_size, device=sims.device)
        sims[diag, diag] = -float("inf")

        arange = torch.arange(batch_size, device=sims.device)
        labels = torch.stack([2 * arange + 1, 2 * arange], dim=1).flatten()
        loss = self.cross_entropy(sims / self.temp, labels)
        with torch.no_grad():
            acc = torch.sum(torch.argmax(sims, dim=1) == labels) / len(labels)
        return loss, acc


if __name__ == "__main__":
    train_simclr()
