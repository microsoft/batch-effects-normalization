import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier

from biomass.datasets import PairedCellOutOfSampleDataset
from biomass.utils import compute_grad_norm


@hydra.main(version_base=None, config_path="configs", config_name="train_pci")
def train_pci(config: DictConfig) -> None:
    "Training function for paired cell inpainting."

    writer = SummaryWriter()

    dataset = instantiate(config.dataset)
    model = instantiate(config.model).to(config.device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=[30, 70], gamma=0.1)

    # Dataloader for paired cells
    paired_dataset = PairedCellOutOfSampleDataset(dataset)
    dataloader1 = DataLoader(
        paired_dataset, config.batch_size, shuffle=True, drop_last=True
    )

    # Dataloader for single cells
    dataloader2 = DataLoader(dataset, config.batch_size, shuffle=False, drop_last=False)

    mse = MSELoss()

    for epoch in range(config.num_epochs):

        # Classification Loop (compute task accuracy)
        with torch.no_grad():
            all_features = []
            all_labels = []
            for protein, nucleus, label in dataloader2:
                protein = protein.to(config.device)
                nucleus = nucleus.to(config.device)
                label = label.to(config.device)
                features = model.encode(nucleus, protein)
                features = torch.amax(features, dim=(-2, -1))
                all_features.append(features.to("cpu"))
                all_labels.append(label.to("cpu"))
            all_features = torch.cat(all_features).numpy()
            all_labels = torch.cat(all_labels).numpy()

        # Standardize features
        mean = all_features.mean(axis=0)
        std = all_features.std(axis=0)
        all_features = (all_features - mean) / std

        # Should be leave-one-out
        classifier = KNeighborsClassifier(n_neighbors=11)
        classifier.fit(all_features, all_labels)
        all_preds = classifier.predict(all_features)
        class_scores = []
        for k in np.unique(all_labels):
            class_score = np.mean(all_preds[all_labels == k] == k)
            class_scores.append(class_score)
        score = np.mean(class_scores)
        print(score)

        writer.add_scalar("Classification/acc", score, epoch)

        # Self-Supervised Loop (self-supervised paired cell inpainting)
        for batch_id, batch in enumerate(dataloader1):
            optimizer.zero_grad()

            batch = [x.to(config.device) for x in batch]
            protein1, nucleus1, label1, protein2, nucleus2, label2 = batch

            pred_protein2 = model(nucleus1, protein1, nucleus2)

            loss = mse(pred_protein2, protein2)
            loss.backward()
            optimizer.step()

            global_step = epoch * len(dataloader1) + batch_id
            writer.add_scalar("Paired-Inpainting/loss", loss, global_step)

            grad_norm = compute_grad_norm(model)
            writer.add_scalar("Paired-Inpainting/grad-norm", grad_norm, global_step)

        writer.add_images("Paired-Inpainting/targets", protein2.unsqueeze(dim=1), epoch)
        writer.add_images(
            "Paired-Inpainting/recons", pred_protein2.unsqueeze(dim=1), epoch
        )

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Paired-Inpainting/lr", lr, epoch)
        scheduler.step()

        # Save model
        if config.save_model and epoch % 10 == 9:
            dt = HydraConfig.get().run.dir[8:].replace("-", "_").replace("/", "_")
            torch.save(model.state_dict(), f"checkpoints/pci_{dt}.pt")


if __name__ == "__main__":
    train_pci()
