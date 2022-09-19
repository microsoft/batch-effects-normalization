from typing import Tuple

import numpy as np
from numpy import ndarray
import h5py
from skimage.exposure import rescale_intensity
import torch
from torch.utils.data import Dataset
from torch import Tensor


class CellOutOfSampleDataset(Dataset):
    """Dataset object for cell-out-of-sample."""

    def __init__(self, path: str, random_flips: bool = False):
        self.path = path
        archive = h5py.File(path, "r")
        self.images = archive["data"]
        self.labels = archive["labels"]
        self.date_ids = np.array(archive["dateIDs"])
        self.microscope_ids = np.array(archive["microscopeIDs"])
        self.plate_ids = np.array(archive["plateIDs"])
        self.well_ids = np.array(archive["wellIDs"])
        self.random_flips = random_flips

    def __len__(self) -> int:
        return len(self.images)

    def rescale(self, img: ndarray) -> ndarray:
        return rescale_intensity(img.astype(np.float32), out_range=(0, 1))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        protein = torch.tensor(self.rescale(self.images[idx][0]))
        nucleus = torch.tensor(self.rescale(self.images[idx][1]))
        label = torch.tensor(self.labels[idx])

        if self.random_flips:
            if np.random.rand() < 0.5:
                protein = torch.fliplr(protein)
                nucleus = torch.fliplr(nucleus)
            if np.random.rand() < 0.5:
                protein = torch.flipud(protein)
                nucleus = torch.flipud(nucleus)

        return protein, nucleus, label


class PairedCellOutOfSampleDataset(Dataset):
    def __init__(self, dataset: CellOutOfSampleDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        protein1, nucleus1, label1 = self.dataset[idx]

        # Sample similar image
        date_mask = self.dataset.date_ids[idx] == self.dataset.date_ids
        micro_mask = self.dataset.microscope_ids[idx] == self.dataset.microscope_ids
        plate_mask = self.dataset.plate_ids[idx] == self.dataset.plate_ids
        well_mask = self.dataset.well_ids[idx] == self.dataset.well_ids
        common_ids = np.where(date_mask & micro_mask & plate_mask & well_mask)[0]
        pair_id = np.random.choice(common_ids[common_ids != idx])
        protein2, nucleus2, label2 = self.dataset[idx]

        return protein1, nucleus1, label1, protein2, nucleus2, label2
