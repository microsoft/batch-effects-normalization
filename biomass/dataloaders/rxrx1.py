from typing import Dict, List, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.utils import split_into_groups

from biomass.transforms import (
    Standardize,
    ToTensor,
    RandomRotate,
    RandomHorizontalFlip,
    Compose,
)
from wilds.common.grouper import CombinatorialGrouper

DEFAULT_TRAIN_TRANSFORM = Compose(
    [RandomRotate(), RandomHorizontalFlip(), ToTensor(), Standardize(dim=(-2, -1))]
)
DEFAULT_EVAL_TRANSFORM = Compose([ToTensor(), Standardize(dim=(-2, -1))])
# METADATA: cell_type, experiment, plate, well, site, y, 1


class RxRx1Dataloaders:
    def __init__(
        self,
        eval_splits: List[str] = ["val"],
        train_transform: Callable = DEFAULT_TRAIN_TRANSFORM,
        eval_transform: Callable = DEFAULT_EVAL_TRANSFORM,
        train_groupby: Optional[List[str]] = None,
        max_groups: Optional[int] = None,
        cross_product: bool = False,
        eval_groupby: Optional[List[str]] = None,
        eval_plate_sampler: bool = False,
        eval_exp_sampler: bool = False,
    ):
        self.dataset = get_dataset(dataset="rxrx1", download=True)

        self.eval_splits = eval_splits
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.train_groupby = train_groupby
        self.max_groups = max_groups
        self.cross_product = cross_product
        self.eval_groupby = eval_groupby

        if eval_plate_sampler and eval_exp_sampler:
            raise ValueError("These cannot both be true.")

        self.eval_plate_sampler = eval_plate_sampler
        self.eval_exp_sampler = eval_exp_sampler

    def get_train_loader(
        self, batch_size: int, use_eval_transform: bool = False
    ) -> DataLoader:
        train_data = self.dataset.get_subset(
            "train",
            transform=self.eval_transform
            if use_eval_transform
            else self.train_transform,
        )
        if self.train_groupby is not None and not use_eval_transform:
            if self.cross_product:
                if len(self.train_groupby) != 2:
                    raise ValueError(
                        "For cross product sampler, we require 2 dimensions for train_groupby."
                    )

                group_ids1 = CombinatorialGrouper(
                    self.dataset, [self.train_groupby[0]]
                ).metadata_to_group(train_data.metadata_array)

                group_ids2 = CombinatorialGrouper(
                    self.dataset, [self.train_groupby[1]]
                ).metadata_to_group(train_data.metadata_array)
                sampler = CrossProductSampler(
                    group_ids1,
                    group_ids2,
                    batch_size,
                    batch_size // self.max_groups,
                    self.max_groups,
                )
                train_loader = DataLoader(
                    train_data,
                    shuffle=None,
                    sampler=None,
                    collate_fn=train_data.collate,
                    batch_sampler=sampler,
                    drop_last=False,
                    num_workers=6,
                    pin_memory=True,
                )
            else:
                train_grouper = CombinatorialGrouper(self.dataset, self.train_groupby)
                train_loader = get_train_loader(
                    "group",
                    train_data,
                    grouper=train_grouper,
                    batch_size=batch_size,
                    n_groups_per_batch=self.max_groups,
                    num_workers=6,
                    pin_memory=True,
                )
        else:
            train_loader = get_train_loader(
                "standard",
                train_data,
                batch_size=batch_size,
                num_workers=6,
                pin_memory=True,
                drop_last=True,
            )
        return train_loader

    def get_eval_loaders(self, batch_size: int) -> Dict[str, DataLoader]:
        eval_data = {
            split: self.dataset.get_subset(split, transform=self.eval_transform)
            for split in self.eval_splits
        }
        if self.eval_plate_sampler:
            eval_loaders = {
                split: DataLoader(
                    split_data,
                    num_workers=12,
                    pin_memory=True,
                    batch_sampler=PlateSampler(split_data),
                )
                for split, split_data in eval_data.items()
            }
        elif self.eval_exp_sampler:
            eval_loaders = {
                split: DataLoader(
                    split_data,
                    num_workers=12,
                    pin_memory=True,
                    batch_sampler=ExpSampler(split_data),
                )
                for split, split_data in eval_data.items()
            }
        elif self.eval_groupby is not None:
            eval_grouper = CombinatorialGrouper(self.dataset, self.eval_groupby)
            eval_loaders = {
                split: get_train_loader(
                    "group",
                    split_data,
                    grouper=eval_grouper,
                    batch_size=batch_size,
                    n_groups_per_batch=self.max_groups,
                    num_workers=6,
                    pin_memory=True,
                )
                for split, split_data in eval_data.items()
            }
        else:
            eval_loaders = {
                split: get_eval_loader(
                    "standard",
                    split_data,
                    batch_size=batch_size,
                    num_workers=6,
                    pin_memory=True,
                )
                for split, split_data in eval_data.items()
            }
        return eval_loaders


class PlateSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.exps = dataset.metadata_array[:, 1].unique()
        self.plates = dataset.metadata_array[:, 2].unique()

    def __iter__(self):
        all_ids = torch.arange(len(self.dataset))
        indexes = np.arange(len(self.exps) * len(self.plates))
        for idx in indexes:
            exp_id = idx // 4
            plate_id = idx % 4
            exp = self.exps[exp_id]
            plate = self.plates[plate_id]
            mask = (self.dataset.metadata_array[:, 1] == exp) & (
                self.dataset.metadata_array[:, 2] == plate
            )
            all_wells = all_ids[mask]
            yield all_wells

    def __len__(self):
        return len(self.exps) * len(self.plates)


class ExpSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.exps = dataset.metadata_array[:, 1].unique()

    def __iter__(self):
        all_ids = torch.arange(len(self.dataset))
        indexes = np.arange(len(self.exps))
        for exp_id in indexes:
            exp = self.exps[exp_id]
            mask = self.dataset.metadata_array[:, 1] == exp
            all_wells = all_ids[mask]
            yield all_wells

    def __len__(self):
        return len(self.exps)


class CrossProductSampler:
    """
    Constructs batches by sampling two sets of groups and every possible combination between those sets.
    """

    def __init__(self, group_ids1, group_ids2, batch_size, n_groups1, n_groups2):

        if batch_size != n_groups1 * n_groups2:
            raise ValueError(
                f"batch_size ({batch_size}) must be equal num groups 1 ({n_groups1}) times num groups2 ({n_groups2})."
            )
        if len(group_ids1) < batch_size:
            raise ValueError(
                f"The dataset has only {len(group_ids1)} examples but the batch size is {batch_size}. There must be enough examples to form at least one complete batch."
            )

        self.n_groups1 = n_groups1
        self.n_groups2 = n_groups2
        self.group_ids1 = group_ids1
        self.group_ids2 = group_ids2
        self.unique_groups1, _, _ = split_into_groups(group_ids1)
        self.unique_groups2, _, _ = split_into_groups(group_ids2)
        self.unique_groups1 = self.unique_groups1.numpy()
        self.unique_groups2 = self.unique_groups2.numpy()

        self.dataset_size = len(group_ids1)
        self.num_batches = self.dataset_size // batch_size

        all_indices = torch.arange(len(group_ids1))
        self.group_indices = []
        for g1 in range(group_ids1.max() + 1):
            g1_lst = []
            g1_mask = group_ids1 == g1
            for g2 in range(group_ids2.max() + 1):
                g2_mask = group_ids2 == g2
                g1_lst.append(all_indices[g1_mask & g2_mask].numpy())
            self.group_indices.append(g1_lst)

    def __iter__(self):
        for batch_id in range(self.num_batches):
            groups1_for_batch = np.random.choice(
                self.unique_groups1, size=self.n_groups1, replace=False
            )
            groups2_for_batch = np.random.choice(
                self.unique_groups2, size=self.n_groups2, replace=False
            )
            sample_ids = []
            for g1 in groups1_for_batch:
                for g2 in groups2_for_batch:
                    candidates = self.group_indices[g1][g2]
                    if len(candidates) > 0:
                        sample_ids.append(np.random.choice(candidates))
            yield np.array(sample_ids)

    def __len__(self):
        return self.num_batches
