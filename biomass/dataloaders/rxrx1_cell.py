from typing import Dict
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)
from PIL import Image


class RxRx1WildsCellDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        summary_file: str,
        subset: str = "train",
        transform=None,
        num_img: int = 2,
        mode: str = "random",
        metadata_file=None,
        include_labels=False,
        cell_type=None,
        use_one_site=False,
    ):
        self.img_dir = img_dir
        self.df = pd.read_csv(
            summary_file,
            names=["experiment", "plate", "well", "segmented", "available", "extra"],
            skiprows=1,
        )
        self.df = self.df[["experiment", "plate", "well", "segmented"]]
        self.df = self.df.dropna()
        self.df = self.df[self.df["segmented"] > 0]

        self.df = self.df[self.df["experiment"].isin(SPLITS[subset])]
        if cell_type is not None:
            self.df = self.df[self.df["experiment"].apply(lambda x: cell_type in x)]

        if use_one_site:
            possible_sites = SITES[subset][:1]
        else:
            possible_sites = SITES[subset]
        self.df = self.df[self.df["well"].apply(lambda x: int(x[-5]) in possible_sites)]
        self.df = self.df.drop_duplicates()
        assert self.df.duplicated(["experiment", "plate", "well"]).sum() == 0

        self.df = self.df.reset_index(drop=True)
        self.subset = subset
        self.transform = transform
        self.num_img = num_img
        self.mode = mode
        self.metadata_file = metadata_file
        self.include_labels = include_labels
        if self.include_labels and self.metadata_file is None:
            raise ValueError("Must include metadata_file if returning labels.")

        if self.metadata_file is not None:
            metadata_df = pd.read_csv(self.metadata_file)
            metadata_df["plate"] = metadata_df["plate"].apply(
                lambda x: "Plate" + str(x)
            )
            metadata_df["well"] = metadata_df[["well", "site"]].apply(
                lambda x: x["well"] + "_s" + str(x["site"]) + ".png", axis=1
            )
            metadata_df = metadata_df.set_index(["experiment", "plate", "well"])
            self.df = self.df.join(metadata_df, on=["experiment", "plate", "well"])

        self.exp_to_id = {k: i for i, k in enumerate(self.df["experiment"].unique())}
        self.plate_to_id = {k: i for i, k in enumerate(self.df["plate"].unique())}
        self.exp_plate_to_id = (
            lambda exp, plate: 4 * self.exp_to_id[exp] + self.plate_to_id[plate]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        entry = self.df.loc[idx]
        img_dir = os.path.join(
            self.img_dir, entry["experiment"], entry["plate"], entry["well"][:-4]
        )
        if self.mode == "random":
            img_idxs = np.random.randint(entry["segmented"], size=self.num_img)
        elif self.mode == "first":
            img_idxs = [0] * self.num_img
        elif self.mode == "all":
            img_idxs = np.arange(int(entry["segmented"]))
        elif self.mode == "random_single":
            chosen = np.random.randint(entry["segmented"], size=1)[0]
            img_idxs = [chosen] * self.num_img
        else:
            raise ValueError("Mode not implemented.")
        imgs = [Image.open(os.path.join(img_dir, f"{x}.png")) for x in img_idxs]
        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]
        if self.include_labels:
            return (
                imgs,
                torch.tensor(entry["sirna_id"]),
                torch.tensor(self.exp_plate_to_id(entry["experiment"], entry["plate"])),
            )
        else:
            return (
                imgs,
                None,
                torch.tensor(self.exp_plate_to_id(entry["experiment"], entry["plate"])),
            )


class RxRx1WildsCellDataloaders:
    def __init__(
        self,
        img_dir: str,
        summary_file: str,
        train_transform=None,
        eval_transform=None,
        metadata_file=None,
        eval_splits=[],
        mode: str = "random",
        sampler: str = "plate",
        include_labels: bool = False,
        num_img: int = 2,
        cell_type=None,
        num_plates_per_batch: int = 1,
        use_one_site=False,
        num_plate_parts: int = 1,
    ):
        self.img_dir = img_dir
        self.summary_file = summary_file
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.eval_splits = eval_splits
        self.mode = mode
        self.sampler = sampler
        self.metadata_file = metadata_file
        self.include_labels = include_labels
        self.num_img = num_img
        self.cell_type = cell_type
        self.num_plates_per_batch = num_plates_per_batch
        self.use_one_site = use_one_site
        self.num_plate_parts = num_plate_parts

    def get_train_loader(
        self, batch_size: int, use_eval_transform: bool = False
    ) -> DataLoader:
        train_data = RxRx1WildsCellDataset(
            self.img_dir,
            self.summary_file,
            subset="train",
            transform=self.eval_transform
            if use_eval_transform
            else self.train_transform,
            mode=self.mode,
            metadata_file=self.metadata_file,
            include_labels=self.include_labels,
            num_img=self.num_img,
            cell_type=self.cell_type,
            use_one_site=self.use_one_site,
        )
        if self.sampler == "plate":
            batch_sampler = MultiplateSampler(
                train_data,
                num_plates=self.num_plates_per_batch,
                num_plate_parts=self.num_plate_parts,
                random=True,  ## CHANGE
            )
        elif self.sampler == "random":
            batch_sampler = BatchSampler(
                RandomSampler(train_data),
                batch_size,
                drop_last=True,
            )
        elif self.sampler == "exp":
            batch_sampler = ExpSampler(train_data, random=True)
        else:
            raise ValueError()
        train_dl = DataLoader(
            train_data,
            num_workers=12,
            pin_memory=True,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )
        return train_dl

    def get_eval_loaders(self, batch_size: int) -> Dict[str, DataLoader]:
        eval_data = {
            split: RxRx1WildsCellDataset(
                self.img_dir,
                self.summary_file,
                subset=split,
                transform=self.eval_transform,
                mode=self.mode,
                metadata_file=self.metadata_file,
                include_labels=self.include_labels,
                num_img=self.num_img,
                cell_type=self.cell_type,
                use_one_site=self.use_one_site,
            )
            for split in self.eval_splits
        }

        eval_loaders = {}
        for split, split_data in eval_data.items():
            if self.sampler == "plate":
                batch_sampler = PlateSampler(split_data, random=False)
            elif self.sampler == "exp":
                batch_sampler = ExpSampler(split_data, random=False)
            elif self.sampler == "random":
                batch_sampler = BatchSampler(
                    SequentialSampler(split_data), batch_size, drop_last=False
                )
            else:
                raise ValueError()
            eval_loaders[split] = DataLoader(
                split_data,
                num_workers=12,
                pin_memory=True,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
            )

        return eval_loaders


class PlateSampler:
    def __init__(self, dataset, random: bool = True, num_plate_parts: int = 1):
        self.dataset = dataset
        self.exps = self.dataset.df["experiment"].unique()
        self.plates = self.dataset.df["plate"].unique()
        self.random = random
        self.num_plate_parts = num_plate_parts

    def __iter__(self):
        indexes = np.arange(len(self.exps) * len(self.plates))
        if self.random:
            np.random.shuffle(indexes)
        for idx in indexes:
            exp_id = idx // 4
            plate_id = idx % 4
            exp = self.exps[exp_id]
            plate = self.plates[plate_id]
            mask = (self.dataset.df["experiment"] == exp) & (
                self.dataset.df["plate"] == plate
            )
            all_wells = np.array(self.dataset.df[mask].index)
            if self.num_plate_parts == 1:
                yield all_wells
            else:
                np.random.shuffle(all_wells)
                for plate_part in np.array_split(all_wells, self.num_plate_parts):
                    yield plate_part

    def __len__(self):
        return len(self.exps) * len(self.plates) * self.num_plate_parts


class MultiplateSampler:
    def __init__(
        self,
        dataset,
        random: bool = True,
        num_plates: int = 1,
        num_plate_parts: int = 1,
    ):
        self.sampler = PlateSampler(dataset, random, num_plate_parts)
        self.num_plates = num_plates
        num_batches = len(self.sampler) / self.num_plates
        if not num_batches.is_integer():
            raise ValueError()

    def __len__(self):
        return len(self.sampler) // self.num_plates

    def __iter__(self):
        iterator = iter(self.sampler)
        for _ in range(self.__len__()):
            wells_lst = []
            for _ in range(self.num_plates):
                wells_lst.append(next(iterator))
            yield np.concatenate(wells_lst, axis=0)


class ExpSampler:
    def __init__(self, dataset, random: bool = True):
        self.dataset = dataset
        self.exps = self.dataset.df["experiment"].unique()
        self.random = random

    def __iter__(self):
        indexes = np.arange(len(self.exps))
        if self.random:
            np.random.shuffle(indexes)
        for exp_id in indexes:
            exp = self.exps[exp_id]
            mask = self.dataset.df["experiment"] == exp
            all_wells = np.array(self.dataset.df[mask].index)
            yield all_wells

    def __len__(self):
        return len(self.exps)


def collate_fn(data):
    img_lists, labels, plates = zip(*data)

    if labels[0] is not None:
        labels = torch.stack(labels)
    else:
        labels = None

    plates = torch.stack(plates)

    imgs = torch.stack([img for img_list in img_lists for img in img_list])
    lens = torch.tensor([len(img_list) for img_list in img_lists])

    return imgs, labels, lens, plates


SPLITS = {
    "train": [
        "HEPG2-01",
        "HEPG2-02",
        "HEPG2-03",
        "HEPG2-04",
        "HEPG2-05",
        "HEPG2-06",
        "HEPG2-07",
        "HUVEC-01",
        "HUVEC-02",
        "HUVEC-03",
        "HUVEC-04",
        "HUVEC-05",
        "HUVEC-06",
        "HUVEC-07",
        "HUVEC-08",
        "HUVEC-09",
        "HUVEC-10",
        "HUVEC-11",
        "HUVEC-12",
        "HUVEC-13",
        "HUVEC-14",
        "HUVEC-15",
        "HUVEC-16",
        "RPE-01",
        "RPE-02",
        "RPE-03",
        "RPE-04",
        "RPE-05",
        "RPE-06",
        "RPE-07",
        "U2OS-01",
        "U2OS-02",
        "U2OS-03",
    ],
    "val": ["HEPG2-08", "HUVEC-17", "RPE-08", "U2OS-04"],
    "test": [
        "HEPG2-09",
        "HEPG2-10",
        "HEPG2-11",
        "HUVEC-18",
        "HUVEC-19",
        "HUVEC-20",
        "HUVEC-21",
        "HUVEC-22",
        "HUVEC-23",
        "HUVEC-24",
        "RPE-09",
        "RPE-10",
        "RPE-11",
        "U2OS-05",
    ],
    "iid_val": [
        "HEPG2-01",
        "HEPG2-02",
        "HEPG2-03",
        "HEPG2-04",
        "HEPG2-05",
        "HEPG2-06",
        "HEPG2-07",
        "HUVEC-01",
        "HUVEC-02",
        "HUVEC-03",
        "HUVEC-04",
        "HUVEC-05",
        "HUVEC-06",
        "HUVEC-07",
        "HUVEC-08",
        "HUVEC-09",
        "HUVEC-10",
        "HUVEC-11",
        "HUVEC-12",
        "HUVEC-13",
        "HUVEC-14",
        "HUVEC-15",
        "HUVEC-16",
        "RPE-01",
        "RPE-02",
        "RPE-03",
        "RPE-04",
        "RPE-05",
        "RPE-06",
        "RPE-07",
        "U2OS-01",
        "U2OS-02",
        "U2OS-03",
    ],
}

SITES = {"train": [1], "val": [1, 2], "test": [1, 2], "iid_val": [2]}
