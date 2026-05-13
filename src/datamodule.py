import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lightning as L
from src.factories import build_transforms
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class XrayDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform, labels, sample_cfg=None):
        self.df = pd.read_csv(csv_path)

        if sample_cfg is not None and sample_cfg.enabled:
            # number of target rows
            target_n = min(sample_cfg.size, len(self.df))

            # build patient-level multilabel matrix
            patient_matrix = (
                self.df.groupby("PatientID", sort=False)[self.labels]
                .max()
                .reset_index()
            )

            # estimate how many patients we need
            avg_images_per_patient = len(self.df) / len(patient_matrix)

            target_patient_count = max(
                1,
                int(target_n / avg_images_per_patient)
            )

            # multilabel stratified sampling at patient level
            X = patient_matrix["PatientID"].values
            y = patient_matrix[self.labels].values

            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                train_size=target_patient_count,
                random_state=sample_cfg.seed,
            )

            sample_patient_idx, _ = next(msss.split(X, y))

            sample_patient_ids = set(X[sample_patient_idx])

            # keep all images belonging to selected patients
            self.df = (
                self.df[self.df["PatientID"].isin(sample_patient_ids)]
                .reset_index(drop=True)
            )

            # optional: trim if slightly above target_n
            if len(self.df) > target_n:
                self.df = self.df.iloc[:target_n].reset_index(drop=True)

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.labels = [l for l in labels if l in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.data_dir / row["image_path"]).convert("RGB")
        label = torch.tensor(row[self.labels].values.astype(float), dtype=torch.float32)
        return self.transform(img), label


class XrayDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg      = cfg
        self.train_tf, self.val_tf = build_transforms(cfg.augmentation)
        self.labels = load_labels(self.cfg.data.labels_path)

    def _loader(self, split, transform, shuffle=False):
        sample_cfg = self.cfg.data.sample if split == "train" else None

        ds = XrayDataset(
            f"{self.cfg.data.data_dir}/{split}.csv",
            self.cfg.data.data_dir,
            transform,
            labels=self.labels,
            sample_cfg=sample_cfg,
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=shuffle,
            pin_memory=self.cfg.data.pin_memory,
            prefetch_factor=self.cfg.data.prefetch_factor,
            persistent_workers=self.cfg.data.persistent_workers,
        )

    def train_dataloader(self): return self._loader("train", self.train_tf, shuffle=True)
    def val_dataloader(self):   return self._loader("val",   self.val_tf)
    def test_dataloader(self):  return self._loader("test",  self.val_tf)