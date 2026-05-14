import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lightning as L
from src.factories import build_transforms


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class XrayDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform, labels, sample_cfg=None):
        self.df = pd.read_csv(csv_path)

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.labels = [l for l in labels if l in self.df.columns]

        if sample_cfg is not None and sample_cfg.enabled:
            target_n = min(int(sample_cfg.size), len(self.df))

            if target_n >= len(self.df):
                print("[Sample] sample size >= train size, using full train set")
            else:
                patient_matrix = (
                    self.df.groupby("PatientID", sort=False)[self.labels]
                    .max()
                )

                label_freq = patient_matrix[self.labels].sum(axis=0)

                def key_for_patient(row):
                    positives = [col for col in self.labels if row[col] == 1]
                    if not positives:
                        return "__no_positive__"
                    return str(label_freq.loc[positives].idxmin())

                patient_matrix["strat_key"] = patient_matrix.apply(key_for_patient, axis=1)

                patients = patient_matrix.index.to_numpy()
                keys = patient_matrix["strat_key"].to_numpy()

                avg_images_per_patient = len(self.df) / len(patient_matrix)
                target_patient_count = max(1, int(target_n / avg_images_per_patient))
                target_patient_count = min(target_patient_count, len(patient_matrix) - 1)

                from sklearn.model_selection import train_test_split

                key_counts = pd.Series(keys).value_counts()
                stratify = keys if len(key_counts) > 1 and not (key_counts < 2).any() else None

                sample_patients, _ = train_test_split(
                    patients,
                    train_size=target_patient_count,
                    random_state=sample_cfg.seed,
                    stratify=stratify,
                )

                self.df = (
                    self.df[self.df["PatientID"].isin(set(sample_patients))]
                    .reset_index(drop=True)
                )

                print(
                    f"[Sample] train sampled: {len(self.df)} images, "
                    f"{self.df['PatientID'].nunique()} patients"
                )
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