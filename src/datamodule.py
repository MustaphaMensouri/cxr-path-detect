import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lightning as L
from src.factories import build_transforms
from sklearn.model_selection import train_test_split


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
            rare_threshold = int(getattr(sample_cfg, "rare_threshold", 500))

            if target_n >= len(self.df):
                print("[Sample] sample size >= train size, using full train set")
            else:

                def stratified_patient_sample(df_pool, target_images, seed, pool_name="pool"):
                    patient_matrix = df_pool.groupby("PatientID", sort=False)[self.labels].max()

                    label_freq = patient_matrix[self.labels].sum(axis=0)

                    def key_for_patient(row):
                        positives = [col for col in self.labels if row[col] == 1]
                        if not positives:
                            return "__no_positive__"
                        return str(label_freq.loc[positives].idxmin())

                    patient_matrix["strat_key"] = patient_matrix.apply(key_for_patient, axis=1)

                    patients = patient_matrix.index.to_numpy()
                    keys = patient_matrix["strat_key"].to_numpy()

                    avg_images_per_patient = len(df_pool) / len(patient_matrix)
                    target_patient_count = max(1, int(target_images / avg_images_per_patient))
                    target_patient_count = min(target_patient_count, len(patient_matrix))

                    key_counts = pd.Series(keys).value_counts()
                    n_classes = len(key_counts)

                    leftover_patient_count = len(patient_matrix) - target_patient_count

                    can_stratify = (
                        n_classes > 1
                        and not (key_counts < 2).any()
                        and target_patient_count >= n_classes
                        and leftover_patient_count >= n_classes
                    )

                    stratify = keys if can_stratify else None

                    print("\n" + "=" * 80)
                    print(f"[Sample:{pool_name}] Pool images: {len(df_pool):,}")
                    print(f"[Sample:{pool_name}] Pool patients: {len(patient_matrix):,}")
                    print(f"[Sample:{pool_name}] Target images requested: {target_images:,}")
                    print(f"[Sample:{pool_name}] Avg images/patient: {avg_images_per_patient:.3f}")
                    print(f"[Sample:{pool_name}] Target patients estimated: {target_patient_count:,}")
                    print(f"[Sample:{pool_name}] Leftover patients if sampled: {leftover_patient_count:,}")
                    print(f"[Sample:{pool_name}] Stratification classes/keys: {n_classes:,}")
                    print(f"[Sample:{pool_name}] Min patients per key: {key_counts.min()}")
                    print(f"[Sample:{pool_name}] Max patients per key: {key_counts.max()}")
                    print(f"[Sample:{pool_name}] Stratification enabled: {can_stratify}")

                    if not can_stratify:
                        print(f"[Sample:{pool_name}] Stratification disabled reason(s):")
                        if n_classes <= 1:
                            print("  - Only one stratification class.")
                        if (key_counts < 2).any():
                            print("  - Some stratification classes have fewer than 2 patients.")
                        if target_patient_count < n_classes:
                            print(
                                f"  - Selected side too small: "
                                f"target_patient_count={target_patient_count} < n_classes={n_classes}"
                            )
                        if leftover_patient_count < n_classes:
                            print(
                                f"  - Leftover side too small: "
                                f"leftover_patient_count={leftover_patient_count} < n_classes={n_classes}"
                            )

                    print(f"[Sample:{pool_name}] Top 10 stratification keys:")
                    print(key_counts.head(10).to_string())
                    print("=" * 80 + "\n")

                    if target_patient_count >= len(patient_matrix):
                        print(f"[Sample:{pool_name}] Target patients >= pool patients, keeping all patients.")
                        return set(patients)

                    sample_patients, _ = train_test_split(
                        patients,
                        train_size=target_patient_count,
                        random_state=seed,
                        stratify=stratify,
                    )

                    return set(sample_patients)

                # 1. Find rare labels inside the current train split
                label_counts = self.df[self.labels].sum(axis=0)
                rare_labels = label_counts[label_counts < rare_threshold].index.tolist()

                # 2. Keep all patients that have at least one rare label
                patient_matrix = self.df.groupby("PatientID", sort=False)[self.labels].max()

                if rare_labels:
                    rare_patient_mask = patient_matrix[rare_labels].sum(axis=1) > 0
                    rare_patients = set(patient_matrix.index[rare_patient_mask])
                else:
                    rare_patients = set()

                rare_df = self.df[self.df["PatientID"].isin(rare_patients)].copy()
                common_df = self.df[~self.df["PatientID"].isin(rare_patients)].copy()

                # 3. If rare block is already bigger than target, sample rare patients only
                print("\n" + "#" * 80)
                print("[Sample] Sampling configuration")
                print(f"[Sample] Target images: {target_n:,}")
                print(f"[Sample] Full train images before sampling: {len(self.df):,}")
                print(f"[Sample] Full train patients before sampling: {self.df['PatientID'].nunique():,}")
                print(f"[Sample] Rare threshold: labels with < {rare_threshold} positives")
                print(f"[Sample] Number of labels used: {len(self.labels)}")
                print(f"[Sample] Number of rare labels: {len(rare_labels)}")

                if rare_labels:
                    print("[Sample] Rare labels and counts:")
                    rare_counts = label_counts[rare_labels].sort_values()
                    print(rare_counts.to_string())
                else:
                    print("[Sample] No rare labels found.")

                print(f"[Sample] Rare patients: {len(rare_patients):,}")
                print(f"[Sample] Rare block images: {len(rare_df):,}")
                print(f"[Sample] Common block images: {len(common_df):,}")
                print(f"[Sample] Common block patients: {common_df['PatientID'].nunique():,}")
                print("#" * 80 + "\n")

                if len(rare_df) >= target_n:
                    print(
                        f"[Sample] Rare block already has {len(rare_df):,} images, "
                        f"which is >= target {target_n:,}."
                    )
                    print("[Sample] Sampling only from rare-patient block.")

                    selected_patients = stratified_patient_sample(
                        rare_df,
                        target_images=target_n,
                        seed=sample_cfg.seed,
                        pool_name="rare_only",
                    )
                else:
                    remaining_budget = target_n - len(rare_df)

                    print(
                        f"[Sample] Keeping all rare patients first: {len(rare_df):,} images."
                    )
                    print(
                        f"[Sample] Remaining image budget to fill from common patients: "
                        f"{remaining_budget:,}"
                    )

                    common_patients = stratified_patient_sample(
                        common_df,
                        target_images=remaining_budget,
                        seed=sample_cfg.seed,
                        pool_name="common_fill",
                    ) if len(common_df) > 0 and remaining_budget > 0 else set()

                    selected_patients = rare_patients | common_patients

                self.df = (
                    self.df[self.df["PatientID"].isin(selected_patients)]
                    .reset_index(drop=True)
                )

                sample_label_counts = self.df[self.labels].sum(axis=0).sort_values(ascending=False)
                sample_patient_count = self.df["PatientID"].nunique()

                print("\n" + "#" * 80)
                print("[Sample] Final sampled train set")
                print(f"[Sample] Final sampled images: {len(self.df):,}")
                print(f"[Sample] Final sampled patients: {sample_patient_count:,}")
                print(f"[Sample] Requested target images: {target_n:,}")
                print(f"[Sample] Difference from target: {len(self.df) - target_n:+,}")
                print("[Sample] Top 15 labels in sampled set:")
                print(sample_label_counts.head(15).to_string())
                print("[Sample] Bottom 15 labels in sampled set:")
                print(sample_label_counts.tail(15).to_string())
                print("#" * 80 + "\n")

                print(
                    f"[Sample] rare threshold: <{rare_threshold} positives | "
                    f"rare labels: {len(rare_labels)} | "
                    f"sampled: {len(self.df)} images, "
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