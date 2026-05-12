#!/usr/bin/env python3
"""
prepare_padchest_datamart.py
============================

Build PadChest data marts and patient-safe train/val/test splits.

Main features
-------------
1. Build different data marts from the same PadChest CSV:
   - PA-only
   - all projections
   - custom projections: PA, L, AP, AP-horizontal, COSTAL, etc.
2. Keep only the labels you want:
   - built-in lung label set
   - or override using a .txt labels file
3. Optional label normalization / aliases.
4. Patient-level split to avoid leakage.
5. Stratification based on each patient's rarest positive label.
6. Split quality reports with per-label image counts and patient counts.
7. Optional image resizing/copying.

Example usage
-------------
# PA-only lung data mart
python prepare_padchest_datamart.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_PA \
  --datamart-name lung_PA \
  --projections PA \
  --labels-file lung_labels.txt \
  --image-size 224

# All projections, no resizing
python prepare_padchest_datamart.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_all_proj \
  --datamart-name lung_all_proj \
  --projections ALL \
  --labels-file lung_labels.txt \
  --no-resize

Labels file format
------------------
One label per line. Blank lines and lines starting with # are ignored.

Example:
    # lung findings / diagnoses
    normal
    infiltrates
    pleural effusion
    pneumonia

Optional aliases file format
----------------------------
CSV with two columns: source,target

Example:
    source,target
    COPD signs,emphysema
    pulmonary artery hypertension,pulmonary hypertension

Important PadChest note
-----------------------
The labels are report-derived and weak image labels. A study can have multiple images
such as PA and lateral sharing the same report labels. Use patient-level splitting to
avoid leakage across train/val/test.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2


PAD_CSV_NAME = "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"


DEFAULT_LUNG_LABELS = [
    # normal / special
    "normal",

    # infiltrates / patterns / opacities
    "infiltrates", "alveolar pattern", "interstitial pattern",
    "ground glass pattern", "reticular interstitial pattern",
    "reticulonodular interstitial pattern", "miliary opacities",
    "consolidation", "increased density", "air bronchogram",
    "bronchovascular markings",

    # atelectasis / volume
    "atelectasis", "laminar atelectasis", "lobar atelectasis",
    "segmental atelectasis", "total atelectasis", "round atelectasis",
    "atelectasis basal", "volume loss", "hypoexpansion",

    # nodules / masses
    "nodule", "multiple nodules", "pseudonodule", "mass", "pulmonary mass",

    # pleural
    "pleural effusion", "loculated pleural effusion", "pleural thickening",
    "apical pleural thickening", "pleural plaques", "costophrenic angle blunting",
    "hydropneumothorax", "empyema", "pleural mass",

    # air / pneumothorax / hyperinflation
    "pneumothorax", "pneumomediastinum", "pneumoperitoneo",
    "subcutaneous emphysema", "air trapping", "bullas",
    "hyperinflated lung", "flattened diaphragm",

    # chronic / fibrotic
    "chronic changes", "pulmonary fibrosis", "fibrotic band",
    "post radiotherapy changes",

    # infectious / diagnoses
    "cavitation", "abscess", "tuberculosis", "tuberculosis sequelae",
    "pneumonia", "atypical pneumonia",

    # vascular / edema / airways
    "vascular redistribution", "hilar enlargement", "vascular hilar enlargement",
    "pulmonary artery enlargement", "hilar congestion", "pulmonary hypertension",
    "pulmonary artery hypertension", "pulmonary venous hypertension",
    "pulmonary edema", "kerley lines", "bronchiectasis", "tracheal shift",

    # other lung-related diagnoses / context
    "lung metastasis", "lymphangitis carcinomatosa", "lepidic adenocarcinoma",
    "emphysema", "COPD signs", "respiratory distress", "asbestosis signs",
    "surgery lung",
]


@dataclass(frozen=True)
class Config:
    data_root: str
    output_dir: str
    datamart_name: str
    csv_name: str
    labels_file: str | None
    aliases_file: str | None
    projections: list[str]
    method_labels: list[str]
    train_split: float
    val_split: float
    seed: int
    min_label_count: int
    image_size: int
    num_workers: int
    no_resize: bool
    copy_original_images: bool
    drop_unchanged: bool
    drop_exclude: bool
    drop_suboptimal: bool
    allow_single_split_classes: bool


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def safe_col(label: str) -> str:
    return (
        normalize_label(label)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "")
    )


def read_label_file(path: Path) -> list[str]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels.append(normalize_label(line))
    return list(dict.fromkeys(labels))


def read_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    required = {"source", "target"}
    if not required.issubset(df.columns):
        fail("aliases file must contain columns: source,target")
    return {
        normalize_label(row["source"]): normalize_label(row["target"])
        for _, row in df.iterrows()
        if pd.notna(row["source"]) and pd.notna(row["target"])
    }


def parse_list_cell(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [normalize_label(x) for x in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if not isinstance(parsed, (list, tuple, set)):
        return []
    return [normalize_label(x) for x in parsed]


def apply_aliases(labels: Iterable[str], aliases: dict[str, str]) -> list[str]:
    out = []
    for label in labels:
        out.append(aliases.get(normalize_label(label), normalize_label(label)))
    return list(dict.fromkeys(out))


def filter_labels(labels: list[str], keep_set: set[str]) -> list[str]:
    return [label for label in labels if label in keep_set]


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        fail(f"CSV is missing required columns: {missing}")


def validate_splits(train_split: float, val_split: float) -> float:
    test_split = round(1.0 - train_split - val_split, 10)
    if train_split <= 0:
        fail("train_split must be > 0")
    if val_split < 0:
        fail("val_split must be >= 0")
    if test_split < 0:
        fail("train_split + val_split cannot be greater than 1")
    if val_split == 0 and test_split == 0:
        fail("at least one of val or test must be > 0")
    return test_split


def build_image_index(data_root: Path) -> dict[str, Path]:
    index = {}
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for path in data_root.glob(f"**/{ext}"):
            # If duplicates exist, keep the first and warn later by count only.
            index.setdefault(path.name, path)
    return index


def resize_or_copy_one(task):
    src, dst, image_size, copy_original = task
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy_original:
            shutil.copy2(src, dst)
        else:
            # Read original image preserving bit depth
            img = cv2.imread(str(src), cv2.IMREAD_ANYDEPTH)

            if img is None:
                raise ValueError(f"Could not read image: {src}")

            # Convert to float32
            img = img.astype(np.float32)

            # Normalize to 0-1
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Convert to uint8
            img = (img * 255).astype(np.uint8)

            # Resize
            img = cv2.resize(
                img,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA
            )

            # Save JPG
            cv2.imwrite(str(dst), img)

        return src.name, True, None
    except Exception as exc:
        return src.name, False, str(exc)


def collapse_rare_strata(keys: np.ndarray, min_count: int = 2) -> np.ndarray:
    counts = pd.Series(keys).value_counts(dropna=False)
    rare = set(counts[counts < min_count].index)
    return np.array(["__rare__" if key in rare else key for key in keys])


def make_stratification_keys(patient_matrix: pd.DataFrame, label_cols: list[str]) -> pd.Series:
    label_freq = patient_matrix[label_cols].sum(axis=0)

    def key_for_patient(row: pd.Series) -> str:
        positives = [col for col in label_cols if row[col] == 1]
        if not positives:
            return "__no_positive__"
        return str(label_freq.loc[positives].idxmin())

    return patient_matrix.apply(key_for_patient, axis=1)


def patient_stratified_split(
    df: pd.DataFrame,
    label_cols: list[str],
    train_split: float,
    val_split: float,
    seed: int,
    allow_single_split_classes: bool,
) -> dict[str, pd.DataFrame]:
    test_split = validate_splits(train_split, val_split)

    patient_matrix = df.groupby("PatientID", sort=False)[label_cols].max()
    patient_matrix["strat_key"] = make_stratification_keys(patient_matrix, label_cols)

    patients = patient_matrix.index.to_numpy()
    raw_keys = patient_matrix["strat_key"].to_numpy()
    keys = collapse_rare_strata(raw_keys, min_count=2)

    key_counts = pd.Series(keys).value_counts()
    if (key_counts < 2).any() and not allow_single_split_classes:
        fail(
            "Some stratification classes have fewer than 2 patients even after collapsing. "
            "Use --allow-single-split-classes to fall back to non-stratified split for that stage."
        )

    temp_size = val_split + test_split
    stratify_first = keys if len(np.unique(keys)) > 1 and not (pd.Series(keys).value_counts() < 2).any() else None

    train_patients, temp_patients, _, temp_keys = train_test_split(
        patients,
        keys,
        test_size=temp_size,
        random_state=seed,
        stratify=stratify_first,
    )

    if val_split == 0:
        val_patients = np.array([], dtype=patients.dtype)
        test_patients = temp_patients
    elif test_split == 0:
        val_patients = temp_patients
        test_patients = np.array([], dtype=patients.dtype)
    else:
        relative_test = test_split / temp_size
        temp_keys_collapsed = collapse_rare_strata(np.array(temp_keys), min_count=2)
        temp_counts = pd.Series(temp_keys_collapsed).value_counts()
        stratify_second = (
            temp_keys_collapsed
            if len(np.unique(temp_keys_collapsed)) > 1 and not (temp_counts < 2).any()
            else None
        )
        if stratify_second is None and not allow_single_split_classes:
            fail(
                "Second split cannot be stratified because the temporary pool has classes with <2 patients. "
                "Use --allow-single-split-classes or increase val/test size."
            )
        val_patients, test_patients, _, _ = train_test_split(
            temp_patients,
            temp_keys_collapsed,
            test_size=relative_test,
            random_state=seed,
            stratify=stratify_second,
        )

    split_patient_sets = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }

    return {
        split: df[df["PatientID"].isin(patient_set)].copy()
        for split, patient_set in split_patient_sets.items()
    }


def write_split_report(output_dir: Path, splits: dict[str, pd.DataFrame], label_cols: list[str]) -> None:
    rows = []
    for split_name, sdf in splits.items():
        for label in label_cols:
            rows.append({
                "split": split_name,
                "label": label,
                "image_count": int(sdf[label].sum()) if len(sdf) else 0,
                "image_prevalence": float(sdf[label].mean()) if len(sdf) else 0.0,
                "patient_count": int(sdf.loc[sdf[label] == 1, "PatientID"].nunique()) if len(sdf) else 0,
            })
    pd.DataFrame(rows).to_csv(output_dir / "label_distribution_by_split.csv", index=False)

    summary = []
    for split_name, sdf in splits.items():
        summary.append({
            "split": split_name,
            "images": int(len(sdf)),
            "patients": int(sdf["PatientID"].nunique()) if len(sdf) else 0,
            "studies": int(sdf["StudyID"].nunique()) if "StudyID" in sdf.columns and len(sdf) else 0,
        })
    pd.DataFrame(summary).to_csv(output_dir / "split_summary.csv", index=False)


def prepare_datamart(cfg: Config) -> None:
    data_root = Path(cfg.data_root)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_split = validate_splits(cfg.train_split, cfg.val_split)
    print(f"Splits: train={cfg.train_split:.2f}, val={cfg.val_split:.2f}, test={test_split:.2f}")

    labels = read_label_file(Path(cfg.labels_file)) if cfg.labels_file else [normalize_label(x) for x in DEFAULT_LUNG_LABELS]
    aliases = read_aliases(Path(cfg.aliases_file) if cfg.aliases_file else None)
    labels = apply_aliases(labels, aliases)
    keep_set = set(labels)

    print(f"Using {len(labels)} target labels")

    csv_path = data_root / cfg.csv_name
    if not csv_path.exists():
        fail(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    validate_columns(df, ["ImageID", "PatientID", "Projection", "MethodLabel", "Labels"])
    print(f"Original rows: {len(df):,}")

    df = df[df["Labels"].notna()].copy()
    df["parsed_labels"] = df["Labels"].apply(parse_list_cell).apply(lambda x: apply_aliases(x, aliases))

    if cfg.drop_unchanged:
        df["parsed_labels"] = df["parsed_labels"].apply(lambda xs: [x for x in xs if x != "unchanged"])
    if cfg.drop_exclude:
        df["parsed_labels"] = df["parsed_labels"].apply(lambda xs: [x for x in xs if x != "exclude"])
    if cfg.drop_suboptimal:
        df["parsed_labels"] = df["parsed_labels"].apply(lambda xs: [x for x in xs if x != "suboptimal study"])

    if cfg.method_labels and cfg.method_labels != ["ALL"]:
        before = len(df)
        df = df[df["MethodLabel"].isin(cfg.method_labels)].copy()
        print(f"After MethodLabel filter: {len(df):,} rows; dropped {before - len(df):,}")

    if cfg.projections and cfg.projections != ["ALL"]:
        before = len(df)
        df = df[df["Projection"].isin(cfg.projections)].copy()
        print(f"After Projection filter: {len(df):,} rows; dropped {before - len(df):,}")

    df["target_labels"] = df["parsed_labels"].apply(lambda xs: filter_labels(xs, keep_set))
    before = len(df)
    df = df[df["target_labels"].apply(len) > 0].copy()
    print(f"After target-label filter: {len(df):,} rows; dropped {before - len(df):,}")

    if df.empty:
        fail("No rows remain after filtering")

    present_labels = sorted({label for labels_for_row in df["target_labels"] for label in labels_for_row})
    label_map = {label: safe_col(label) for label in present_labels}

    for label, col in label_map.items():
        df[col] = df["target_labels"].apply(lambda xs, label=label: int(label in xs))

    label_cols = list(label_map.values())

    # REMOVE RARE LABELS

    min_label_count = cfg.min_label_count

    label_counts = df[label_cols].sum()

    keep_label_cols = (
        label_counts[label_counts >= min_label_count]
        .index
        .tolist()
    )

    drop_label_cols = (
        label_counts[label_counts < min_label_count]
        .index
        .tolist()
    )

    print(f"Keeping {len(keep_label_cols)} labels")
    print(f"Dropping {len(drop_label_cols)} rare labels")

    if drop_label_cols:
        print(drop_label_cols)

    label_cols = keep_label_cols

    # remove rows that no longer contain any label
    before = len(df)

    df = df[df[label_cols].sum(axis=1) > 0].copy()

    print(f"Dropped {before - len(df)} rows with no remaining labels")

    meta_cols = [
        "ImageID", "ImageDir", "StudyDate_DICOM", "StudyID", "PatientID",
        "PatientBirth", "PatientSex_DICOM", "Projection", "MethodProjection",
        "MethodLabel", "ViewPosition_DICOM", "Modality_DICOM", "Rows_DICOM", "Columns_DICOM",
    ]
    meta_cols = [col for col in meta_cols if col in df.columns]
    df = df[meta_cols + ["target_labels"] + label_cols].copy()

    # Add a readable semicolon-separated label list for debugging.
    df["target_labels_str"] = df["target_labels"].apply(lambda xs: ";".join(xs))

    splits = patient_stratified_split(
        df=df,
        label_cols=label_cols,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        allow_single_split_classes=cfg.allow_single_split_classes,
    )

    # Ensure no patient leakage.
    train_p = set(splits["train"]["PatientID"])
    val_p = set(splits["val"]["PatientID"])
    test_p = set(splits["test"]["PatientID"])
    assert train_p.isdisjoint(val_p)
    assert train_p.isdisjoint(test_p)
    assert val_p.isdisjoint(test_p)

    # Optional image processing.
    image_output_dir = output_dir / "images"
    missing_images = []
    if not cfg.no_resize:
        image_index = build_image_index(data_root)
        all_image_ids = sorted(set(df["ImageID"].astype(str)))
        tasks = []
        for image_id in all_image_ids:
            src = image_index.get(image_id)
            if src is None:
                missing_images.append(image_id)
                continue
            if cfg.copy_original_images:
                dst = image_output_dir / image_id
            else:
                dst = image_output_dir / Path(image_id).with_suffix(".jpg").name
            if not dst.exists():
                tasks.append((src, dst, cfg.image_size, cfg.copy_original_images))

        print(f"Images found in index: {len(image_index):,}")
        print(f"Images to process: {len(tasks):,}; missing: {len(missing_images):,}")

        if tasks:
            failures = []
            with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
                for image_id, ok, err in tqdm(pool.map(resize_or_copy_one, tasks), total=len(tasks), desc="images"):
                    if not ok:
                        failures.append({"ImageID": image_id, "error": err})
            if failures:
                pd.DataFrame(failures).to_csv(output_dir / "image_processing_failures.csv", index=False)
                print(f"Image failures written: {len(failures):,}")

    if missing_images:
        pd.Series(missing_images, name="ImageID").to_csv(output_dir / "missing_images.csv", index=False)

    # Save CSVs.
    image_ext = "png" if cfg.copy_original_images else "jpg"
    for split_name, sdf in splits.items():
        out = sdf.copy()
        out.insert(
            0,
            "image_path",
            "images/" + out["ImageID"].astype(str).apply(lambda x: Path(x).with_suffix(f".{image_ext}").name),
        )
        out = out.drop(columns=["target_labels"])
        out.to_csv(output_dir / f"{split_name}.csv", index=False)
        print(f"{split_name}: {len(out):,} images, {out['PatientID'].nunique():,} patients")

    # Also save the full filtered mart before split.
    full = df.copy()
    full.insert(
        0,
        "image_path",
        "images/" + full["ImageID"].astype(str).apply(lambda x: Path(x).with_suffix(f".{image_ext}").name),
    )
    full.drop(columns=["target_labels"]).to_csv(output_dir / f"{cfg.datamart_name}_full.csv", index=False)

    # Metadata outputs.
    pd.DataFrame(
        [{"label": label, "column": col} for label, col in label_map.items()]
    ).to_csv(output_dir / "label_map.csv", index=False)

    Path(output_dir / "labels_used.txt").write_text("\n".join(present_labels), encoding="utf-8")
    Path(output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    write_split_report(output_dir, splits, label_cols)

    print(f"Done. Output directory: {output_dir}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Build PadChest data marts and patient-safe stratified splits.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--datamart-name", default="padchest_datamart")
    p.add_argument("--csv-name", default=PAD_CSV_NAME)
    p.add_argument("--labels-file", default=None)
    p.add_argument("--aliases-file", default=None)
    p.add_argument("--projections", nargs="+", default=["PA"], help="Use ALL or values like PA L AP AP-horizontal COSTAL")
    p.add_argument("--method-labels", nargs="+", default=["Physician", "RNN_model"], help="Use ALL to keep all")
    p.add_argument("--train-split", type=float, default=0.70)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_label_count", type=int, default=20, help="Minimum number of positive examples for a label to be kept")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-resize", action="store_true")
    p.add_argument("--copy-original-images", action="store_true", help="Copy original images instead of resizing to JPG")
    p.add_argument("--drop-unchanged", action="store_true", default=True)
    p.add_argument("--keep-unchanged", action="store_false", dest="drop_unchanged")
    p.add_argument("--drop-exclude", action="store_true", default=True)
    p.add_argument("--keep-exclude", action="store_false", dest="drop_exclude")
    p.add_argument("--drop-suboptimal", action="store_true", default=False)
    p.add_argument("--allow-single-split-classes", action="store_true")
    args = p.parse_args()

    return Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        datamart_name=args.datamart_name,
        csv_name=args.csv_name,
        labels_file=args.labels_file,
        aliases_file=args.aliases_file,
        projections=args.projections,
        method_labels=args.method_labels,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        image_size=args.image_size,
        num_workers=args.num_workers,
        no_resize=args.no_resize,
        copy_original_images=args.copy_original_images,
        drop_unchanged=args.drop_unchanged,
        drop_exclude=args.drop_exclude,
        drop_suboptimal=args.drop_suboptimal,
        allow_single_split_classes=args.allow_single_split_classes,
        min_label_count=args.min_label_count,
    )


if __name__ == "__main__":
    prepare_datamart(parse_args())
