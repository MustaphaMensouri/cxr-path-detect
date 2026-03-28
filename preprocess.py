"""
NIH Chest X-ray Preprocessing Script
=====================================
Resizes all images to 224x224 (ResNet standard) and produces:
  output_dir/
    images/          <- all resized PNGs (flat folder)
    train.csv
    val.csv
    test.csv

Usage:
    python preprocess_nih_xray.py \
        --data_root   /path/to/NIH_Chest_Xrays \
        --output_dir  /path/to/output \
        --val_split   0.1 \
        --num_workers 8
"""

import argparse
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ── constants ────────────────────────────────────────────────────────────────

TARGET_SIZE = (224, 224)

# Subfolders inside the dataset root that hold the raw images
IMAGE_SUBDIRS = [f"images_{i:03d}/images" for i in range(1, 13)]


# ── helpers ──────────────────────────────────────────────────────────────────

def build_image_index(data_root: Path) -> dict[str, Path]:
    """Walk all images_XXX/images sub-folders and map filename → full path."""
    index = {}
    for subdir in IMAGE_SUBDIRS:
        folder = data_root / subdir
        if not folder.exists():
            print(f"  [warn] folder not found, skipping: {folder}")
            continue
        for p in folder.glob("*.png"):
            index[p.name] = p
    return index


def resize_one(args):
    """Worker: resize a single image and save to dest_path."""
    src_path, dest_path = args
    try:
        with Image.open(src_path) as img:
            # Convert to RGB to handle any greyscale / RGBA edge cases
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(dest_path, format="PNG", optimize=True)
        return dest_path.name, None
    except Exception as exc:
        return dest_path.name, str(exc)


def parse_labels(labels_str: str) -> list[str]:
    """Split 'Finding Labels' pipe-separated string into a list."""
    return [l.strip() for l in labels_str.split("|")]


def build_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one binary column per unique finding label.
    'No Finding' stays as its own column; everything else is a pathology.
    """
    all_labels = set()
    for row in df["Finding Labels"]:
        all_labels.update(parse_labels(row))
    all_labels = sorted(all_labels)

    for label in all_labels:
        col = label.replace(" ", "_")
        df[col] = df["Finding Labels"].apply(
            lambda s: 1 if label in parse_labels(s) else 0
        )
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIH Chest X-ray preprocessor")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder of the NIH dataset (contains images_001 … images_012 and the CSV/txt files)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./nih_preprocessed",
        help="Where to write resized images and split CSVs (default: ./nih_preprocessed)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of train_val_list to use for validation (default: 0.10 → 10%%)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Parallel workers for resizing (default: 4)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images that already exist in the output folder",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    print("\n[1/5] Loading metadata …")

    labels_csv = data_root / "Data_Entry_2017.csv"
    train_val_txt = data_root / "train_val_list.txt"
    test_txt = data_root / "test_list.txt"

    for f in [labels_csv, train_val_txt, test_txt]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    df = pd.read_csv(labels_csv)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Keep only the columns we care about
    keep_cols = [
        "Image Index",
        "Finding Labels",
        "Follow-up #",
        "Patient ID",
        "Patient Age",
        "Patient Gender",
        "View Position",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Add one-hot label columns
    df = build_label_columns(df)

    print(f"   Total entries in Data_Entry_2017.csv : {len(df):,}")

    # ── 2. Load split lists ───────────────────────────────────────────────────
    print("\n[2/5] Parsing split lists …")

    train_val_files = set(train_val_txt.read_text().strip().splitlines())
    test_files = set(test_txt.read_text().strip().splitlines())

    # Sanity check: no overlap
    overlap = train_val_files & test_files
    if overlap:
        print(f"   [warn] {len(overlap)} files appear in both lists – excluding from train/val")
        train_val_files -= overlap

    print(f"   train_val pool : {len(train_val_files):,} images")
    print(f"   test pool      : {len(test_files):,} images")

    # ── 3. Split train_val → train + val (patient-aware) ─────────────────────
    print(f"\n[3/5] Splitting train/val (val_fraction={args.val_split}) …")

    # Filter df to train_val pool
    tv_df = df[df["Image Index"].isin(train_val_files)].copy()

    # Patient-level split to avoid leakage between train and val
    patient_ids = tv_df["Patient ID"].unique()
    rng = pd.np if hasattr(pd, "np") else __import__("numpy").random
    import numpy as np
    rng = np.random.default_rng(seed=42)
    rng.shuffle(patient_ids)

    n_val_patients = max(1, int(len(patient_ids) * args.val_split))
    val_patients = set(patient_ids[:n_val_patients])
    train_patients = set(patient_ids[n_val_patients:])

    train_df = tv_df[tv_df["Patient ID"].isin(train_patients)].copy()
    val_df = tv_df[tv_df["Patient ID"].isin(val_patients)].copy()
    test_df = df[df["Image Index"].isin(test_files)].copy()

    print(f"   train : {len(train_df):,} images  ({len(train_patients):,} patients)")
    print(f"   val   : {len(val_df):,} images  ({len(val_patients):,} patients)")
    print(f"   test  : {len(test_df):,} images")

    # ── 4. Resize images ──────────────────────────────────────────────────────
    print("\n[4/5] Indexing source images …")
    image_index = build_image_index(data_root)
    print(f"   Found {len(image_index):,} source images across all sub-folders")

    all_needed = (
        set(train_df["Image Index"])
        | set(val_df["Image Index"])
        | set(test_df["Image Index"])
    )
    missing = all_needed - set(image_index.keys())
    if missing:
        print(f"   [warn] {len(missing):,} images listed in CSVs/txts are missing on disk – they will be skipped")

    tasks = []
    for fname in all_needed:
        if fname not in image_index:
            continue
        dest = images_out / fname
        if args.skip_existing and dest.exists():
            continue
        tasks.append((image_index[fname], dest))

    print(f"\n   Resizing {len(tasks):,} images → {TARGET_SIZE} using {args.num_workers} workers …")
    errors = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(resize_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), unit="img"):
            fname, err = fut.result()
            if err:
                errors.append((fname, err))

    if errors:
        print(f"\n   [warn] {len(errors)} images failed to resize:")
        for fname, err in errors[:10]:
            print(f"     {fname}: {err}")

    # ── 5. Save split CSVs ────────────────────────────────────────────────────
    print("\n[5/5] Writing split CSVs …")

    # Add a convenience column with the relative path to the resized image
    for split_df, split_name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        split_df = split_df.copy()
        split_df.insert(0, "image_path", split_df["Image Index"].apply(lambda f: f"images/{f}"))
        out_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"   Saved {out_path}  ({len(split_df):,} rows)")

    print("\n✓ Done!")
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  Structure:")
    print(f"    {output_dir}/")
    print(f"    ├── images/       ({len(list(images_out.glob('*.png'))):,} resized PNGs)")
    print(f"    ├── train.csv")
    print(f"    ├── val.csv")
    print(f"    └── test.csv")


if __name__ == "__main__":
    main()