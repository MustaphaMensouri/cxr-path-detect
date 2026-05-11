"""
prepare_padchest.py
====================
Preprocess the PadChest dataset: filter labels / projection / method,
resize images, and produce patient-stratified train / val / test CSVs.

Usage
-----
python prepare_padchest.py \
    --data_root      /path/to/padchest \
    --output_dir     ./data/padchest_preprocessed \
    --labels_file    ./lung_labels.txt \
    --projections    PA L \
    --method_labels  Physician RNN_model \
    --train_split    0.70 \
    --val_split      0.15 \
    --image_size     224 \
    --num_workers    8

Labels file format  (one label per line, lines starting with # are ignored):
    normal
    infiltrates
    pleural effusion
    ...

data_root layout expected
--------------------------
<data_root>/
    PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
    0/   (or 0.zip unpacked)  ← zip sub-folders with *.png images
    1/
    ...
    (images live inside numbered sub-folders, each named like the ImageDir field)
"""

import argparse
import ast
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ─────────────────────────── default label set ────────────────────────────
DEFAULT_LUNG_LABELS = [
    # infiltrates / patterns
    "infiltrates", "alveolar pattern", "interstitial pattern",
    "ground glass pattern", "reticular interstitial pattern",
    "reticulonodular interstitial pattern", "miliary opacities",
    "consolidation", "increased density", "air bronchogram",
    # atelectasis
    "atelectasis", "laminar atelectasis", "lobar atelectasis",
    "segmental atelectasis", "total atelectasis", "round atelectasis",
    "atelectasis basal", "volume loss", "hypoexpansion",
    # nodules / masses
    "nodule", "multiple nodules", "pseudonodule", "mass", "pulmonary mass",
    # pleural
    "pleural effusion", "loculated pleural effusion", "pleural thickening",
    "apical pleural thickening", "pleural plaques",
    "costophrenic angle blunting", "hydropneumothorax", "empyema",
    "pleural mass",
    # air / pneumothorax
    "pneumothorax", "pneumomediastinum", "pneumoperitoneo",
    "subcutaneous emphysema", "air trapping", "bullas",
    "hyperinflated lung", "flattened diaphragm",
    # chronic / fibrotic
    "chronic changes", "pulmonary fibrosis", "fibrotic band",
    "post radiotherapy changes",
    # infectious
    "cavitation", "abscess", "tuberculosis", "tuberculosis sequelae",
    "pneumonia", "atypical pneumonia",
    # vascular
    "vascular redistribution", "hilar enlargement",
    "vascular hilar enlargement", "pulmonary artery enlargement",
    "hilar congestion", "pulmonary hypertension",
    "pulmonary artery hypertension", "pulmonary venous hypertension",
    # edema / airways
    "pulmonary edema", "kerley lines", "bronchiectasis", "tracheal shift",
    # other lung pathology
    "lung metastasis", "lymphangitis carcinomatosa", "lepidic adenocarcinoma",
    "emphysema", "COPD signs", "respiratory distress", "asbestosis signs",
    "surgery lung",
    # context / normal
    "normal", "cardiomegaly",
]

# ──────────────────────────── helpers ─────────────────────────────────────

def load_labels_file(path: Path) -> list[str]:
    """Read one label per line; skip blank lines and comments (#)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def parse_labels(raw) -> list[str]:
    """Safely parse a string-encoded Python list from the CSV."""
    if not isinstance(raw, str) or pd.isna(raw):
        return []
    try:
        result = ast.literal_eval(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def filter_row_labels(labels_list: list, keep_set: set) -> list[str]:
    """Return only labels that are in keep_set."""
    return [l for l in labels_list if l in keep_set]


def resize_one(args):
    src, dst = args
    try:
        dst = dst.with_suffix(".jpg")
        with Image.open(src) as img:
            img.convert("RGB").resize((224, 224), Image.LANCZOS).save(dst, format="JPEG", quality=95)
    except Exception as e:
        print(f"[warn] {src.name}: {e}")


def find_image_index(data_root: Path) -> dict[str, Path]:
    """
    Build a filename→Path mapping by scanning all numbered sub-directories.
    PadChest images live in zip-extracted folders named 0, 1, 2 … 54.
    Each folder may have images directly or in a nested sub-folder.
    """
    index: dict[str, Path] = {}
    # numbered zip sub-dirs (0/ … 54/) and any flat images
    patterns = ["**/*.png", "**/*.PNG"]
    for pattern in patterns:
        for p in data_root.glob(pattern):
            index[p.name] = p
    return index


# ──────────────────────────── main ────────────────────────────────────────

def main(
    data_root: str,
    output_dir: str,
    labels_file: str | None,
    projections: list[str],
    method_labels: list[str],
    train_split: float,
    val_split: float,
    image_size: int,
    num_workers: int,
    seed: int,
    no_resize: bool,
):
    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    img_out    = output_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # ── validate splits ───────────────────────────────────────────────────
    test_split = round(1.0 - train_split - val_split, 10)
    if test_split < 0:
        sys.exit("ERROR: train_split + val_split > 1.0")
    print(f"Splits → train={train_split:.0%}  val={val_split:.0%}  test={test_split:.0%}")

    # ── load keep-set ─────────────────────────────────────────────────────
    if labels_file:
        keep_labels = load_labels_file(Path(labels_file))
        print(f"Labels from file ({len(keep_labels)}): {keep_labels[:5]} …")
    else:
        keep_labels = DEFAULT_LUNG_LABELS
        print(f"Using default lung label set ({len(keep_labels)} labels).")
    keep_set = set(keep_labels)

    # ── load CSV ──────────────────────────────────────────────────────────
    csv_path = data_root / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Rows: {len(df):,}")

    # ── drop missing labels ───────────────────────────────────────────────
    df = df[df["Labels"].notna()].copy()

    # ── parse labels ──────────────────────────────────────────────────────
    df["parsed_labels"] = df["Labels"].apply(parse_labels)

    # ── filter by MethodLabel ─────────────────────────────────────────────
    if method_labels:
        before = len(df)
        df = df[df["MethodLabel"].isin(method_labels)]
        print(f"  After MethodLabel filter {method_labels}: {len(df):,}  (dropped {before-len(df):,})")

    # ── filter by Projection ──────────────────────────────────────────────
    if projections:
        before = len(df)
        df = df[df["Projection"].isin(projections)]
        print(f"  After Projection filter {projections}: {len(df):,}  (dropped {before-len(df):,})")

    # ── apply lung-label filter ───────────────────────────────────────────
    df["lung_labels"] = df["parsed_labels"].apply(
        lambda labs: filter_row_labels(labs, keep_set)
    )
    before = len(df)
    df = df[df["lung_labels"].apply(len) > 0].copy()
    print(f"  After lung-label filter: {len(df):,}  (dropped {before-len(df):,})")

    if df.empty:
        sys.exit("ERROR: No rows remaining after filtering. Check your labels file / filters.")

    # ── build binary label columns ────────────────────────────────────────
    # Use only the labels that actually appear after filtering
    present_labels = sorted({l for labs in df["lung_labels"] for l in labs})
    print(f"  Distinct labels present: {len(present_labels)}")
    for lbl in present_labels:
        safe = lbl.replace(" ", "_")
        df[safe] = df["lung_labels"].apply(lambda x, l=lbl: int(l in x))

    label_cols = [l.replace(" ", "_") for l in present_labels]

    # ── select output columns ─────────────────────────────────────────────
    meta_cols = [
        "ImageID", "ImageDir", "StudyDate_DICOM", "StudyID", "PatientID",
        "PatientBirth", "PatientSex_DICOM", "Projection", "MethodLabel",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    df = df[meta_cols + label_cols].copy()

    # ── patient-stratified splits ─────────────────────────────────────────
    patients = df["PatientID"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n      = len(patients)
    n_val  = int(n * val_split)
    n_test = int(n * test_split)

    val_patients   = set(patients[:n_val])
    test_patients  = set(patients[n_val : n_val + n_test])
    train_patients = set(patients[n_val + n_test:])

    splits = {
        "train": df[df["PatientID"].isin(train_patients)],
        "val":   df[df["PatientID"].isin(val_patients)],
        "test":  df[df["PatientID"].isin(test_patients)],
    }

    print("\nSplit sizes:")
    for name, sdf in splits.items():
        n_normal = int(sdf["normal"].sum()) if "normal" in sdf.columns else 0
        print(f"  {name:5s}: {len(sdf):>7,} images  |  {sdf['PatientID'].nunique():>5,} patients"
              f"  |  {n_normal:>6,} normal")

    # ── resize images ─────────────────────────────────────────────────────
    if not no_resize:
        src_index = find_image_index(data_root)
        print(f"  Found {len(src_index):,} PNG files on disk.")

        all_ids = set().union(*[set(s["ImageID"]) for s in splits.values()])
        tasks = [
            (src_index[fid], img_out / fid)
            for fid in all_ids
            if fid in src_index and not (img_out / Path(fid).with_suffix(".jpg").name).exists()
        ]
        print(f"  Images to resize: {len(tasks):,}  (size={image_size}×{image_size})")
        if tasks:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                list(tqdm(pool.map(resize_one, tasks),
                          total=len(tasks), desc="resizing"))
        else:
            print("  Nothing to resize (all already exist).")
    else:
        print("\n--no_resize set: skipping image resizing.")

    # ── save CSVs ─────────────────────────────────────────────────────────
    print()
    for name, sdf in splits.items():
        out = sdf.copy()
        out.insert(0, "image_path", "images/" + out["ImageID"].astype(str).str.replace(".png", ".jpg", regex=False))
        out_path = output_dir / f"{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"  {name}.csv → {len(out):,} rows  [{out_path}]")

    # ── save label list used ──────────────────────────────────────────────
    label_list_path = output_dir / "labels_used.txt"
    label_list_path.write_text("\n".join(present_labels), encoding="utf-8")
    print(f"\n  labels_used.txt → {len(present_labels)} labels  [{label_list_path}]")
    print("\nDone.")
# ──────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocess PadChest: filter, resize, split by patient."
    )
    p.add_argument(
        "--data_root", required=True,
        help="Root directory containing the PadChest CSV and image sub-folders.",
    )
    p.add_argument(
        "--output_dir", default="./data/padchest_preprocessed",
        help="Where to write images/ and the split CSVs.",
    )
    p.add_argument(
        "--labels_file", default=None,
        help="Path to a .txt file with one label per line to keep. "
             "Lines starting with # are comments. "
             "If omitted, a built-in lung-label set is used.",
    )
    p.add_argument(
        "--projections", nargs="*", default=["PA"],
        help="Projection types to keep (e.g. PA L AP). Default: PA only.",
    )
    p.add_argument(
        "--method_labels", nargs="*", default=["Physician", "RNN_model"],
        help="MethodLabel values to keep. Default: Physician RNN_model.",
    )
    p.add_argument(
        "--train_split", type=float, default=0.70,
        help="Fraction of patients for training (default 0.70).",
    )
    p.add_argument(
        "--val_split", type=float, default=0.15,
        help="Fraction of patients for validation (default 0.15). "
             "test = 1 - train - val.",
    )
    p.add_argument(
        "--image_size", type=int, default=224,
        help="Output image size in pixels (square). Default: 224.",
    )
    p.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel workers for resizing. Default: 4.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits. Default: 42.",
    )
    p.add_argument(
        "--no_resize", action="store_true",
        help="Skip image resizing (useful when images are already preprocessed).",
    )
    args = p.parse_args()
    main(
        data_root=args.data_root,
        output_dir=args.output_dir,
        labels_file=args.labels_file,
        projections=args.projections,
        method_labels=args.method_labels,
        train_split=args.train_split,
        val_split=args.val_split,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        no_resize=args.no_resize,
    )