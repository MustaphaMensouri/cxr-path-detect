import argparse
import os
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── silence noisy TF logs before import ──────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input  # adjust if your model uses a different backbone


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_gamma_lut(gamma: float = 1.5) -> np.ndarray:
    """Pre-compute an 8-bit gamma look-up table (applied *before* model normalisation)."""
    inv_gamma = 1.0 / gamma
    return np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )


def apply_gamma(rgb_uint8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(rgb_uint8, lut)


def load_as_rgb(img_path: str, seg_size: int) -> np.ndarray:
    """
    Read an image (any channel count), convert to single-channel, resize to
    seg_size × seg_size, then convert to RGB uint8 — matching the original
    segment_image() pipeline.
    """
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=1, expand_animations=False)
    img = tf.image.resize(img, (seg_size, seg_size))
    img_np = img.numpy().astype(np.uint8)           # (H, W, 1)
    return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)  # (H, W, 3)


def predict_mask(model, rgb_gamma: np.ndarray) -> np.ndarray:
    """Run the U-Net and return a binary uint8 mask (H, W)."""
    tensor = preprocess_input(tf.cast(rgb_gamma, tf.float32))
    tensor = tf.expand_dims(tensor, axis=0)          # (1, H, W, 3)
    probs  = model.predict(tensor, verbose=0)[0]     # (H, W, 1) or (H, W)
    mask   = (probs > 0.5).astype(np.uint8)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)             # (H, W)
    return mask


def refine_mask(mask: np.ndarray, dilation_kernel: int = 30) -> np.ndarray:
    """
    Dilate → find two largest contours → convex hull → filled mask.
    Mirrors the original ROI extraction logic exactly.
    """
    kernel  = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: return the dilated mask as-is
        return dilated

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    combined = np.vstack(contours)
    hull     = cv2.convexHull(combined)

    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED)
    return hull_mask


def extract_roi(rgb_gamma: np.ndarray, hull_mask: np.ndarray) -> np.ndarray:
    """Multiply the gamma-corrected image by the hull mask to black-out background."""
    mask_3d = hull_mask[:, :, np.newaxis]            # broadcast over channels
    roi     = rgb_gamma * mask_3d
    if roi.max() <= 1.0:                             # safety: scale to [0,255]
        roi = (roi * 255)
    return roi.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image worker
# ─────────────────────────────────────────────────────────────────────────────

def process_one(
    src_path:   Path,
    dst_path:   Path,
    model,
    gamma_lut:  np.ndarray,
    seg_size:   int,
    jpeg_quality: int,
) -> bool:
    """
    Full pipeline for a single image.
    Returns True on success, False on failure.
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load → single-channel → RGB
        rgb = load_as_rgb(str(src_path), seg_size)

        # 2. Gamma correction
        rgb_gamma = apply_gamma(rgb, gamma_lut)

        # 3. Segmentation
        raw_mask  = predict_mask(model, rgb_gamma)

        # 4. Refine mask → convex hull
        hull_mask = refine_mask(raw_mask)

        # 5. ROI extraction
        roi_rgb   = extract_roi(rgb_gamma, hull_mask)

        # 6. Save as JPEG (OpenCV expects BGR)
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(dst_path),
            roi_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        return True

    except Exception as exc:
        warnings.warn(f"[warn] {src_path.name}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level processing
# ─────────────────────────────────────────────────────────────────────────────

def process_split(
    csv_path:    Path,
    prep_dir:    Path,
    out_dir:     Path,
    model,
    gamma_lut:   np.ndarray,
    seg_size:    int,
    jpeg_quality: int,
    num_workers: int,
) -> None:
    split_name = csv_path.stem                       # "train" / "val" / "test"
    df         = pd.read_csv(csv_path)

    # Build source → dest path mapping
    # The stage-1 script stores "images/<filename>.png" in the image_path column.
    records = []
    for _, row in df.iterrows():
        src = prep_dir / row["image_path"]           # e.g.  preprocessed/images/00001.png
        dst_name = src.stem + ".jpg"
        dst = out_dir / "images" / dst_name          # e.g.  segmented/images/00001.jpg
        records.append((src, dst))

    # ── parallel inference using a thread pool ──────────────────────────────
    # TensorFlow is thread-safe for inference; ProcessPool would require
    # re-loading the model in every worker, which is expensive.
    successes = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                process_one,
                src, dst, model, gamma_lut, seg_size, jpeg_quality,
            ): (src, dst)
            for src, dst in records
        }
        for future in tqdm(futures, total=len(futures), desc=f"{split_name}"):
            src, dst = futures[future]
            ok = future.result()
            successes.append((row, dst, ok) for row, (_, _) in [(df.iloc[list(futures.keys()).index(future)], (src, dst))])

    # ── rebuild CSV with updated image_path column (JPEG, relative) ─────────
    out_df = df.copy()
    out_df["image_path"] = out_df["image_path"].apply(
        lambda p: "images/" + Path(p).stem + ".jpg"
    )

    # Drop rows whose image failed to process
    failed_stems = {
        Path(dst).stem
        for (src, dst), future in zip(records, futures)
        if not future.result()   # already resolved; result() is cached
    }
    if failed_stems:
        before = len(out_df)
        out_df = out_df[~out_df["image_path"].apply(lambda p: Path(p).stem).isin(failed_stems)]
        print(f"  [{split_name}] dropped {before - len(out_df)} failed images")

    out_csv = out_dir / f"{split_name}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"  [{split_name}] saved {len(out_df):,} rows → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    preprocessed_dir: str,
    output_dir:       str,
    model_path:       str,
    gamma:            float,
    seg_size:         int,
    jpeg_quality:     int,
    num_workers:      int,
    splits:           list[str],
) -> None:
    prep_dir = Path(preprocessed_dir)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    gamma_lut = build_gamma_lut(gamma)

    for split in splits:
        csv_path = prep_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue
        print(f"\nProcessing split: {split}  ({csv_path})")
        process_split(
            csv_path    = csv_path,
            prep_dir    = prep_dir,
            out_dir     = out_dir,
            model       = model,
            gamma_lut   = gamma_lut,
            seg_size    = seg_size,
            jpeg_quality= jpeg_quality,
            num_workers = num_workers,
        )

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gamma + segmentation + ROI extraction pipeline")
    p.add_argument("--preprocessed_dir", required=True,
                   help="Output directory of stage-1 preprocessing script")
    p.add_argument("--output_dir",       default="./data/segmented",
                   help="Where to write JPEG ROIs and updated CSVs")
    p.add_argument("--model_path",       required=True,
                   help="Path to the .keras U-Net segmentation model")
    p.add_argument("--gamma",            type=float, default=1.5,
                   help="Gamma correction exponent (default 1.5)")
    p.add_argument("--seg_size",         type=int,   default=512,
                   help="Image size fed to the segmentation model (default 512)")
    p.add_argument("--jpeg_quality",     type=int,   default=95,
                   help="JPEG compression quality 0-100 (default 95)")
    p.add_argument("--num_workers",      type=int,   default=4,
                   help="Parallel threads for inference (default 4)")
    p.add_argument("--splits",           nargs="+",  default=["train", "val", "test"],
                   help="Which splits to process (default: train val test)")
    args = p.parse_args()

    main(
        preprocessed_dir = args.preprocessed_dir,
        output_dir       = args.output_dir,
        model_path       = args.model_path,
        gamma            = args.gamma,
        seg_size         = args.seg_size,
        jpeg_quality     = args.jpeg_quality,
        num_workers      = args.num_workers,
        splits           = args.splits,
    )