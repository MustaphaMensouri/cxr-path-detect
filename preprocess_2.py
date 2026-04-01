import argparse
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── silence noisy TF logs ──────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_gamma_lut(gamma: float = 1.5) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    return np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )

def apply_gamma(rgb_uint8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(rgb_uint8, lut)

def load_as_rgb(img_path: str, seg_size: int) -> np.ndarray:
    """Reads image and upscales to seg_size (e.g. 512) for the U-Net."""
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=1, expand_animations=False)
    img = tf.image.resize(img, (seg_size, seg_size))
    img_np = img.numpy().astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

def predict_mask(model, rgb_gamma: np.ndarray) -> np.ndarray:
    tensor = preprocess_input(tf.cast(rgb_gamma, tf.float32))
    tensor = tf.expand_dims(tensor, axis=0)
    probs  = model.predict(tensor, verbose=0)[0]
    mask   = (probs > 0.5).astype(np.uint8)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)
    return mask

def refine_mask(mask: np.ndarray, dilation_kernel: int = 30) -> np.ndarray:
    kernel  = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dilated
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    combined = np.vstack(contours)
    hull     = cv2.convexHull(combined)
    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED)
    return hull_mask

def extract_roi(rgb_gamma: np.ndarray, hull_mask: np.ndarray, final_size: int = 224) -> np.ndarray:
    """Masks the image at 512x512 then downsamples to final_size (224)."""
    mask_3d = hull_mask[:, :, np.newaxis]
    roi = (rgb_gamma * mask_3d).astype(np.uint8)
    # LANCZOS4 is high-quality for downsampling
    return cv2.resize(roi, (final_size, final_size), interpolation=cv2.INTER_LANCZOS4)

# ─────────────────────────────────────────────────────────────────────────────
# Processing Logic
# ─────────────────────────────────────────────────────────────────────────────

def process_one(src_path, dst_path, model, gamma_lut, seg_size, jpeg_quality, final_size) -> bool:
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # 1. Load & Upscale to 512
        rgb = load_as_rgb(str(src_path), seg_size)
        # 2. Enhance
        rgb_gamma = apply_gamma(rgb, gamma_lut)
        # 3. Segment at 512
        raw_mask  = predict_mask(model, rgb_gamma)
        hull_mask = refine_mask(raw_mask)
        # 4. Extract & Downscale to final_size (224)
        roi_rgb   = extract_roi(rgb_gamma, hull_mask, final_size=final_size)
        # 5. Save
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst_path), roi_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return True
    except Exception as exc:
        warnings.warn(f"[warn] {src_path.name}: {exc}")
        return False

def process_split(csv_path, prep_dir, out_dir, model, gamma_lut, seg_size, jpeg_quality, final_size):
    split_name = csv_path.stem
    df = pd.read_csv(csv_path)
    
    # Map input PNGs to output JPEGs
    records = []
    for _, row in df.iterrows():
        src = prep_dir / row["image_path"]
        dst = out_dir / "images" / (Path(row["image_path"]).stem + ".jpg")
        records.append((src, dst))

    failed_stems = set()
    for src, dst in tqdm(records, desc=f"Processing {split_name}"):
        ok = process_one(src, dst, model, gamma_lut, seg_size, jpeg_quality, final_size)
        if not ok:
            failed_stems.add(src.stem)

    # ── Update and Clean CSV ──
    out_df = df.copy()
    out_df["image_path"] = out_df["image_path"].apply(
        lambda p: "images/" + Path(p).stem + ".jpg"
    )
    
    if failed_stems:
        before = len(out_df)
        out_df = out_df[~out_df["image_path"].apply(lambda p: Path(p).stem).isin(failed_stems)]
        print(f"  [{split_name}] dropped {before - len(out_df)} failed images")

    out_csv = out_dir / f"{split_name}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"  [{split_name}] saved {len(out_df):,} rows -> {out_csv}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--output_dir", default="./data/segmented")
    p.add_argument("--model_path", required=True)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--seg_size", type=int, default=512)
    p.add_argument("--final_size", type=int, default=224) 
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    prep_dir = Path(args.preprocessed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    gamma_lut = build_gamma_lut(args.gamma)

    for split in args.splits:
        csv_path = prep_dir / f"{split}.csv"
        if csv_path.exists():
            process_split(csv_path, prep_dir, out_dir, model, gamma_lut, 
                          args.seg_size, args.jpeg_quality, args.final_size)

    print("\nDone.")

if __name__ == "__main__":
    main()