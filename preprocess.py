import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def preprocess_binary_nih(raw_dir: str, output_dir: str, csv_path: str, val_size: float = 0.1):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    df = pd.read_csv(csv_path)
    df['Binary_Label'] = df['Finding Labels'].map(lambda x: 0 if x == 'No Finding' else 1)

    for split in ['train', 'val', 'test']:
        (out_path / split).mkdir(parents=True, exist_ok=True)

    def copy_files(file_list, split_name, split_df):
        print(f"Copying {split_name}: {len(file_list)} images...")
        for fname in file_list:
            for i in range(1, 13):
                src = raw_path / f"images_{i:03d}" / "images" / fname
                if src.exists():
                    shutil.copy(src, out_path / split_name / fname)
                    break

    # Load train_val list and split it
    with open(raw_path / 'train_val_list.txt', 'r') as f:
        train_val_files = [l.strip() for l in f]

    train_files, val_files = train_test_split(train_val_files, test_size=val_size, random_state=42)

    train_df = df[df['Image Index'].isin(train_files)]
    val_df   = df[df['Image Index'].isin(val_files)]

    copy_files(train_files, 'train', train_df)
    copy_files(val_files,   'val',   val_df)

    # Test split
    with open(raw_path / 'test_list.txt', 'r') as f:
        test_files = [l.strip() for l in f]
    test_df = df[df['Image Index'].isin(test_files)]
    copy_files(test_files, 'test', test_df)

    # Save labels
    all_df = pd.concat([train_df, val_df, test_df])[['Image Index', 'Binary_Label']]
    all_df.to_csv(out_path / 'binary_labels.csv', index=False)

    print("\nDataset Balance:")
    print(all_df['Binary_Label'].value_counts())


if __name__ == "__main__":
    preprocess_binary_nih(
        raw_dir="data/raw/NIH_Chest_Xrays",
        output_dir="data/preprocessed",
        csv_path="data/raw/NIH_Chest_Xrays/Data_Entry_2017.csv"
    )