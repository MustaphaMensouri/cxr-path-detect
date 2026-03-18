import os
import shutil
import pandas as pd
from pathlib import Path

def preprocess_binary_nih(raw_dir: str, output_dir: str, csv_path: str):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    
    # Load the original metadata [cite: 10, 571]
    df = pd.read_csv(csv_path)
    
    # 1. Create Binary Label: 'Normal' vs 'Pathology'
    # In the NIH dataset, 'No Finding' indicates a normal chest X-ray [cite: 571]
    df['Binary_Label'] = df['Finding Labels'].map(lambda x: 0 if x == 'No Finding' else 1)
    
    # 2. Create simplified output directories
    for split in ['train', 'val', 'test']:
        (out_path / split).mkdir(parents=True, exist_ok=True)

    # 3. Use official lists to move files (ensures patient-level isolation) [cite: 219, 627]
    def process_split(list_filename, split_name):
        with open(raw_path / list_filename, 'r') as f:
            valid_files = set(line.strip() for line in f)
        
        # Filter dataframe for this split
        split_df = df[df['Image Index'].isin(valid_files)].copy()
        
        print(f"Processing {split_name} split: {len(split_df)} images...")
        
        for fname in split_df['Image Index']:
            # Search in the 12 subfolders [cite: 59, 135]
            for i in range(1, 13):
                src = raw_path / f"images_{i:03d}" / "images" / fname
                if src.exists():
                    shutil.copy(src, out_path / split_name / fname)
                    break
        
        return split_df

    # Process official splits [cite: 219, 626]
    # Note: 'train_val_list.txt' contains both training and validation patients
    train_val_df = process_split('train_val_list.txt', 'train')
    test_df = process_split('test_list.txt', 'test')

    # 4. Save the binary mapping file
    binary_df = pd.concat([train_val_df, test_df])[['Image Index', 'Binary_Label']]
    binary_df.to_csv(out_path / 'binary_labels.csv', index=False)
    
    # Show the class balance 
    print("\nDataset Balance:")
    print(binary_df['Binary_Label'].value_counts())

if __name__ == "__main__":
    RAW_DATA_DIR = "NIH_Chest_Xrays"
    OUTPUT_DIR = "data/binary_processed"
    CSV_FILE = "NIH_Chest_Xrays/Data_Entry_2017.csv"
    
    preprocess_binary_nih(RAW_DATA_DIR, OUTPUT_DIR, CSV_FILE)