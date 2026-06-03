# Preprocessing Documentation

This document explains how the preprocessing step converts the raw PadChest dataset into a training-ready data mart.

The preprocessing script is responsible for filtering the dataset, selecting labels, creating patient-safe train/validation/test splits, processing images, and saving the final files required by the training pipeline.

---

## 1. Purpose

The raw PadChest dataset is not used directly for training. It must first be converted into a clean data mart.

The preprocessing step creates:

* Clean split CSV files: `train.csv`, `val.csv`, and `test.csv`
* A final label list: `labels_used.txt`
* A processed image folder: `images/`
* Split statistics and label distribution reports
* A configuration file describing how the data mart was created

The main goal is to produce a reproducible dataset version that can be used by the training pipeline.

The preprocessing script supports:

* Selecting specific projection views
* Selecting target labels
* Applying optional label aliases
* Dropping unwanted labels such as `unchanged` and `exclude`
* Removing rare labels below a minimum count
* Creating patient-level train/validation/test splits
* Processing images by resizing or copying them
* Saving quality-control reports

---

## 2. Input Requirements

The preprocessing script expects the PadChest root folder to contain the main PadChest CSV file and the image files.

Expected structure:

```text
/path/to/padchest/
├── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
├── ...
└── image files
```

By default, the script looks for the CSV file named:

```text
PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
```

This can be changed using:

```bash
--csv-name
```

The CSV file must contain the following columns:

```text
ImageID
PatientID
Projection
MethodLabel
Labels
```

Other metadata columns are kept if available, such as:

```text
ImageDir
StudyDate_DICOM
StudyID
PatientBirth
PatientSex_DICOM
MethodProjection
ViewPosition_DICOM
Modality_DICOM
Rows_DICOM
Columns_DICOM
```

---

## 3. Preprocessing Workflow

The preprocessing pipeline follows these main steps:

```text
Raw PadChest CSV + images
        ↓
Load CSV
        ↓
Parse report-derived labels
        ↓
Apply optional label aliases
        ↓
Drop unwanted labels
        ↓
Filter by annotation method
        ↓
Filter by projection/view
        ↓
Keep only target labels
        ↓
Remove rare labels
        ↓
Remove rows with no remaining labels
        ↓
Create patient-level train/validation/test split
        ↓
Resize or copy images
        ↓
Save final data mart files
```

The output is a complete data mart that can be passed directly to the training code.

---

## 4. Label Selection and Rare Label Filtering

### Target labels

The script can use labels in two ways.

If a labels file is provided using:

```bash
--labels-file
```

then the script reads the target labels from that file.

The labels file should contain one label per line:

```text
normal
infiltrates
pleural effusion
pneumonia
```

Blank lines and lines starting with `#` are ignored.

If no labels file is provided, the script uses the default lung-related label list defined inside `preprocess.py`.

### Label normalization

Labels are normalized before being used. The script:

* Converts labels to lowercase
* Strips extra spaces
* Converts label names to safe column names

For example:

```text
pleural effusion → pleural_effusion
COPD signs → copd_signs
apical pleural thickening → apical_pleural_thickening
```

### Optional aliases

An optional alias file can be provided using:

```bash
--aliases-file
```

The aliases file must be a CSV file with two columns:

```text
source,target
```

Example:

```csv
source,target
COPD signs,emphysema
pulmonary artery hypertension,pulmonary hypertension
```

This is useful when two labels should be merged or normalized to the same target label.

### Dropping unwanted labels

By default, the script drops:

```text
unchanged
exclude
```

The label `suboptimal study` is not dropped by default unless this option is used:

```bash
--drop-suboptimal
```

Available options:

```bash
--drop-unchanged
--keep-unchanged
--drop-exclude
--keep-exclude
--drop-suboptimal
```

### Rare label filtering

After target labels are selected, the script removes labels with too few positive examples.

The minimum count is controlled by:

```bash
--min-label-count
```

Default value:

```bash
--min-label-count 20
```

Labels with fewer than this number of positive images are removed.

After rare labels are removed, rows that no longer contain any remaining positive label are also removed.

The final label list is saved in:

```text
labels_used.txt
```

This file is very important because the training code uses it to define the model output labels.

---

## 5. Projection Filtering

PadChest contains multiple projection views. The preprocessing script can create different data marts using different projections.

Projection filtering is controlled by:

```bash
--projections
```

Examples:

```bash
--projections PA
```

```bash
--projections PA AP
```

```bash
--projections PA L
```

```bash
--projections PA L AP
```

```bash
--projections ALL
```

The default projection is:

```text
PA
```

Projection filtering is important because different views can have different visual patterns. For example, PA, AP, and lateral images are not identical from a modeling point of view.

Creating separate data marts makes it easier to compare experiments such as:

* PA-only training
* PA + AP training
* PA + lateral training
* All-projection training

---

## 6. Patient-level Train/Validation/Test Split

The script creates train, validation, and test splits at the patient level.

This means that images from the same patient are kept in only one split.

This is important because PadChest can contain multiple images or studies for the same patient. If the same patient appears in both training and testing, the evaluation may be too optimistic because of patient leakage.

### Split ratios

The script uses:

```bash
--train-split
--val-split
```

The test split is calculated automatically:

```text
test_split = 1.0 - train_split - val_split
```

Default values in the script:

```text
train = 0.70
validation = 0.15
test = 0.15
```

For the project setup using 70% train, 10% validation, and 20% test, use:

```bash
--train-split 0.70 --val-split 0.10
```

### Stratification logic

The split is patient-level and rare-label-aware.

The script first creates a patient-level label matrix. For each patient, it checks which labels appear in any image belonging to that patient.

Then, each patient receives a stratification key based on the rarest positive label present for that patient.

This helps preserve the distribution of rare labels across the splits as much as possible.

### Leakage check

After splitting, the script checks that patients do not overlap between:

* train and validation
* train and test
* validation and test

If patient overlap exists, the assertion fails.

---

## 7. Image Processing

The script can either resize images, copy original images, or skip image processing depending on the options used.

### Default behavior

By default, the script:

* Searches for images inside the PadChest data root
* Reads each image
* Normalizes pixel values to the range 0–1
* Converts the image to 8-bit
* Resizes it to the selected image size
* Saves it as `.jpg` inside the output `images/` folder

Default image size:

```text
224 × 224
```

This can be changed using:

```bash
--image-size
```

Example:

```bash
--image-size 320
```

### Copy original images

To copy original images instead of resizing them, use:

```bash
--copy-original-images
```

In this mode, images are copied to the output folder without resizing.

### No resize mode

The option:

```bash
--no-resize
```

skips the image processing step.

Important note: in the current implementation, the final split CSVs are filtered by checking whether the expected image files exist inside the output `images/` folder. Therefore, `--no-resize` should only be used if the expected image files are already present in the output data mart.

If the images do not exist, many or all rows may be removed from the final split CSVs.

### Missing images

If an image cannot be found, it is recorded in:

```text
missing_images.csv
```

If image reading or writing fails, the error is recorded in:

```text
image_processing_failures.csv
```

---

## 8. Output Data Mart Structure

After preprocessing, the output folder has the following structure:

```text
data_marts/<data_mart_name>/
├── images/
├── train.csv
├── val.csv
├── test.csv
├── labels_used.txt
├── label_map.csv
├── split_summary.csv
├── label_distribution_by_split.csv
├── config.json
├── missing_images.csv
├── image_processing_failures.csv
└── <data_mart_name>_full.csv
```

Some files are created only when needed. For example, `missing_images.csv` is created only if missing images are found.

### Main output files

| File                              | Description                                              |
| --------------------------------- | -------------------------------------------------------- |
| `images/`                         | Folder containing resized or copied images               |
| `train.csv`                       | Training split                                           |
| `val.csv`                         | Validation split                                         |
| `test.csv`                        | Test split                                               |
| `labels_used.txt`                 | Final label columns used by the model                    |
| `label_map.csv`                   | Mapping between original labels and safe column names    |
| `split_summary.csv`               | Number of images, patients, and studies per split        |
| `label_distribution_by_split.csv` | Per-label image and patient counts for each split        |
| `config.json`                     | Preprocessing configuration used to create the data mart |
| `<data_mart_name>_full.csv`       | Full filtered dataset before splitting                   |
| `missing_images.csv`              | List of image IDs that were not found                    |
| `image_processing_failures.csv`   | Image processing errors, if any                          |

### CSV format

Each split CSV contains:

* `image_path`
* metadata columns
* `target_labels_str`
* one binary column per final label

Example label columns:

```text
normal
pleural_effusion
pneumonia
atelectasis
chronic_changes
```

The training pipeline reads `image_path` and the label columns from these CSV files.

---

## 9. Example Commands

### PA-only data mart

```bash
python preprocess.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_PA \
  --datamart-name lung_PA \
  --projections PA \
  --labels-file lung_labels.txt \
  --train-split 0.70 \
  --val-split 0.10 \
  --min-label-count 20 \
  --image-size 224
```

This creates a PA-only data mart with:

```text
70% train
10% validation
20% test
```

### PA + AP data mart

```bash
python preprocess.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_PA_AP \
  --datamart-name lung_PA_AP \
  --projections PA AP \
  --labels-file lung_labels.txt \
  --train-split 0.70 \
  --val-split 0.10 \
  --min-label-count 20 \
  --image-size 224
```

### PA + lateral data mart

```bash
python preprocess.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_PA_L \
  --datamart-name lung_PA_L \
  --projections PA L \
  --labels-file lung_labels.txt \
  --train-split 0.70 \
  --val-split 0.10 \
  --min-label-count 20 \
  --image-size 224
```

### All projections

```bash
python preprocess.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_all \
  --datamart-name lung_all \
  --projections ALL \
  --labels-file lung_labels.txt \
  --train-split 0.70 \
  --val-split 0.10 \
  --min-label-count 20 \
  --image-size 224
```

### Use only manually annotated labels

```bash
python preprocess.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_PA_physician \
  --datamart-name lung_PA_physician \
  --projections PA \
  --method-labels Physician \
  --labels-file lung_labels.txt \
  --train-split 0.70 \
  --val-split 0.10 \
  --min-label-count 20 \
  --image-size 224
```

### Keep all annotation methods

```bash
python preprocess.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_PA_all_methods \
  --datamart-name lung_PA_all_methods \
  --projections PA \
  --method-labels ALL \
  --labels-file lung_labels.txt \
  --train-split 0.70 \
  --val-split 0.10 \
  --min-label-count 20 \
  --image-size 224
```

---

## 10. Verification and Quality Checks

After creating a data mart, check the generated reports before training.

### Check split sizes

Open:

```text
split_summary.csv
```

Verify:

* Number of images in train, validation, and test
* Number of patients in each split
* Number of studies in each split

### Check label distribution

Open:

```text
label_distribution_by_split.csv
```

Verify:

* Rare labels are still present
* Label distribution is not extremely different between splits
* No important label disappeared from validation or test

### Check final labels

Open:

```text
labels_used.txt
```

Verify that the expected labels are present.

This file must match the label columns used by `train.csv`, `val.csv`, and `test.csv`.

### Check missing images

If this file exists:

```text
missing_images.csv
```

review which image IDs were not found.

If this file exists:

```text
image_processing_failures.csv
```

review which images failed during reading, resizing, or writing.

### Check CSV image paths

Each split CSV contains an `image_path` column.

Example:

```text
images/example_image.jpg
```

Verify that the file exists relative to the data mart folder:

```text
data_marts/<data_mart_name>/images/example_image.jpg
```

---

## 11. Common Issues

### 1. No rows remain after filtering

This can happen if:

* The selected projections do not match the dataset values
* The labels file contains labels that are not present in the dataset
* `--min-label-count` is too high
* Too many labels are dropped

Possible fixes:

* Check the spelling of labels in the labels file
* Use `--projections ALL` to test if projection filtering is the issue
* Lower `--min-label-count`
* Check that the `Labels` column is not empty

### 2. Many labels are dropped

This happens when labels have fewer positive examples than:

```bash
--min-label-count
```

Possible fixes:

* Lower `--min-label-count`
* Use more projections
* Use a larger dataset subset
* Merge similar labels using an aliases file

### 3. Validation or test has very few examples for rare labels

This can happen even with stratification if a label has very few patients.

Possible fixes:

* Use a larger dataset
* Lower the number of target labels
* Merge related labels
* Use a higher-level label hierarchy
* Avoid evaluating very rare labels independently

### 4. Split cannot be stratified

The script may fail if some stratification classes have too few patients.

Possible fix:

```bash
--allow-single-split-classes
```

This allows the script to fall back to a non-stratified split for that stage when stratification is impossible.

### 5. Missing images are removed from the final CSVs

The script checks whether the expected image files exist. If an image is missing, the row is removed from the final split CSV.

Check:

```text
missing_images.csv
```

### 6. Using `--no-resize` removes many rows

In the current implementation, `--no-resize` skips image creation but the final CSVs still check whether the expected files exist in the output `images/` folder.

Use `--no-resize` only when the expected image files already exist in the output data mart.

### 7. Output labels are safe column names

Original labels are converted into safe column names.

For example:

```text
pleural effusion → pleural_effusion
apical pleural thickening → apical_pleural_thickening
COPD signs → copd_signs
```

The final model uses the safe column names stored in:

```text
labels_used.txt
```

### 8. The same study may contain multiple images

PadChest reports can correspond to studies with multiple images. The same report labels may be assigned to multiple views from the same study.

This is one reason why patient-level splitting is important.
